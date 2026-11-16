import io
import json

import numpy as np
import pytest
import soundfile as sf

from mara_bridge import character
from mara_bridge import server as br
from mixmatch.sim import demo_project, tap


def _bridge(tmp_path):
    song, hub, env, live = demo_project()
    for t in [*song.tracks, song.master_track]:
        t.devices.append(tap())
    return br.Bridge(live=live, hub=hub, env=env, state_dir=tmp_path / "bridge"), song


def _song(path, f0=55.0, decay=0.15, click=0.3, bass=0.15, gain=0.5, seconds=12, sr=44100, seed=0):
    """Four-on-the-floor kick (pitch sweep to f0), sidechained 45 Hz bass, pink noise."""
    rng = np.random.default_rng(seed)
    n, t = int(seconds * sr), np.arange(int(seconds * sr)) / sr
    tk = np.arange(int(0.5 * sr)) / sr
    kick = np.sin(2 * np.pi * np.cumsum(f0 + 2 * f0 * np.exp(-tk / 0.02)) / sr) * np.exp(-tk / decay)
    kick[:int(0.004 * sr)] += click * rng.standard_normal(int(0.004 * sr))
    x = np.zeros(n)
    for s in np.arange(0, seconds - 0.5, 0.5):
        i = int(s * sr)
        x[i:i + len(kick)] += kick
    x += bass * np.sin(2 * np.pi * 45 * t) * np.minimum(1, (t % 0.5) / 0.15)
    f = np.fft.rfftfreq(n, 1 / sr)
    noise = np.fft.irfft(np.fft.rfft(rng.standard_normal(n)) / np.sqrt(np.maximum(f, 20)), n=n)
    x += noise * 0.08 / np.std(noise)
    x *= gain / np.max(np.abs(x))
    sf.write(path, np.stack([x, x], 1).astype(np.float32), sr)
    return str(path)


def _rpc(server, method, params=None, mid=1):
    return server.handle({"jsonrpc": "2.0", "id": mid, "method": method, "params": params or {}})


def _call(server, name, **args):
    result = _rpc(server, "tools/call", {"name": name, "arguments": args})["result"]
    return json.loads(result["content"][0]["text"]), result["isError"]


def test_protocol_handshake_and_tool_list(tmp_path):
    server = br.Server(_bridge(tmp_path)[0])
    init = _rpc(server, "initialize", {"protocolVersion": "2025-03-26", "capabilities": {}})["result"]
    assert init["protocolVersion"] == "2025-03-26"
    assert init["serverInfo"]["name"] == "mara-bridge"
    assert server.handle({"jsonrpc": "2.0", "method": "notifications/initialized"}) is None
    names = {t["name"] for t in _rpc(server, "tools/list")["result"]["tools"]}
    assert {"listen", "set_parameters", "record_and_compare", "analyze_audio", "restore"} <= names
    assert _rpc(server, "nope")["error"]["code"] == -32601
    data, error = _call(server, "nope")
    assert error and "Unknown tool" in data["error"]


def test_listen_change_and_restore(tmp_path):
    bridge, song = _bridge(tmp_path)
    server = br.Server(bridge)
    heard, error = _call(server, "listen", seconds=4)
    assert not error, heard
    assert heard["main"]["loudness_lufs"] < 0
    names = [t["name"] for t in heard["tracks"]]
    assert "Kick" in names and "REF Song" not in names
    assert any(m["by"] == "Bass" or m["masked"] == "Bass" for m in heard["masking"])

    kick = next(t for t in heard["tracks"] if t["name"] == "Kick")
    done, error = _call(server, "set_parameters", changes=[{"track": "Kick", "param": "Volume", "target": -6}])
    assert not error, done
    assert done["applied"][0]["now"].startswith("-6")
    louder, _ = _call(server, "listen", seconds=4)
    assert next(t for t in louder["tracks"] if t["name"] == "Kick")["loudness_lufs"] < kick["loudness_lufs"] - 3

    restored, error = _call(server, "restore")
    assert not error and restored["parameters_changed"] == 1
    assert song.tracks[0].mixer_device.volume.value == pytest.approx(0.85)


def test_tap_is_protected_and_errors_are_tool_errors(tmp_path):
    bridge, _ = _bridge(tmp_path)
    server = br.Server(bridge)
    state = bridge.live.get_set()
    kick = next(t for t in state["tracks"] if t["name"] == "Kick")
    index = next(d["index"] for d in kick["devices"] if d["name"] == br.TAP)
    data, error = _call(server, "set_parameters",
                        changes=[{"track": "Kick", "device": index, "param": "Device On", "value": 0}])
    assert error and "measurement device" in data["error"]
    data, error = _call(server, "set_parameters", changes=[{"track": "Nope", "param": "Volume", "target": 0}])
    assert error and "No track" in data["error"]


def test_kick_character_from_a_full_mix(tmp_path):
    a = character.describe(_song(tmp_path / "a.wav", f0=55, decay=0.15))
    b = character.describe(_song(tmp_path / "b.wav", f0=48, decay=0.35, click=1.2, gain=0.95))
    assert 21 <= a["kick"]["hits"] <= 24            # 23 hits in 12 s at 120 BPM
    assert a["kick"]["fundamental_hz"] == pytest.approx(55, abs=3)
    assert b["kick"]["fundamental_hz"] == pytest.approx(48, abs=3)
    diff = character.difference(a, b)
    assert diff["kick"]["decay_ms_to_-10db"] < -100          # the reference kick rings longer
    assert diff["kick"]["spectrum_vs_low_db"]["click_2k_6k"] > 0 or diff["kick"]["over_bass_db"] != 0
    assert diff["loudness"]["integrated_lufs"] < -1          # the reference is louder
    assert set(character.only(a, ["kick"])) == {"seconds", "kick"}


def test_record_and_compare_against_a_reference(tmp_path, monkeypatch):
    from mixmatch import autoref

    bridge, _ = _bridge(tmp_path)
    mix = _song(tmp_path / "mix.wav", gain=0.4)
    ref = _song(tmp_path / "ref.wav", f0=48, gain=0.95, seed=1)
    calls = []
    monkeypatch.setattr(autoref, "capture_begin", lambda live, beats, source, start=None: calls.append(
        (beats, source, start)) or 7)
    monkeypatch.setattr(autoref, "capture_end", lambda live, track, timeout=20: mix)
    out = bridge.record_and_compare(start_beat=32, bars=4, reference=ref, aspects=["loudness", "kick"])
    assert calls == [(16.0, None, 32.0)]
    result = out["references"]["ref.wav"]["mix_minus_reference"]
    assert result["loudness"]["integrated_lufs"] < -3
    assert "tonal" not in result


def test_serve_keeps_stdout_for_the_protocol(tmp_path, capsys):
    bridge, _ = _bridge(tmp_path)
    lines = [json.dumps({"jsonrpc": "2.0", "id": 1, "method": "initialize", "params": {}}),
             json.dumps({"jsonrpc": "2.0", "method": "notifications/initialized"}),
             "not json",
             json.dumps({"jsonrpc": "2.0", "id": 2, "method": "tools/call",
                         "params": {"name": "add_taps", "arguments": {}}})]
    out = io.StringIO()
    br.serve(io.StringIO("\n".join(lines) + "\n"), out, bridge)
    replies = [json.loads(line) for line in out.getvalue().splitlines()]
    assert [r.get("id") for r in replies] == [1, None, 2]
    assert replies[1]["error"]["code"] == -32700


def test_compare_elements_groups_channels_by_role(tmp_path):
    bridge, _ = _bridge(tmp_path)
    ref = _song(tmp_path / "ref.wav", f0=48, gain=0.95)
    out = bridge.compare_elements(reference=ref, seconds=4)
    el = out["elements"]
    assert [c["track"] for c in el["kick"]["your_channels"]] == ["Kick"]
    assert [c["track"] for c in el["vocals"]["your_channels"]] == ["Lead Vox"]
    assert el["kick"]["yours"]["level_vs_mix_db"] < 0
    assert el["kick"]["references"]["ref.wav"]["sound"]["fundamental_hz"] == pytest.approx(48, abs=3)
    assert el["vocals"]["references"]["ref.wav"]["sound"] is None      # not separable in a stereo master
    out = bridge.compare_elements(reference=ref, seconds=4, assign={"Pad": "none", "Lead Vox": "music"})
    assert out["elements"]["vocals"]["your_channels"] is None
    assert {c["track"] for c in out["elements"]["music"]["your_channels"]} == {"Lead Vox"}


def test_record_a_solo_track_against_the_reference(tmp_path, monkeypatch):
    from mixmatch import autoref

    bridge, _ = _bridge(tmp_path)
    mine = _song(tmp_path / "mykick.wav", f0=60, decay=0.08, bass=0.0)
    monkeypatch.setattr(autoref, "capture_begin", lambda live, beats, source, start=None: 7)
    monkeypatch.setattr(autoref, "capture_end", lambda live, track, timeout=20: mine)
    out = bridge.record_and_compare(start_beat=0, bars=4, reference=_song(tmp_path / "ref.wav", f0=48, bass=0.0),
                                    track="Kick")
    assert out["element"] == "kick"
    diff = out["references"]["ref.wav"]["difference"]
    assert diff["fundamental_hz"] > 5            # my kick sits higher than the reference's
    assert diff["decay_ms_to_-10db"] < 0          # and is shorter


def test_listen_reports_rms_and_crest(tmp_path):
    bridge, _ = _bridge(tmp_path)
    heard = bridge.listen(seconds=3)
    assert heard["main"]["rms_dbfs"] < 0 and heard["main"]["crest_db"] > 0
    kick = next(t for t in heard["tracks"] if t["name"] == "Kick")
    assert {"rms_dbfs", "crest_db", "plr_db"} <= set(kick)


def test_spectrum_and_spectrogram_with_pictures(tmp_path):
    bridge, _ = _bridge(tmp_path)
    server = br.Server(bridge)
    out = _rpc(server, "tools/call", {"name": "spectrum", "arguments": {"tracks": ["Main", "Bass"], "seconds": 3}})
    content = out["result"]["content"]
    data = json.loads(content[0]["text"])
    assert set(data["spectra"]) == {"Main", "Bass"}
    assert data["spectra"]["Bass"]["columns"] == ["hz", "avg_db", "max_db"]
    assert content[1]["type"] == "image" and content[1]["mimeType"] == "image/png"
    import base64
    assert base64.b64decode(content[1]["data"])[:8] == b"\x89PNG\r\n\x1a\n"

    ref = _song(tmp_path / "ref.wav")
    fine = bridge.spectrum(path=ref, resolution=24, compare_to=ref, image=False)
    diff = fine["spectra"]["difference"]["points"]
    assert all(abs(d) < 0.01 for _, d in diff)                     # a file against itself
    assert len(fine["spectra"]["ref.wav"]["points"]) > 200          # 1/24 octave over ten octaves

    gram = bridge.spectrogram(track="Kick", seconds=3)
    assert gram["table"]["rows"] and gram["_images"][0][:4] == b"\x89PNG"


def test_spectrum_finds_a_resonance(tmp_path):
    from mara_bridge import spectrum as sp

    sr, n = 44100, 44100 * 4
    rng = np.random.default_rng(0)
    x = rng.standard_normal(n) * 0.05 + 0.3 * np.sin(2 * np.pi * 440 * np.arange(n) / sr)
    path = tmp_path / "tone.wav"
    sf.write(path, np.stack([x, x], 1).astype(np.float32), sr)
    peaks = sp.from_audio(str(path), 12)["peaks"]
    assert peaks and abs(peaks[0]["hz"] - 440) < 30


def test_mixer_eq_and_compressor(tmp_path):
    bridge, song = _bridge(tmp_path)
    bridge.mixer("Bass", volume_db=-4, pan=-10)
    assert song.tracks[1].mixer_device.volume.str_for_value(song.tracks[1].mixer_device.volume.value).startswith("-4")
    out = bridge.eq("Bass", [{"freq_hz": 250, "gain_db": -3, "q": 1.2, "type": "bell"},
                             {"freq_hz": 30, "type": "low_cut"}])
    assert not out["errors"]
    eq = next(d for d in song.tracks[1].devices if d.class_name == "Eq8")
    bands = {n: eq.p(f"{n} Filter Type A").str_for_value(eq.p(f"{n} Filter Type A").value) for n in (1, 2)}
    assert bands == {1: "Bell", 2: "Low Cut 48"}
    assert eq.p("1 Gain A").value == pytest.approx(-3, abs=0.1)
    # The same band is adjusted, not a new one used.
    bridge.eq("Bass", [{"freq_hz": 260, "gain_db": -5, "type": "bell"}])
    assert eq.p("1 Gain A").value == pytest.approx(-5, abs=0.1) and eq.p("3 Filter On A").value == 0

    bridge.compressor("Bass", threshold_db=-24, ratio=4, attack_ms=10, release_ms=120, model="rms")
    comp = next(d for d in song.tracks[1].devices if d.class_name == "Compressor2")
    assert comp.p("Ratio").value == pytest.approx(4, abs=0.1)
    assert comp.p("Model").value == 1
    assert "Threshold" in bridge.compressor("Bass")["settings"]
