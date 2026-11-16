"""MARA MCP: an MCP server that lets Claude Code, Codex or any MCP client hear and change an Ableton set.

    mara-mcp              stdio server, started by the MCP client, not by hand (see `mara-mcp --help`)

The client drives the mix itself. The bridge gives it what a chat model cannot get from Live's API alone: the
measured sound of every channel and of Main (loudness, peaks, tonal balance, stereo width, mono low end), which
channels collide in which frequencies, and the distance to reference tracks. Changes go through the same remote
script as MARA's runs; a snapshot of every parameter is taken before the first change, so `restore` can undo them.

References work two ways. `load_references` makes finished songs the target of `listen` (tonal balance, loudness,
width per channel and on Main, measured live). `record_and_compare` records a passage of Main in Live and
analyses it like a reference file, for «mix it like this», «make it loud like this» or «make the kick like this»:
loudness and density, tonal balance, stereo, and the kick and bass as heard in the full mix.

The protocol is MCP over stdio (newline-delimited JSON-RPC 2.0), implemented here without extra dependencies.
Stdout carries the protocol only: everything else goes to stderr.
"""
from __future__ import annotations

import json
import math
import os
import re
import sys
import time
import traceback
from pathlib import Path

import numpy as np

PROTOCOL_VERSIONS = ("2025-06-18", "2025-03-26", "2024-11-05")
from . import SURFACE, TAP
MAX_LISTEN_S = 60.0
MASKING_LIMIT = 8

INSTRUCTIONS = """\
This server connects you to the user's Ableton Live set and lets you hear it through measurement taps.
Loop: `status` (connection, taps, cue points) → `get_set` → measure (`listen`, `spectrum`, `spectrogram`,
`compare_elements`) → change (`mixer`, `eq`, `compressor`, `set_parameters`) → measure the same passage again
(start_beat/bars) and compare. Keep what measurably helps and sounds plausible; `restore` returns to the state
before your first change. Taps measure before the fader; reported channel levels are after faders and groups.
References: the user gives a file path (Claude Code: drag the file into the chat) or puts the song on a track
named «REF …» (tools find those without a path). «Mix it / make it loud like this»: `record_and_compare` on Main.
«Make the kick / snare / bass / vocal like this»: `compare_elements`, then `record_and_compare` with track=….
Spectra against a reference: `spectrum` with compare_to. `load_references` makes songs the target of `listen`.
A reference is a stereo master: its kick, snare, hats and bass are read from their timing, its vocals and music
cannot be separated, say so instead of guessing. Loudness: limiter/clipper on Main, about 5 dB of gain reduction
at most, check true peak and PLR against the reference. Recordings play audibly.
Numbers are decision support, not taste: tell the user what you changed and why, and let them listen."""


def _log(*a):
    print(*a, file=sys.stderr, flush=True)


def _r(x, n=1):
    return None if x is None or not math.isfinite(x) else round(float(x), n)


class BridgeError(Exception):
    """A problem the client can act on: returned as a tool error, not a crash."""


class Bridge:
    def __init__(self, live=None, hub=None, env=None, state_dir: str | Path | None = None):
        from mixmatch.agent import RealEnv

        self._live, self._hub = live, hub
        self.env = env or RealEnv()
        if state_dir is None:
            from mixmatch.platforms import workspace
            state_dir = workspace() / "bridge"
        self.state_dir = Path(state_dir)
        self.target = None
        self.target_info: dict | None = None
        self.started: str | None = None   # snapshot taken before the first change of this session
        self.last: dict | None = None      # the last listen: set state, Main and channel Stats

    # ── connections (lazy: the server starts even while Live is closed) ─────
    @property
    def live(self):
        if self._live is None:
            from mixmatch.live_client import LiveClient
            try:
                client = LiveClient().connect()
                client.request("ping")
            except OSError:
                raise BridgeError("Cannot reach Live on port 9877. Open Live and select «" + SURFACE + "» as a control "
                                  "surface (Settings → Link/Tempo/MIDI). MARA's «Set up the connection» installs it.")
            self._live = client
        return self._live

    @property
    def hub(self):
        if self._hub is None:
            from mixmatch.taps import TapHub
            hub = TapHub()
            try:
                hub.listen()
            except OSError:
                raise BridgeError("Port 9878 is in use: the MARA app is probably running. Close it, the taps can "
                                  "only report to one program.")
            self._hub = hub
        return self._hub

    def _state(self) -> dict:
        from mixmatch.live_client import LiveError
        try:
            return self.live.get_set()
        except (OSError, LiveError) as e:
            self._live = None   # reconnect on the next call
            raise BridgeError(f"Lost the connection to Live: {e}")

    def _track(self, state: dict, key: str) -> dict:
        for t in state["tracks"]:
            if t["key"] == key or t["name"].lower() == str(key).lower():
                return t
        raise BridgeError(f"No track {key!r}. Use the keys from get_set (t0, t1 …, r0 …, m for Main).")

    # ── tools ───────────────────────────────────────────────────────────────
    def status(self) -> dict:
        state = self._state()
        taps = {t["key"]: any(d["name"] == TAP for d in t["devices"]) for t in state["tracks"]}
        from mixmatch.taps import is_reference_name
        without = [t["name"] for t in state["tracks"] if not taps[t["key"]]
                   and not (t["kind"] == "track" and is_reference_name(t["name"]))]
        return {
            "set": state.get("name") or Path(state.get("file_path") or "unsaved").stem,
            "tempo": state.get("tempo"),
            "playing": state.get("is_playing"),
            "song_time_beats": _r(state.get("song_time"), 2),
            "loop": state.get("loop"),
            "cue_points": state.get("cue_points", []),
            "tracks": len(state["tracks"]),
            "tracks_without_tap": without,
            "hint": "Call add_taps to measure these tracks." if without else None,
            "references": self.target_info,
            "snapshot_before_changes": self.started,
        }

    def get_set(self, track: str | None = None, parameters: bool = False) -> dict:
        state = self._state()
        tracks = [self._track(state, track)] if track else state["tracks"]
        detail = parameters or track is not None

        def param(p):
            out = {"i": p["index"], "name": p["name"], "display": p["display"]}
            if detail:
                out.update(value=_r(p["value"], 4), min=p["min"], max=p["max"])
                if p.get("items"):
                    out["items"] = p["items"]
                if p.get("automated"):
                    out["automated"] = True
            return out

        out = []
        for t in tracks:
            devices = []
            for d in t["devices"]:
                if d["name"] == TAP:
                    continue
                on = next((p["value"] >= 0.5 for p in d["params"] if p["name"] == "Device On"), True)
                entry = {"index": d["index"], "name": d["name"], "class": d.get("class_name"), "on": on}
                if detail:
                    entry["params"] = [param(p) for p in d["params"] if p["name"] != "Device On"]
                devices.append(entry)
            out.append({"key": t["key"], "name": t["name"], "kind": t["kind"],
                        **({"group": t["group"]} if t.get("group") else {}),
                        **({"is_group": True} if t.get("is_group") else {}),
                        **({"muted": True} if t.get("muted") else {}),
                        "mixer": {p["name"]: p["display"] for p in t["mixer"]}, "devices": devices})
        note = None if detail else "Parameters left out: get_set(track=…) or parameters=true lists them."
        return {"tempo": state.get("tempo"), "tracks": out, **({"note": note} if note else {})}

    def set_parameters(self, changes: list[dict]) -> dict:
        if not changes:
            raise BridgeError("No changes given.")
        state = self._state()
        batch = []
        for c in changes:
            t = self._track(state, c.get("track"))
            device = c.get("device", "mixer")
            if isinstance(device, str) and device != "mixer":
                device = self._device_index(t, device)
            if device != "mixer" and any(d["index"] == device and d["name"] == TAP for d in t["devices"]):
                raise BridgeError(f"The {TAP} is the measurement device; leave it as it is.")
            if "param" not in c or ("value" not in c and "target" not in c):
                raise BridgeError(f"Each change needs param and target (display units) or value (raw): {c}")
            item = {"track": t["key"], "device": device, "param": c["param"]}
            item["target" if "target" in c else "value"] = float(c["target"] if "target" in c else c["value"])
            batch.append(item)
        self._snapshot_once(state)
        result = self.live.batch(batch)
        applied = [{"track": b["track"], "device": b["device"], "param": b["param"], "now": r["display"]}
                   for b, r in zip(batch, result["results"]) if r]
        return {"applied": applied, "errors": result.get("errors", []),
                "undo": f"restore(name={self.started!r}) returns to the state before your first change."}

    def _device_index(self, track: dict, name: str) -> int:
        hits = [d["index"] for d in track["devices"] if d["name"].lower() == name.lower()]
        if len(hits) != 1:
            raise BridgeError(f"Device {name!r} on {track['name']}: {'not found' if not hits else 'ambiguous'}; "
                              "use its index from get_set.")
        return hits[0]

    def add_device(self, track: str, name: str, source: str = "audio_effects") -> dict:
        state = self._state()
        t = self._track(state, track)
        self._snapshot_once(state)
        from mixmatch.live_client import LiveError
        try:
            index = self.live.load_device(t["key"], name, source=source)
        except LiveError as e:
            raise BridgeError(str(e))
        after = self._track(self._state(), t["key"])
        d = next(d for d in after["devices"] if d["index"] == index)
        return {"track": t["key"], "device": {"index": index, "name": d["name"],
                                               "params": [{"i": p["index"], "name": p["name"], "display": p["display"]}
                                                          for p in d["params"]]},
                "note": "Placed before the tap, so listen hears it."}

    def remove_device(self, track: str, device: int) -> dict:
        state = self._state()
        t = self._track(state, track)
        d = next((d for d in t["devices"] if d["index"] == device), None)
        if d is None:
            raise BridgeError(f"No device {device} on {t['name']}.")
        if d["name"] == TAP:
            raise BridgeError(f"The {TAP} is the measurement device; leave it in place.")
        self._snapshot_once(state)
        self.live.delete_device(t["key"], device)
        return {"removed": d["name"], "track": t["name"],
                "note": "restore only resets parameters; a removed device is not loaded back."}

    def transport(self, play: bool | None = None, time_beats: float | None = None, loop_start: float | None = None,
                  loop_length: float | None = None, loop: bool | None = None) -> dict:
        kw = {k: v for k, v in (("play", play), ("time", time_beats), ("loop_start", loop_start),
                                ("loop_length", loop_length), ("loop", loop)) if v is not None}
        return self.live.transport(**kw)

    def listen(self, seconds: float | None = None, start_beat: float | None = None, bars: float | None = None,
               tracks_detail: bool = True) -> dict:
        from mixmatch.agent import fader_gains, peak_percentile, post_fader
        from mixmatch.analysis import compare, masking, masking_frames, stats
        from mixmatch.taps import is_reference_name

        state = self._state()
        hub = self.hub
        tempo = state.get("tempo") or 120.0
        if bars is not None and seconds is None:
            seconds = bars * 4 * 60.0 / tempo
        seconds = min(MAX_LISTEN_S, max(2.0, float(seconds or 8.0)))
        woke = self._wake_taps(state)
        started_here = not state.get("is_playing")
        if start_beat is not None:
            length = bars * 4 if bars else seconds * tempo / 60.0
            self.live.transport(loop_start=float(start_beat), loop_length=length, loop=True,
                                time=float(start_beat), play=True)
        elif started_here:
            self.live.transport(play=True)
        self.env.wait(1.0 if woke or started_here else 0.2)   # taps and audio need a moment to settle
        start = self.env.now()
        self.env.wait(seconds)
        end = self.env.now()
        position = self.live.transport()
        if started_here:
            self.live.transport(play=False)

        master_tap = hub.latest("master")
        master = stats(master_tap.window(start, end)) if master_tap else None
        if master is None:
            raise BridgeError("Main stayed silent: is Live playing, is the audio engine on, and is there a tap on "
                              "Main (add_taps)? Muted output is fine, the taps measure before the fader.")
        in_tap = hub.latest("master_in")
        gains = fader_gains(state["tracks"])
        by_name = hub.by_name()
        measured, windows, silent, no_tap, measured_stats = [], {}, [], [], {}
        for t in state["tracks"]:
            if t["kind"] == "master" or (t["kind"] == "track" and is_reference_name(t["name"])):
                continue
            tap = by_name.get((t["kind"], t["name"]))
            if tap is None:
                no_tap.append(t["name"])
                continue
            frames = tap.window(start, end)
            st = post_fader(stats(frames), gains.get(t["key"], 0.0))
            if st is None:
                silent.append(t["name"])
                continue
            windows[t["key"]] = (t, frames)
            measured_stats[t["key"]] = (t, st)
            entry = {"key": t["key"], "name": t["name"], "loudness_lufs": _r(st.loudness), "peak_dbfs": _r(st.peak_db),
                     **_rms_crest(frames, gains.get(t["key"], 0.0)), "plr_db": _r(st.plr),
                     "active": _r(st.active, 2), "width_db": _r(st.width_db), "low_side_db": _r(st.low_side_db)}
            if t.get("is_group"):
                entry["group_bus"] = True
            if tracks_detail:
                entry["tonal_balance_db"] = st.summary()["tonal_balance_db"]
            measured.append((st.loudness, entry))

        main_frames = master_tap.window(start, end)
        self.last = {"state": state, "main": master, "tracks": measured_stats, "gains": gains,
                     "frames": {k: f for k, (_, f) in windows.items()}, "main_frames": main_frames}
        out = {
            "window": {"seconds": round(end - start, 1), "until_beat": _r(position.get("song_time"), 2)},
            "main": {**master.summary(), **_rms_crest(master_tap.window(start, end), 0.0),
                     "peak99_before_limiter_dbfs": _r(peak_percentile(in_tap.window(start, end)))
                     if in_tap else None},
            "tracks": [e for _, e in sorted(measured, key=lambda x: -x[0])],
            "masking": self._masking(windows, gains, masking, masking_frames),
        }
        if silent:
            out["silent_tracks"] = silent
        if no_tap:
            out["tracks_without_tap"] = no_tap
        if self.target is not None:
            t = self.target
            cmp = compare(master, t.stats, t.loudness, t.tolerance, t.one_sided, offset=t.offset, anchor=t.anchor,
                          lump=t.lump_low)
            out["vs_references"] = {**cmp.summary(),
                                    "meaning": "positive = the mix has more than the references (dB, at equal "
                                               "loudness for tonal bands); largest_* is beyond their spread"}
        out["legend"] = ("loudness ≈ short-term LUFS; tonal_balance_db = band level relative to the track's "
                         "loudness; width_db = side vs mid; low_side_db = side below 150 Hz vs mid (≈ −20 or "
                         "lower is a mono low end); plr = peak to loudness; rms_dbfs = stereo RMS; crest_db = "
                         "typical peak over RMS per 100 ms (high = transient, punchy; low = dense, compressed); "
                         "masking severity 0–1. Exact curves: spectrum, over time: spectrogram")
        return out

    def _masking(self, windows: dict, gains: dict, masking, masking_frames) -> list[dict]:
        keys = [k for k, (t, _) in windows.items() if not t.get("is_group") and t["kind"] == "track"
                and math.isfinite(gains.get(k, 0.0))]
        frames = {k: masking_frames(windows[k][1], gains.get(k, 0.0)) for k in keys}
        hits = []
        for i, a in enumerate(keys):
            for b in keys[i + 1:]:
                for fg, bg in ((a, b), (b, a)):
                    hit = masking(frames[fg], frames[bg])
                    if hit and hit["severity"] >= 0.05:
                        hits.append({"masked": windows[fg][0]["name"], "by": windows[bg][0]["name"],
                                     "severity": round(hit["severity"], 2),
                                     "bands_hz": [x["freq_hz"] for x in hit["bands"]],
                                     "overlap": hit["overlap_fraction"]})
        hits.sort(key=lambda h: -h["severity"])
        return hits[:MASKING_LIMIT]

    def _wake_taps(self, state: dict) -> bool:
        from mixmatch.cli import set_taps_on
        changed = set_taps_on(self.live, True)
        if not any(d["name"] == TAP for t in state["tracks"] for d in t["devices"]):
            raise BridgeError(f"No {TAP} in this set: call add_taps first.")
        return changed > 0

    def add_taps(self) -> dict:
        from mixmatch.cli import add_taps
        from mixmatch.live_client import LiveError
        try:
            n = add_taps(self.live, log=_log)
        except LiveError as e:
            raise BridgeError(f"{e}. Install the MARA connection first (MARA app → Set up the connection).")
        return {"added": n, "note": "Every track, group, return and Main has a tap now."}

    def _files(self, paths: list[str] | None) -> list[str]:
        """Audio files from paths and folders; None means the clips on the set's «REF …» tracks."""
        from mixmatch.cli import AUDIO

        if paths is None:
            clips = self.live.request("reference_clips").get("clips", [])
            files = [c["file"] for c in clips if c.get("file") and os.path.exists(c["file"])]
            if not files:
                raise BridgeError("No reference given and no audio clip on a track named «REF …» in the set. Ask the "
                                  "user for the file path (in Claude Code they can drag the file into the chat).")
            return files
        files = []
        for p in paths:
            p = os.path.expanduser(str(p).strip().strip("'\""))
            if os.path.isdir(p):
                files += sorted(os.path.join(p, f) for f in os.listdir(p) if f.lower().endswith(AUDIO))
            elif os.path.exists(p):
                files.append(p)
            else:
                raise BridgeError(f"Not found: {p}")
        return files

    def load_references(self, paths: list[str] | None = None, vocal_room: bool = False) -> dict:
        from mixmatch.reference import ReferenceSet

        files = self._files(paths)
        if not files:
            self.target, self.target_info = None, None
            return {"references": None, "note": "Cleared."}
        refs = ReferenceSet(files, log=_log)
        self.target = refs.target(beat=vocal_room)
        self.target_info = {"files": [os.path.basename(f) for f in files], "paths": files, "vocal_room": vocal_room,
                            "how": self.target.description}
        return {**{k: v for k, v in self.target_info.items() if k != "paths"}, "target": {**self.target.stats.summary(), "loudness_lufs": _r(self.target.loudness)},
                "per_reference_lufs": {os.path.basename(r.path): _r(r.full.loudness) for r in refs.refs}}

    def analyze_audio(self, path: str | None = None, aspects: list[str] | None = None,
                      elements: bool = True) -> dict:
        from . import character

        files = self._files([path] if path else None)
        out = {}
        for f in files:
            try:
                out[os.path.basename(f)] = character.only(character.describe(f), aspects)
            except ValueError as e:
                raise BridgeError(str(e))
        result = {"analysis": out, "legend": CHARACTER_LEGEND}
        if elements:
            for name, el in self._reference_elements(files).items():
                out[name]["elements"] = el
            result["elements_legend"] = ELEMENTS_LEGEND
        return result

    def record_and_compare(self, start_beat: float, bars: float = 8, reference: str | None = None,
                           aspects: list[str] | None = None, track: str | None = None,
                           element: str | None = None) -> dict:
        """Records the passage from Main (or one track after its devices) and compares it with the reference."""
        from . import character

        if not 1 <= bars <= 64:
            raise BridgeError("bars: 1–64.")
        state = self._state()
        source = self._track(state, track)["key"] if track else None
        if source == "m":
            source = None
        if reference:
            ref_files = self._files([reference])
        elif self.target_info:
            ref_files = self._loaded_reference_files()
        else:
            try:
                ref_files = self._files(None)   # «REF …» tracks in the set
            except BridgeError:
                ref_files = []
        path, _ = self._record(start_beat, bars, source)
        if source is not None:
            return self._compare_solo(path, state, source, element, ref_files)
        try:
            mix = character.describe(path)
        except ValueError as e:
            raise BridgeError(str(e))
        out = {"recorded": path, "source": "Main", "mix": character.only(mix, aspects)}
        for f in ref_files or []:
            ref = character.describe(f)
            out.setdefault("references", {})[os.path.basename(f)] = {
                "reference": character.only(ref, aspects),
                "mix_minus_reference": character.only(character.difference(mix, ref), aspects)}
        if not ref_files:
            out["note"] = (out.get("note", "") + " No reference: give `reference` or load_references.").strip()
        out["legend"] = CHARACTER_LEGEND
        return out

    # ── elements: kick, snare, hats, drums, bass, vocals, music ────────────────
    def _role(self, track: dict, st, assign: dict | None) -> tuple[str | None, str]:
        """Element of a channel: the agent's assignment, else MARA's role guess (name, devices, sound)."""
        from mixmatch import roles

        from .elements import ELEMENTS, element_of_role
        for key in (track["key"], track["name"], track["name"].lower()):
            if assign and key in assign:
                element = assign[key]
                if element not in ELEMENTS and element not in (None, "none"):
                    raise BridgeError(f"Unknown element {element!r}; choose from {list(ELEMENTS)} or none.")
                return (None if element in (None, "none") else element), "assigned"
        guess = roles.guess(track, st)
        return element_of_role(guess.role), f"{guess.role} ({guess.source})"

    def _reference_elements(self, ref_files: list[str]) -> dict:
        """What each reference reveals of its elements (kick, snare, hats and bass by timing)."""
        from .elements import reference_elements

        return {os.path.basename(f): reference_elements(f) for f in ref_files}

    def compare_elements(self, reference: str | None = None, start_beat: float | None = None,
                         bars: float | None = None, seconds: float | None = None,
                         assign: dict | None = None) -> dict:
        """Every element of the set, measured on its channels, with what the reference reveals of it."""
        from .elements import ELEMENTS, balance, combine

        if reference:
            ref_files = self._files([reference])
        else:
            try:
                ref_files = self._loaded_reference_files() or self._files(None)
            except BridgeError:
                ref_files = []
        refs = self._reference_elements(ref_files)
        self.listen(seconds=seconds, start_beat=start_beat, bars=bars, tracks_detail=False)
        last = self.last
        groups: dict[str, list] = {e: [] for e in ELEMENTS}
        channels, ignored = {e: [] for e in ELEMENTS}, []
        for key, (t, st) in last["tracks"].items():
            if t.get("is_group"):
                continue   # a group bus holds its tracks: counting both would double them
            element, why = self._role(t, st, assign)
            if element is None:
                ignored.append({"track": t["name"], "role": why})
                continue
            groups[element].append(st)
            channels[element].append({"track": t["name"], "key": key, "role": why})
            if element in ("kick", "snare", "hats"):
                groups["drums"].append(st)
        mine = {e: combine(v) for e, v in groups.items()}
        out = {"elements": {}, "ignored_channels": ignored}
        for e in ELEMENTS:
            entry = {"your_channels": channels[e] or None, "yours": balance(mine[e], last["main"])}
            if refs:
                entry["references"] = {name: ref.get(e) for name, ref in refs.items()}
            out["elements"][e] = entry
        out["next"] = ("Wrong channel assignment? Pass assign={\"track name\": \"kick\" | … | \"none\"}. How one "
                       "element sounds against the reference (hits, decay, click, sub, sibilance …): "
                       "record_and_compare with track=…. Element levels in the reference: record_and_compare on Main "
                       "(kick level and punch against the bass in both mixes). Exact curves: spectrum.")
        out["legend"] = ELEMENTS_LEGEND
        return out

    def _compare_solo(self, path: str, state: dict, key: str, element: str | None, ref_files: list[str]) -> dict:
        from .elements import ELEMENTS, sound

        t = self._track(state, key)
        if element is None:
            st = self.last["tracks"].get(key, (None, None))[1] if self.last else None
            element, _ = self._role(t, st, None)
        if element not in ELEMENTS:
            raise BridgeError(f"Which element is {t['name']}? Pass element= one of {list(ELEMENTS)}.")
        yours = sound(path, element)
        if yours is None:
            raise BridgeError(f"{t['name']} was silent or had no clear {element} hits in this passage.")
        out = {"recorded": path, "track": t["name"], "element": element, "yours": yours, "references": {}}
        for name, ref in self._reference_elements(ref_files).items():
            r = (ref.get(element) or {}).get("sound")
            out["references"][name] = ({"sound": r, "difference": _diff(yours, r)} if r else
                                       {"sound": None, "why": (ref.get(element) or {}).get("why", "not found")})
        out["legend"] = ELEMENTS_LEGEND
        return out

    # ── analyser: spectrum and spectrogram ──────────────────────────────────
    def _record(self, start_beat: float, bars: float, track: str | None) -> tuple[str, str]:
        """Records Main or one track (after its devices) for `bars` from `start_beat`; returns (path, label)."""
        from mixmatch import autoref

        state = self._state()
        source = self._track(state, track)["key"] if track else None
        if source == "m":
            source = None
        main = next(t for t in state["tracks"] if t["key"] == "m")
        volume = next((p for p in main["mixer"] if p["name"] == "Volume"), None)
        if source is None and volume is not None and volume["value"] <= 1e-6:
            raise BridgeError("Main is at -inf dB: the recording comes from behind the Main fader. Raise it first.")
        tempo, beats = state.get("tempo") or 120.0, float(bars) * 4
        was_playing = state.get("is_playing")
        self.live.transport(loop_start=float(start_beat), loop_length=beats, loop=True)
        capture = autoref.capture_begin(self.live, beats, source, start=float(start_beat))
        path = autoref.capture_end(self.live, capture, timeout=beats * 60 / tempo + 20)
        if not was_playing:
            self.live.transport(play=False)
        if not path or autoref.silent_file(path):
            raise BridgeError("The recording is missing or silent. Is the passage playing something?")
        return path, (self._track(state, track)["name"] if source else "Main")

    def spectrum(self, track: str | None = None, tracks: list[str] | None = None, path: str | None = None,
                 compare_to: str | None = None, source: str = "taps", start_beat: float | None = None,
                 bars: float | None = None, seconds: float | None = None, resolution: int = 12,
                 slope: float = 0.0, image: bool = True) -> dict:
        from . import plot
        from . import spectrum as sp

        if resolution not in (3, 6, 12, 24):
            raise BridgeError("resolution: 3, 6, 12 or 24 bands per octave.")
        curves, out = [], {}
        if path:
            out[os.path.basename(path)] = sp.from_audio(self._files([path])[0], resolution, slope)
        elif source == "recording":
            if start_beat is None:
                raise BridgeError("A recording needs start_beat (and bars).")
            rec, label = self._record(start_beat, bars or 4, track)
            out[label] = {**sp.from_audio(rec, resolution, slope), "recorded": rec}
        else:
            self.listen(seconds=seconds, start_beat=start_beat, bars=bars, tracks_detail=False)
            last = self.last
            names = tracks or ([track] if track else ["m"])
            for name in names:
                t = self._track(last["state"], name)
                frames = last["main_frames"] if t["key"] == "m" else last["frames"].get(t["key"])
                if not frames:
                    out[t["name"]] = None
                    continue
                out[t["name"]] = sp.from_frames(frames, 0.0 if t["key"] == "m" else last["gains"].get(t["key"], 0.0),
                                                slope)
        if compare_to:
            per_octave = 3 if source == "taps" and not path else resolution
            ref_file = self._files([compare_to])[0]
            ref = sp.from_audio(ref_file, per_octave, slope)
            first = next((v for v in out.values() if v), None)
            out["reference " + os.path.basename(ref_file)] = ref
            if first:
                out["difference"] = sp.difference(first, ref)
        result = {"spectra": out, "legend": "points: [hz, avg_db, max_db (max hold), side_db]; dBFS, a full-scale "
                  "sine ≈ 0 dB; taps are after faders. peaks: narrow bumps above their surroundings (resonances, "
                  "strong notes). difference: yours minus reference, level-matched over 100 Hz–10 kHz. Image: "
                  "first curve yellow, then blue, red, green."}
        if image:
            lines = [([p[0] for p in v["points"]], [p[1] for p in v["points"]]) for k, v in out.items()
                     if v and k != "difference" and "points" in v]
            if lines:
                result["_images"] = [plot.spectrum(lines)]
        return result

    def spectrogram(self, track: str | None = None, path: str | None = None, source: str = "taps",
                    start_beat: float | None = None, bars: float | None = None, seconds: float | None = None,
                    step_s: float = 1.0, image: bool = True) -> dict:
        from . import plot
        from . import spectrum as sp

        if path:
            label, data = os.path.basename(path), sp.spectrogram_from_audio(self._files([path])[0])
        elif source == "recording":
            if start_beat is None:
                raise BridgeError("A recording needs start_beat (and bars).")
            rec, label = self._record(start_beat, bars or 4, track)
            data = sp.spectrogram_from_audio(rec)
        else:
            self.listen(seconds=seconds, start_beat=start_beat, bars=bars, tracks_detail=False)
            last = self.last
            t = self._track(last["state"], track or "m")
            label = t["name"]
            frames = last["main_frames"] if t["key"] == "m" else last["frames"].get(t["key"])
            data = sp.spectrogram_from_frames(frames or [], last["gains"].get(t["key"], 0.0))
            if data is None:
                raise BridgeError(f"{label} was silent in this window.")
        fc, secs, db = data
        result = {"source": label, "seconds": round(secs, 1), "table": sp.spectrogram_table(fc, secs, db, step_s),
                  "legend": "Image: time left to right, frequency bottom (20 Hz) to top (20 kHz), colour = level "
                            "(dark = quiet, yellow = loud, 70 dB range)."}
        if image:
            result["_images"] = [plot.spectrogram(fc, secs, db)]
        return result

    # ── direct controls: mixer, EQ Eight, Compressor ─────────────────────────
    def mixer(self, track: str, volume_db: float | None = None, pan: float | None = None,
              sends: dict | None = None) -> dict:
        changes = []
        if volume_db is not None:
            changes.append({"track": track, "param": "Volume", "target": float(volume_db)})
        if pan is not None:
            if not -50 <= pan <= 50:
                raise BridgeError("pan: -50 (left) … 0 (centre) … 50 (right).")
            changes.append({"track": track, "param": "Pan", "target": float(pan)})
        for send, db in (sends or {}).items():
            name = send if str(send).lower().startswith("send") else f"Send {str(send).upper()}"
            changes.append({"track": track, "param": name, "target": float(db)})
        return self.set_parameters(changes)

    def _device(self, track: str, class_name: str, browser_name: str, index: int | None) -> tuple[dict, dict]:
        """The track and the chosen (or last) device of this class; loads one when there is none."""
        state = self._state()
        t = self._track(state, track)
        found = [d for d in t["devices"] if d.get("class_name") == class_name]
        if index is not None:
            found = [d for d in t["devices"] if d["index"] == index]
            if not found or found[0].get("class_name") != class_name:
                raise BridgeError(f"Device {index} on {t['name']} is not {browser_name}.")
        if not found:
            self.add_device(t["key"], browser_name)
            t = self._track(self._state(), t["key"])
            found = [d for d in t["devices"] if d.get("class_name") == class_name]
            if not found:
                raise BridgeError(f"Could not load {browser_name} on {t['name']}.")
        return t, found[-1]

    def eq(self, track: str, bands: list[dict], device: int | None = None, reset: bool = False) -> dict:
        """EQ Eight bands by frequency, gain, Q and type; finds a free or matching band unless `band` is given."""
        t, d = self._device(track, "Eq8", "EQ Eight", device)
        params = {p["name"]: p for p in d["params"]}
        q_name = "Resonance A" if "1 Resonance A" in params else "Q A"
        changes = []
        state = {}
        for n in range(1, 9):
            on = params.get(f"{n} Filter On A")
            gain = params.get(f"{n} Gain A")
            state[n] = {"on": bool(on and on["value"] >= 0.5) and not reset,
                        "type": (params.get(f"{n} Filter Type A") or {}).get("display", ""),
                        "freq": _num((params.get(f"{n} Frequency A") or {}).get("display")),
                        "gain": _num(gain["display"]) if gain else 0.0}
            if reset and on and on["value"] >= 0.5:
                changes.append({"track": t["key"], "device": d["index"], "param": f"{n} Filter On A", "value": 0})
        used = set()
        for b in bands:
            kind = str(b.get("type", "bell")).lower().replace(" ", "_")
            item = _eq_item(params.get("1 Filter Type A", {}).get("items", []), kind)
            n = b.get("band")
            if n is None:
                f = float(b["freq_hz"])
                same = [k for k, v in state.items() if v["on"] and k not in used and v["freq"]
                        and abs(math.log2(v["freq"] / f)) <= 1 / 3 and _eq_item([v["type"]], kind) is not None]
                free = [k for k, v in state.items() if k not in used and (not v["on"] or (
                    abs(v["gain"] or 0) < 0.05 and v["type"].lower() in ("bell", "low shelf", "high shelf")))]
                if not (same or free):
                    raise BridgeError(f"All eight bands of EQ Eight on {t['name']} are in use; pass band=1–8.")
                n = (same or free)[0]
            used.add(n)
            if item is None:
                raise BridgeError(f"Unknown EQ type {b.get('type')!r}: bell, low_shelf, high_shelf, low_cut, "
                                  "low_cut_12, high_cut, high_cut_12, notch.")
            base = {"track": t["key"], "device": d["index"]}
            changes.append({**base, "param": f"{n} Filter Type A", "value": float(item)})
            if "freq_hz" in b:
                changes.append({**base, "param": f"{n} Frequency A", "target": float(b["freq_hz"])})
            if "gain_db" in b:
                changes.append({**base, "param": f"{n} Gain A", "target": float(b["gain_db"])})
            if "q" in b:
                changes.append({**base, "param": f"{n} {q_name}", "target": float(b["q"])})
            changes.append({**base, "param": f"{n} Filter On A", "value": 0.0 if b.get("on") is False else 1.0})
        result = self.set_parameters(changes)
        result["device"] = {"track": t["name"], "index": d["index"], "name": d["name"]}
        return result

    def compressor(self, track: str, threshold_db: float | None = None, ratio: float | None = None,
                   attack_ms: float | None = None, release_ms: float | None = None,
                   makeup_db: float | None = None, knee_db: float | None = None, dry_wet: float | None = None,
                   model: str | None = None, on: bool | None = None, device: int | None = None) -> dict:
        t, d = self._device(track, "Compressor2", "Compressor", device)
        params = {p["name"]: p for p in d["params"]}
        base = {"track": t["key"], "device": d["index"]}
        changes = []
        for name, value in (("Threshold", threshold_db), ("Ratio", ratio), ("Attack", attack_ms),
                            ("Release", release_ms), ("Output", makeup_db), ("Knee", knee_db), ("Dry/Wet", dry_wet)):
            if value is not None:
                if name not in params:
                    raise BridgeError(f"This Compressor has no {name}.")
                changes.append({**base, "param": name, "target": float(value)})
        if model is not None:
            items = [i.lower() for i in params.get("Model", {}).get("items", [])]
            if model.lower() not in items:
                raise BridgeError(f"model: one of {items}.")
            changes.append({**base, "param": "Model", "value": float(items.index(model.lower()))})
        if on is not None:
            changes.append({**base, "param": "Device On", "value": 1.0 if on else 0.0})
        if not changes:
            return {"device": {"track": t["name"], "index": d["index"]},
                    "settings": {p["name"]: p["display"] for p in d["params"]}}
        result = self.set_parameters(changes)
        result["device"] = {"track": t["name"], "index": d["index"], "name": d["name"]}
        return result

    def _loaded_reference_files(self) -> list[str]:
        return list(self.target_info.get("paths", [])) if self.target_info else []

    # ── snapshots ───────────────────────────────────────────────────────────
    def _dir(self, state: dict) -> Path:
        name = state.get("name") or Path(state.get("file_path") or "unsaved").stem or "unsaved"
        return self.state_dir / re.sub(r"[^\w .-]", "_", name)

    def _snapshot_once(self, state: dict) -> None:
        if self.started is None:
            self.started = self.snapshot(None, state)["name"]

    def snapshot(self, name: str | None = None, state: dict | None = None) -> dict:
        from mixmatch import ab
        state = state or self._state()
        name = re.sub(r"[^\w.-]", "_", name or time.strftime("%Y%m%d-%H%M%S"))
        folder = self._dir(state)
        folder.mkdir(parents=True, exist_ok=True)
        (folder / f"{name}.json").write_text(json.dumps(ab.snapshot(state)))
        return {"name": name}

    def restore(self, name: str | None = None) -> dict:
        from mixmatch import ab
        state = self._state()
        name = name or self.started
        if not name:
            raise BridgeError("Nothing to restore: no snapshot in this session. list_snapshots shows older ones.")
        path = self._dir(state) / f"{name}.json"
        if not path.exists():
            raise BridgeError(f"No snapshot {name!r}.")
        changes = ab.changes_to(json.loads(path.read_text()), state, bypass_new=True)
        result = self.live.batch(changes) if changes else {"errors": []}
        return {"restored": name, "parameters_changed": len(changes), "errors": result.get("errors", []),
                "note": "Devices added since then are switched off, not removed."}

    def list_snapshots(self) -> dict:
        folder = self._dir(self._state())
        names = sorted(p.stem for p in folder.glob("*.json")) if folder.exists() else []
        return {"snapshots": names, "this_session": self.started}

    def sleep_taps(self) -> None:
        """Taps compute 31 filters per channel; switch them off when the client goes away."""
        if self._live is None:
            return
        try:
            from mixmatch.cli import set_taps_on
            set_taps_on(self._live, False)
        except Exception:
            pass


def _rms_crest(frames, gain_db: float) -> dict:
    """Stereo RMS (mean of L² and R² = M² + S²) and the typical crest factor of the 100 ms frames."""
    from mixmatch.analysis import SILENCE
    live = [f for f in frames if f.m_rms > SILENCE]
    if not live:
        return {}
    power = np.array([f.m_rms ** 2 + f.s_rms ** 2 for f in live])
    crest = [20 * math.log10(max(f.peak, 1e-9) / math.sqrt(p)) for f, p in zip(live, power)]
    return {"rms_dbfs": _r(10 * math.log10(power.mean()) + gain_db), "crest_db": _r(float(np.median(crest)))}


def _num(display) -> float | None:
    from mixmatch.display import parse_display
    v = parse_display(display) if display is not None else None
    if v is not None and isinstance(display, str) and display.strip().lower().endswith("khz"):
        v *= 1000
    return v


EQ_TYPES = {"low_cut": ("low cut 48", "low cut"), "low_cut_48": ("low cut 48",), "low_cut_12": ("low cut 12",),
            "high_cut": ("high cut 48", "high cut"), "high_cut_48": ("high cut 48",),
            "high_cut_12": ("high cut 12",), "low_shelf": ("low shelf",), "high_shelf": ("high shelf",),
            "bell": ("bell",), "notch": ("notch",)}


def _eq_item(items: list[str], kind: str) -> int | None:
    """Index of the EQ Eight filter type, by name (Live versions spell «Low Cut 48dB» differently)."""
    names = [i.lower().replace("db", "").strip() for i in items]
    for want in EQ_TYPES.get(kind, (kind.replace("_", " "),)):
        for i, n in enumerate(names):
            if n.startswith(want):
                return i
    return None


def _diff(yours: dict, ref: dict) -> dict:
    from .character import difference
    return difference(yours, ref)


ELEMENTS_LEGEND = (
    "Elements: kick, snare, hats, drums (whole kit), bass, vocals, music (synths, keys, guitars, pads). yours "
    "(channel taps, after faders): level_vs_mix_db = element loudness minus Main loudness, plr, width, mono low "
    "end, tonal balance. sound (audio of a solo recording; in a reference read from the full mix by timing, so "
    "only kick, snare, hats and bass): hits_per_minute, decay_ms_to_-10db, fundamental_hz, kick sub/knock/click "
    "vs its low end, snare crack/snap/box vs its body, hats brightness_hz, bass sub vs upper low end and "
    "fundamental range, vocals presence, sibilance, air, mud, level_spread_db (small = even, compressed), "
    "width_db. difference = yours minus reference. A reference's vocals and music cannot be told apart in a "
    "stereo master: compare them through the full mix and spectrum.")

CHARACTER_LEGEND = (
    "loudness: integrated/loud_part LUFS, true peak dBTP, plr = true peak minus loud part (smaller = denser), "
    "loudness_range LU, short_crest_db = how far 10 ms peaks rise over the 400 ms body (smaller = more limited "
    "or clipped). tonal: band level relative to loudness in the loudest part. stereo: width_db side vs mid, "
    "low_side_db side below 150 Hz (−20 or lower = mono low end). kick (from the full mix by its hits in "
    "35–150 Hz): fundamental_hz, decay_ms_to_-10db, level_vs_mix_db (hit peak vs loudness), over_bass_db (hit "
    "vs what stays between hits: punch against the bass), spectrum_vs_low_db = what a hit adds per region "
    "relative to its own low end (click_2k_6k high = clicky, body_150_400 = knock), sub_vs_low_db. bass: what "
    "sits in 35–150 Hz between the hits, level and sub vs upper low end. mix_minus_reference: positive = the "
    "mix has more. A bass louder than the kick hides hits.")

ASPECTS_SCHEMA = {"type": "array", "items": {"type": "string",
                                             "enum": ["loudness", "tonal", "stereo", "kick", "bass"]},
                  "description": "Only these parts (default all)"}

# ── MCP tool table ─────────────────────────────────────────────────────────
_TRACK = {"type": "string", "description": "Track key from get_set (t0, r0, m for Main) or the exact track name"}
_DEVICE = {"description": "\"mixer\" (Volume, Pan, Send A …) or the device index from get_set, or its unique name",
           "anyOf": [{"type": "integer"}, {"type": "string"}]}

TOOLS = [
    ("status", "Connection, tempo, transport, cue points, which tracks lack a measurement tap, loaded references.",
     {}),
    ("get_set", "Tracks with mixer and devices. Without `track` only device names; with `track` (or "
                "parameters=true) every parameter with its display value, raw value and range.",
     {"track": _TRACK, "parameters": {"type": "boolean", "default": False}}),
    ("listen", "Play and measure. Returns Main (loudness, peaks, dynamics, width, mono low end, tonal balance), "
               "every channel after its fader, the strongest masking between channels, and the distance to the "
               "references if loaded. Give start_beat (and bars) to loop a passage, so before/after compare the same "
               "music; otherwise it measures wherever Live is playing.",
     {"seconds": {"type": "number", "description": "2–60, default 8 (or the length of `bars`)"},
      "start_beat": {"type": "number", "description": "Song position in beats (cue points from status)"},
      "bars": {"type": "number", "description": "Loop length in bars (4 beats each)"},
      "tracks_detail": {"type": "boolean", "default": True, "description": "Tonal balance per channel"}}),
    ("set_parameters", "Change parameters in one batch. Prefer `target` in display units (dB, Hz, %, e.g. "
                       "Volume target -3.5); `value` is the raw value. The first change of a session saves a "
                       "snapshot for restore.",
     {"changes": {"type": "array", "items": {"type": "object", "properties": {
         "track": _TRACK, "device": _DEVICE,
         "param": {"description": "Parameter name or index", "anyOf": [{"type": "string"}, {"type": "integer"}]},
         "target": {"type": "number"}, "value": {"type": "number"}}, "required": ["track", "param"]}}}),
    ("add_device", "Load a device at the end of a track's chain (before the tap). source: audio_effects (EQ Eight, "
                   "Compressor, Utility, Glue Compressor, Saturator …), plugins (installed VST3/AU by name) or "
                   "user_library.",
     {"track": _TRACK, "name": {"type": "string"},
      "source": {"type": "string", "enum": ["audio_effects", "plugins", "user_library"], "default": "audio_effects"}}),
    ("remove_device", "Delete a device from a track (not the tap).",
     {"track": _TRACK, "device": {"type": "integer"}}),
    ("transport", "Play/stop, jump (time_beats) and set the loop.",
     {"play": {"type": "boolean"}, "time_beats": {"type": "number"}, "loop_start": {"type": "number"},
      "loop_length": {"type": "number", "description": "beats"}, "loop": {"type": "boolean"}}),
    ("load_references", "Make finished songs (files or folders: WAV, AIFF, FLAC, MP3) the target of `listen`. "
                        "Without paths: the clips on «REF …» tracks in the set. vocal_room=true for a beat that still "
                        "gets vocals: 200 Hz–5 kHz may sit under the references. An empty list clears them.",
     {"paths": {"type": "array", "items": {"type": "string"}}, "vocal_room": {"type": "boolean", "default": False}}),
    ("analyze_audio", "Describe a finished song from its audio file: loudness and density, tonal balance, stereo, "
                      "and kick, snare, hats and bass as far as a stereo master reveals them (by their timing). "
                      "Without path: the clips on «REF …» tracks.",
     {"path": {"type": "string", "description": "Audio file on this computer"}, "aspects": ASPECTS_SCHEMA,
      "elements": {"type": "boolean", "default": True}}),
    ("record_and_compare", "Record a passage in Live (it plays, audibly) and compare with the reference. Main "
                           "(default; must not be at -inf): loudness, density, tonal balance, stereo, kick and bass in "
                           "the mix, for «make it loud / mix it like this». With track: that channel solo, compared "
                           "with the same element's stem of the reference, for «make the kick / snare / vocal / bass "
                           "like this» (hits, decay, fundamental, click, body, sub, presence, sibilance …).",
     {"start_beat": {"type": "number"}, "bars": {"type": "number", "default": 8},
      "reference": {"type": "string", "description": "Audio file; default: loaded references or «REF …» tracks"},
      "aspects": ASPECTS_SCHEMA, "track": _TRACK,
      "element": {"type": "string", "enum": ["kick", "snare", "hats", "drums", "bass", "vocals", "music"],
                  "description": "With track: what the track is (default: MARA's role guess)"}}),
    ("compare_elements", "Every element of the set measured on its own channels: kick, snare, hats, drums, bass, "
                         "vocals, music (grouped by MARA's role guess; correct it with assign). Level against Main, "
                         "tonal balance, width, mono low end, dynamics; plus what the reference reveals of each "
                         "element (kick, snare, hats, bass).",
     {"reference": {"type": "string", "description": "Audio file; default: loaded references or «REF …» tracks"},
      "start_beat": {"type": "number"}, "bars": {"type": "number"}, "seconds": {"type": "number"},
      "assign": {"type": "object", "additionalProperties": {"type": "string"},
                 "description": "Track name or key → kick, snare, hats, drums, bass, vocals, music or none"}}),
    ("spectrum", "Frequency response like a spectrum analyser (SPAN): average, max hold and side, with a "
                 "picture. source taps (default): any channels or Main while Live plays, 1/3 octave, many at once "
                 "(tracks). source recording: one track or Main recorded, FFT up to 1/24 octave. path: an audio "
                 "file. compare_to: overlay a reference and get the level-matched difference. Lists resonance "
                 "peaks.",
     {"track": _TRACK, "tracks": {"type": "array", "items": {"type": "string"}},
      "path": {"type": "string"}, "compare_to": {"type": "string", "description": "Reference audio file"},
      "source": {"type": "string", "enum": ["taps", "recording"], "default": "taps"},
      "start_beat": {"type": "number"}, "bars": {"type": "number"}, "seconds": {"type": "number"},
      "resolution": {"type": "integer", "enum": [3, 6, 12, 24], "default": 12,
                     "description": "Bands per octave (recordings and files)"},
      "slope": {"type": "number", "default": 0, "description": "dB/octave tilt like SPAN (4.5 is common)"},
      "image": {"type": "boolean", "default": True}}),
    ("spectrogram", "Level over time and frequency of a channel, Main or an audio file, as a picture and a "
                    "compact table: transients, masking moments, sidechain pumping, resonances that ring.",
     {"track": _TRACK, "path": {"type": "string"},
      "source": {"type": "string", "enum": ["taps", "recording"], "default": "taps"},
      "start_beat": {"type": "number"}, "bars": {"type": "number"}, "seconds": {"type": "number"},
      "step_s": {"type": "number", "default": 1.0}, "image": {"type": "boolean", "default": True}}),
    ("mixer", "Volume (dB), pan (-50 left … 50 right) and sends (dB) of a track in one call.",
     {"track": _TRACK, "volume_db": {"type": "number"}, "pan": {"type": "number"},
      "sends": {"type": "object", "additionalProperties": {"type": "number"},
                "description": "e.g. {\"A\": -12} for Send A"}}),
    ("eq", "Set EQ Eight bands on a track (loads EQ Eight if there is none): frequency, gain, Q and type. A band "
           "near the same frequency and type is adjusted, otherwise a free band is used; band=1–8 forces one.",
     {"track": _TRACK, "device": {"type": "integer", "description": "Which EQ Eight (default: the last one)"},
      "reset": {"type": "boolean", "default": False, "description": "Switch all bands off first"},
      "bands": {"type": "array", "items": {"type": "object", "properties": {
          "freq_hz": {"type": "number"}, "gain_db": {"type": "number"}, "q": {"type": "number"},
          "type": {"type": "string", "enum": list(EQ_TYPES)}, "band": {"type": "integer"},
          "on": {"type": "boolean"}}}}}),
    ("compressor", "Set Live's Compressor on a track (loads one if there is none). Without values it returns the "
                   "current settings.",
     {"track": _TRACK, "threshold_db": {"type": "number"}, "ratio": {"type": "number"},
      "attack_ms": {"type": "number"}, "release_ms": {"type": "number"}, "makeup_db": {"type": "number"},
      "knee_db": {"type": "number"}, "dry_wet": {"type": "number", "description": "percent"},
      "model": {"type": "string", "enum": ["peak", "rms", "expand"]}, "on": {"type": "boolean"},
      "device": {"type": "integer"}}),
    ("add_taps", "Put the " + TAP + " measurement device on every track, group, return and Main that lacks one.",
     {}),
    ("snapshot", "Save every parameter of the set under a name.", {"name": {"type": "string"}}),
    ("restore", "Set every parameter back to a snapshot (default: the one before your first change).",
     {"name": {"type": "string"}}),
    ("list_snapshots", "Saved snapshots of this set.", {}),
]

_REQUIRED = {"mixer": ["track"], "eq": ["track", "bands"], "compressor": ["track"], "set_parameters": ["changes"], "add_device": ["track", "name"], "remove_device": ["track", "device"],
             "record_and_compare": ["start_beat"]}


def tool_list() -> list[dict]:
    return [{"name": n, "description": d,
             "inputSchema": {"type": "object", "properties": props, "required": _REQUIRED.get(n, [])}}
            for n, d, props in TOOLS]


class Server:
    def __init__(self, bridge: Bridge):
        self.bridge = bridge
        self.names = {n for n, _, _ in TOOLS}

    def handle(self, msg: dict) -> dict | None:
        method, mid = msg.get("method"), msg.get("id")
        if mid is None:            # a notification: never answered
            return None
        try:
            if method == "initialize":
                asked = (msg.get("params") or {}).get("protocolVersion")
                from . import __version__ as version
                result = {"protocolVersion": asked if asked in PROTOCOL_VERSIONS else PROTOCOL_VERSIONS[0],
                          "capabilities": {"tools": {}},
                          "serverInfo": {"name": "mara-mcp", "title": "MARA MCP", "version": version},
                          "instructions": INSTRUCTIONS}
            elif method == "ping":
                result = {}
            elif method == "tools/list":
                result = {"tools": tool_list()}
            elif method == "tools/call":
                result = self.call((msg.get("params") or {}).get("name"), (msg.get("params") or {}).get("arguments") or {})
            else:
                return {"jsonrpc": "2.0", "id": mid, "error": {"code": -32601, "message": f"Unknown method {method}"}}
        except Exception as e:   # protocol-level failure
            return {"jsonrpc": "2.0", "id": mid, "error": {"code": -32603, "message": str(e)}}
        return {"jsonrpc": "2.0", "id": mid, "result": result}

    def call(self, name: str, args: dict) -> dict:
        if name not in self.names:
            return _text({"error": f"Unknown tool {name}"}, error=True)
        try:
            result = getattr(self.bridge, name)(**args)
            images = result.pop("_images", []) if isinstance(result, dict) else []
            out = _text(result)
            import base64
            out["content"] += [{"type": "image", "data": base64.b64encode(img).decode(), "mimeType": "image/png"}
                               for img in images]
            return out
        except BridgeError as e:
            return _text({"error": str(e)}, error=True)
        except TypeError as e:
            return _text({"error": f"Bad arguments: {e}"}, error=True)
        except Exception as e:
            _log(traceback.format_exc())
            return _text({"error": f"{type(e).__name__}: {e}"}, error=True)


def _text(data, error: bool = False) -> dict:
    return {"content": [{"type": "text", "text": json.dumps(data, ensure_ascii=False)}], "isError": error}


def serve(stdin=None, stdout=None, bridge: Bridge | None = None) -> None:
    stdin, stdout = stdin or sys.stdin, stdout or sys.stdout
    # Stdout belongs to the protocol: a stray print anywhere in MARA would corrupt it, so prints go to stderr.
    real_stdout, sys.stdout = sys.stdout, sys.stderr
    server = Server(bridge or Bridge())
    _log("MARA MCP ready (MCP over stdio)")
    try:
        for line in stdin:
            line = line.strip()
            if not line:
                continue
            try:
                msg = json.loads(line)
            except ValueError:
                reply = {"jsonrpc": "2.0", "id": None, "error": {"code": -32700, "message": "Parse error"}}
            else:
                reply = server.handle(msg) if isinstance(msg, dict) else None
            if reply is not None:
                stdout.write(json.dumps(reply, ensure_ascii=False) + "\n")
                stdout.flush()
    finally:
        server.bridge.sleep_taps()
        sys.stdout = real_stdout


def main(argv=None) -> None:
    """`mara-mcp` serves MCP on stdio; `setup`, `check` and `config` are for people."""
    import argparse

    p = argparse.ArgumentParser(prog="mara-mcp", description="MARA MCP: let Claude Code, Codex or any MCP "
                                "client hear and change your Ableton Live set.")
    sub = p.add_subparsers(dest="cmd")
    s = sub.add_parser("setup", help="install the Live connection (remote script and measurement tap)")
    s.add_argument("--library", help="your Ableton User Library (default: the standard location)")
    sub.add_parser("check", help="test the connection to Live and the taps")
    sub.add_parser("config", help="print the lines that register the bridge in Claude Code, Codex and others")
    sub.add_parser("serve", help="MCP over stdio (the default without a command)")
    args = p.parse_args(argv)
    if args.cmd == "setup":
        from .install import setup
        setup(args.library)
    elif args.cmd == "check":
        from .install import check
        check()
    elif args.cmd == "config":
        from .install import config
        config()
    else:
        from mixmatch.cli import load_env
        load_env()
        serve()
