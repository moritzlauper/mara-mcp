"""The elements of a song: kick, snare, hats, drums (the whole kit), bass, vocals, music (synths, keys, pads …).

In the user's set every element is its own channel, so it is measured solo:

* **balance** from the channel taps: how loud the element sits against the mix (after faders and groups), its
  tonal balance, width, mono low end and dynamics.
* **sound** from a solo recording of the channel: for drums the hits (count, decay, fundamental, attack
  spectrum: sub, knock, click, crack, snap, brightness), for bass sub and fundamental range, for vocals presence,
  sibilance, air, mud and how even the level is.

A reference is a finished stereo master, no stems. What can be read from it by timing is measured there too:
kick, snare and hats by their hits (each a sharp rise in its own band that falls again), the bass as what stays
in 35–150 Hz between the kicks. Vocals and the rest of the music cannot be told apart in a stereo master; for
them the comparison is the full mix (tonal balance, loudness, stereo) via record_and_compare on Main.
"""
from __future__ import annotations

import math

import numpy as np

from .character import HOP_S, _band_power, _db, _envelope, _onsets, _r

ELEMENTS = ("kick", "snare", "hats", "drums", "bass", "vocals", "music")
FROM_MIX = {"kick", "snare", "hats", "bass"}   # measurable in a stereo master
HITS = {   # element: (band, minimum gap between hits in s, hops within which a hit falls 3 dB)
    "kick": ((35, 150), 0.2, 30),
    "snare": ((150, 5000), 0.15, 25),
    "hats": ((6000, 16000), 0.05, 10),
}


def element_of_role(role: str | None) -> str | None:
    from mixmatch.roles import LOW_END_ROLES, ROLE_GROUPS

    if role in ("kick",):
        return "kick"
    if role in ("snare", "clap"):
        return "snare"
    if role in ("hats", "cymbals", "shaker"):
        return "hats"
    if role in ROLE_GROUPS["drums"]:
        return "drums"
    if role in LOW_END_ROLES:
        return "bass"
    if role in ROLE_GROUPS["vocals"]:
        return "vocals"
    if role in ROLE_GROUPS["synths"] or role in ROLE_GROUPS["instruments"]:
        return "music"
    return None   # fx, busses, other: not compared


# ── balance (channel taps) ─────────────────────────────────────────────────
def balance(st, mix) -> dict | None:
    """One element (Stats of its channels, after faders) against Main over the same window."""
    if st is None or mix is None:
        return None
    s = st.summary()
    return {"level_vs_mix_db": _r(st.loudness - mix.loudness), "peak_dbfs": s["peak_dbfs"], "plr_db": s["plr_db"],
            "width_db": s["width_db"], "low_side_db": s["low_side_db"], "tonal_balance_db": s["tonal_balance_db"]}


def combine(items: list):
    """Several channels as one element: powers add up (uncorrelated sources)."""
    from mixmatch.analysis import Stats

    items = [s for s in items if s is not None]
    if not items:
        return None
    if len(items) == 1:
        return items[0]
    w = np.array([10 ** (s.loudness / 10) for s in items])
    bands = 10 * np.log10(np.sum([10 ** (np.asarray(s.bands_db) / 10) for s in items], axis=0))
    loud = float(10 * np.log10(w.sum()))
    mean_db = lambda attr: float(10 * np.log10(np.sum(w * 10 ** (np.array([getattr(s, attr) for s in items]) / 10))
                                               / w.sum()))
    peak = max(s.peak_db for s in items)
    return Stats(frames=max(s.frames for s in items), loudness=loud, bands_db=bands, shape_db=bands - loud,
                 peak_db=peak, plr=peak - loud, width_db=mean_db("width_db"), low_side_db=mean_db("low_side_db"),
                 active=max(s.active for s in items))


# ── sound (audio) ──────────────────────────────────────────────────────────
def _mono(path: str) -> tuple[np.ndarray, int, np.ndarray]:
    from mixmatch.reference import load

    data, sr = load(path)
    data = data.astype(np.float64)
    return data.mean(axis=1), sr, data


def _hits(mono: np.ndarray, sr: int, element: str, avoid: np.ndarray | None = None) -> dict | None:
    from scipy.signal import butter, sosfilt

    band, gap, fall = HITS[element]
    hi = min(band[1], sr / 2 * 0.95)
    filtered = sosfilt(butter(4, (band[0], hi), btype="bandpass", fs=sr, output="sos"), mono)
    env_db = _db(_envelope(filtered, sr, window=3 if element == "kick" else 1))
    onsets = _onsets(env_db, gap_s=gap, fall_hops=fall)
    if avoid is not None and len(avoid) and len(onsets):
        # On a drum stem the kick's attack also rises in the snare band.
        near = np.min(np.abs(onsets[:, None] - avoid[None, :]), axis=1) <= 4
        onsets = onsets[~near]
    if len(onsets) < 3:
        return None
    hop = int(sr * HOP_S)
    peaks, decays, fundamentals = [], [], []
    added = []
    n_fft = 1 << int(math.ceil(math.log2(sr)))
    for j, o in enumerate(onsets):
        nxt = onsets[j + 1] if j + 1 < len(onsets) else min(len(env_db), o + int(0.6 / HOP_S))
        if nxt - o < 3:
            continue
        p = o + int(np.argmax(env_db[o:o + 5]))
        peaks.append(env_db[p])
        below = np.nonzero(env_db[p:nxt] <= env_db[p] - 10)[0]
        if len(below):
            decays.append(below[0] * HOP_S * 1000)
        start = o * hop
        if element in ("kick", "snare"):
            a, b = (0.04, 0.16) if element == "kick" else (0.01, 0.08)
            body = mono[start + int(a * sr): start + int(b * sr)]
            if len(body) == int(b * sr) - int(a * sr):
                f, s = _band_power(body, sr, n_fft)
                lo, hi_f = (35, 150) if element == "kick" else (150, 350)
                sel = (f >= lo) & (f <= hi_f)
                fundamentals.append(f[sel][np.argmax(s[sel])])
        hit = mono[start: start + int(0.05 * sr)]
        pre = mono[max(0, start - int(0.06 * sr)): start - int(0.01 * sr)]
        if len(hit) == int(0.05 * sr) and len(pre) == int(0.05 * sr):
            f, s_hit = _band_power(hit, sr, 4096)
            _, s_pre = _band_power(pre, sr, 4096)
            added.append(np.maximum(s_hit - s_pre, 0))
    if not peaks:
        return None
    out = {"hits_per_minute": _r(len(onsets) / (len(mono) / sr) * 60, 0),
           "decay_ms_to_-10db": _r(np.median(decays), 0) if len(decays) >= len(peaks) / 2 else None,
           "rings_into_next_hit": len(decays) < len(peaks) / 2}
    if fundamentals:
        out["fundamental_hz"] = _r(np.median(fundamentals), 0)
    if added:
        f = np.fft.rfftfreq(4096, 1 / sr)
        gain = np.median(np.array(added), axis=0)
        region = lambda lo, hi: 10 * np.log10(max(gain[(f >= lo) & (f < hi)].sum(), 1e-18))
        if element == "kick":
            low = region(35, 150)
            out.update(sub_vs_low_db=_r(region(35, 70) - region(70, 150)),
                       knock_150_400_vs_low_db=_r(region(150, 400) - low),
                       click_2k_6k_vs_low_db=_r(region(2000, 6000) - low))
        elif element == "snare":
            body = region(150, 400)
            out.update(crack_2k_6k_vs_body_db=_r(region(2000, 6000) - body),
                       snap_6k_12k_vs_body_db=_r(region(6000, 12000) - body),
                       box_400_1k_vs_body_db=_r(region(400, 1000) - body))
        else:
            sel = (f >= 2000) & (f < min(20000, sr / 2))
            out["brightness_hz"] = _r(float((f[sel] * gain[sel]).sum() / max(gain[sel].sum(), 1e-18)), 0)
    return out


def _ratio(mono: np.ndarray, sr: int, a, b) -> float | None:
    """Power in band a over band b, in dB, over the active 400 ms windows."""
    win = int(0.4 * sr)
    level = _db(_envelope(mono, sr, hop_s=0.4, window=1))
    active = level > max(-70, level.max() - 30)
    segs = [mono[i * win:(i + 1) * win] for i in np.nonzero(active)[0][:150]]
    if not segs:
        return None
    f, s = _band_power(np.concatenate(segs), sr, 1 << int(math.ceil(math.log2(sum(len(x) for x in segs)))))
    pa, pb = s[(f >= a[0]) & (f < a[1])].sum(), s[(f >= b[0]) & (f < b[1])].sum()
    return float(10 * np.log10(max(pa, 1e-18) / max(pb, 1e-18)))


def sound(path: str, element: str, avoid_kick: bool = False) -> dict | None:
    """What one element sounds like on its own, from a stem or a solo recording."""
    mono, sr, data = _mono(path)
    if not mono.size or float(np.max(np.abs(mono))) < 1e-5:
        return None
    if element in HITS:
        avoid = None
        if avoid_kick and element != "kick":
            from scipy.signal import butter, sosfilt
            low = sosfilt(butter(4, (35, 150), btype="bandpass", fs=sr, output="sos"), mono)
            avoid = _onsets(_db(_envelope(low, sr, window=3)))
        return _hits(mono, sr, element, avoid)
    level = _db(_envelope(mono, sr, hop_s=0.4, window=1))
    active = level[level > max(-70, level.max() - 30)]
    out = {"level_spread_db": _r(np.std(active)) if len(active) > 3 else None,   # small = even, compressed
           "active_share": _r(len(active) / max(1, len(level)), 2)}
    if element == "bass":
        win = int(0.2 * sr)
        n_fft = 1 << int(math.ceil(math.log2(sr // 2)))
        notes = []
        for i in range(0, len(mono) - win, win):
            seg = mono[i:i + win]
            if _db(np.sqrt(np.mean(seg ** 2))) > level.max() - 20:
                f, s = _band_power(seg, sr, n_fft)
                sel = (f >= 30) & (f <= 250)
                notes.append(f[sel][np.argmax(s[sel])])
        out.update(sub_35_70_vs_low_70_150_db=_r(_ratio(mono, sr, (35, 70), (70, 150))),
                   growl_150_800_vs_low_db=_r(_ratio(mono, sr, (150, 800), (35, 150))),
                   fundamental_range_hz=[_r(np.percentile(notes, 10), 0), _r(np.percentile(notes, 90), 0)]
                   if notes else None)
    elif element == "vocals":
        out.update(presence_2k_5k_vs_body_200_800_db=_r(_ratio(mono, sr, (2000, 5000), (200, 800))),
                   sibilance_5k_10k_vs_1k_4k_db=_r(_ratio(mono, sr, (5000, 10000), (1000, 4000))),
                   air_10k_16k_vs_1k_4k_db=_r(_ratio(mono, sr, (10000, 16000), (1000, 4000))),
                   mud_200_500_vs_1k_4k_db=_r(_ratio(mono, sr, (200, 500), (1000, 4000))))
    elif element in ("music", "drums"):
        out.update(low_vs_mid_db=_r(_ratio(mono, sr, (40, 250), (250, 4000))),
                   high_vs_mid_db=_r(_ratio(mono, sr, (4000, 16000), (250, 4000))))
    if data.shape[1] == 2:
        m, s = (data[:, 0] + data[:, 1]) / 2, (data[:, 0] - data[:, 1]) / 2
        out["width_db"] = _r(_db(np.sqrt(np.mean(s ** 2))) - _db(np.sqrt(np.mean(m ** 2))))
    return out


def reference_elements(mix_path: str) -> dict:
    """What can be measured of each element in a finished stereo master, cached per file."""
    import json
    import os

    from mixmatch.reference import CACHE, _key

    from .character import describe

    cached = _key(mix_path, "elements-v2")
    if os.path.exists(cached):
        with open(cached) as f:
            return json.load(f)
    out = {e: {"sound": sound(mix_path, e, avoid_kick=e != "kick")} for e in ("kick", "snare", "hats")}
    bass = (describe(mix_path).get("bass") or {})
    out["bass"] = {"sound": {"sub_35_70_vs_low_70_150_db": bass.get("sub_vs_low_db"),
                             "between_kicks_level_vs_mix_db": bass.get("level_vs_mix_db")} if bass else None}
    for e in ("drums", "vocals", "music"):
        out[e] = {"sound": None, "why": "not separable in a stereo master: compare the full mix on Main"}
    os.makedirs(CACHE, exist_ok=True)
    with open(cached, "w") as f:
        json.dump(out, f)
    return out
