"""Frequency response like a spectrum analyser (SPAN-style): average and max hold, mid and side.

Two sources, same output shape:

* **taps**: every channel at once while Live plays, 31 third-octave bands from MARA's measurement taps.
* **audio**: a recording of one channel or Main, or any audio file (a reference). Hann-windowed FFT
  (8192 points, half overlap), smoothed to 1/N octave. Level in dBFS: a full-scale sine reads about 0 dB.

`slope` tilts the display like SPAN's (4.5 dB/octave makes a typical mastered mix look roughly flat; 0 shows
raw levels). Peaks lists narrow bumps that stick out of their neighbourhood: resonances or a strong note.
"""
from __future__ import annotations

import math

import numpy as np

N_FFT = 8192
LOW, HIGH = 20.0, 20000.0


def centres(per_octave: int) -> np.ndarray:
    n = int(math.floor(math.log2(HIGH / LOW) * per_octave)) + 1
    return LOW * 2 ** (np.arange(n) / per_octave)


def _label(f: float) -> str:
    return f"{f / 1000:.2f}k".rstrip("0").rstrip(".") + ("" if f < 1000 else "") if f >= 1000 else f"{f:.0f}"


def _smooth(freqs: np.ndarray, power: np.ndarray, per_octave: int) -> tuple[np.ndarray, np.ndarray]:
    """Mean power per 1/N-octave band; bands narrower than one FFT bin take the interpolated bin value."""
    fc = centres(per_octave)
    half = 2 ** (1 / (2 * per_octave))
    out = np.empty((len(fc),) + power.shape[1:])
    for i, f in enumerate(fc):
        sel = (freqs >= f / half) & (freqs < f * half)
        if sel.sum() >= 1:
            out[i] = power[sel].mean(axis=0)
        else:
            out[i] = np.array([np.interp(f, freqs, power[:, k]) for k in range(power.shape[1])]) \
                if power.ndim == 2 else np.interp(f, freqs, power)
    return fc, out


def _peaks(fc: np.ndarray, db: np.ndarray, per_octave: int, limit: int = 8) -> list[dict]:
    """Bumps at least 3 dB above the median of the surrounding octave."""
    out = []
    w = max(2, per_octave // 2)
    for i in range(w, len(db) - w):
        around = np.concatenate([db[i - w:i], db[i + 1:i + w + 1]])
        prominence = db[i] - float(np.median(around))
        if db[i] == db[i - 1:i + 2].max() and prominence >= 3.0 and 30 <= fc[i] <= 16000:
            out.append({"hz": round(float(fc[i])), "db": round(float(db[i]), 1), "above_surroundings_db":
                        round(prominence, 1)})
    return sorted(out, key=lambda p: -p["above_surroundings_db"])[:limit]


def from_audio(path: str, per_octave: int = 12, slope: float = 0.0, max_points: int | None = None) -> dict:
    from mixmatch.reference import load

    data, sr = load(path)
    data = data.astype(np.float64)
    if not data.size or float(np.max(np.abs(data))) < 1e-6:
        raise ValueError(f"{path} is silent")
    mid, side = (data[:, 0] + data[:, 1]) / 2, (data[:, 0] - data[:, 1]) / 2
    win = np.hanning(N_FFT)
    scale = 2.0 / win.sum()
    hop = N_FFT // 2
    starts = range(0, max(1, len(mid) - N_FFT + 1), hop)
    frames_m, frames_s = [], []
    for i in starts:
        seg_m, seg_s = mid[i:i + N_FFT], side[i:i + N_FFT]
        if len(seg_m) < N_FFT:
            seg_m, seg_s = np.pad(seg_m, (0, N_FFT - len(seg_m))), np.pad(seg_s, (0, N_FFT - len(seg_s)))
        if np.sqrt(np.mean(seg_m ** 2)) < 1e-5:
            continue   # silence would pull the average down
        frames_m.append(np.abs(np.fft.rfft(seg_m * win) * scale) ** 2)
        frames_s.append(np.abs(np.fft.rfft(seg_s * win) * scale) ** 2)
    if not frames_m:
        raise ValueError(f"{path} is silent")
    freqs = np.fft.rfftfreq(N_FFT, 1 / sr)
    pm, ps = np.array(frames_m).T, np.array(frames_s).T            # bins × frames
    fc, sm = _smooth(freqs, pm, per_octave)
    _, ss = _smooth(freqs, ps, per_octave)
    tilt = slope * np.log2(fc / 1000.0)
    db = lambda p: 10 * np.log10(np.maximum(p, 1e-20))
    avg, hold, side_avg = db(sm.mean(axis=1)) + tilt, db(sm.max(axis=1)) + tilt, db(ss.mean(axis=1)) + tilt
    return _result(fc, avg, hold, side_avg, per_octave, slope, f"FFT {N_FFT}, 1/{per_octave} octave",
                   len(frames_m) * hop / sr, max_points)


def from_frames(frames, gain_db: float = 0.0, slope: float = 0.0) -> dict | None:
    """Third-octave spectrum from tap frames (calibrated like MARA's stats, after `gain_db` of faders)."""
    from mixmatch.analysis import masking_frames
    from mixmatch.bands import CENTRES

    mf = masking_frames(frames, gain_db)
    keep = mf.level_db > -90
    if keep.sum() < 3:
        return None
    bands = mf.bands_db[keep]                                       # frames × 31, dB
    fc = np.array(CENTRES)
    tilt = slope * np.log2(fc / 1000.0)
    avg = 10 * np.log10(np.mean(10 ** (bands / 10), axis=0)) + tilt
    hold = bands.max(axis=0) + tilt
    return _result(fc, avg, hold, None, 3, slope, "taps, 1/3 octave", keep.sum() * 0.1, None)


def _result(fc, avg, hold, side, per_octave, slope, how, seconds, max_points) -> dict:
    step = max(1, int(math.ceil(len(fc) / max_points))) if max_points else 1
    idx = range(0, len(fc), step)
    cols = ["hz", "avg_db", "max_db"] + (["side_db"] if side is not None else [])
    rows = [[round(float(fc[i])), round(float(avg[i]), 1), round(float(hold[i]), 1)]
            + ([round(float(side[i]), 1)] if side is not None else []) for i in idx]
    return {"how": how, "slope_db_per_octave": slope, "seconds": round(float(seconds), 1), "columns": cols,
            "points": rows, "peaks": _peaks(np.asarray(fc), np.asarray(avg), per_octave)}


def difference(mine: dict, ref: dict) -> dict:
    """Mine minus reference per frequency, both moved to the same average level in 100 Hz–10 kHz."""
    a = {p[0]: p[1] for p in mine["points"]}
    b = {p[0]: p[1] for p in ref["points"]}
    common = sorted(set(a) & set(b))
    if not common:
        return {}
    mid = [f for f in common if 100 <= f <= 10000] or common
    offset = np.mean([a[f] for f in mid]) - np.mean([b[f] for f in mid])
    return {"level_matched_by_db": round(float(offset), 1), "columns": ["hz", "mine_minus_reference_db"],
            "points": [[f, round(a[f] - b[f] - offset, 1)] for f in common]}


# ── spectrogram ────────────────────────────────────────────────────────────
def spectrogram_from_audio(path: str, per_octave: int = 6, hop_s: float = 0.05) -> tuple[np.ndarray, float, np.ndarray]:
    """(band centres, seconds, dB time × band) of the mid signal, 2048-point FFT."""
    from mixmatch.reference import load

    data, sr = load(path)
    mono = data.astype(np.float64).mean(axis=1)
    n_fft, hop = 2048, max(1, int(sr * hop_s))
    win = np.hanning(n_fft)
    scale = 2.0 / win.sum()
    frames = [np.abs(np.fft.rfft(mono[i:i + n_fft] * win) * scale) ** 2
              for i in range(0, max(1, len(mono) - n_fft), hop)]
    if not frames:
        raise ValueError(f"{path} is too short")
    freqs = np.fft.rfftfreq(n_fft, 1 / sr)
    fc, power = _smooth(freqs, np.array(frames).T, per_octave)
    return fc, len(mono) / sr, 10 * np.log10(np.maximum(power.T, 1e-20))


def spectrogram_from_frames(frames, gain_db: float = 0.0) -> tuple[np.ndarray, float, np.ndarray] | None:
    from mixmatch.analysis import masking_frames
    from mixmatch.bands import CENTRES

    mf = masking_frames(frames, gain_db)
    if len(mf.times) < 3:
        return None
    return np.array(CENTRES), float(mf.times[-1] - mf.times[0] + 0.1), mf.bands_db


def spectrogram_table(fc, seconds: float, db: np.ndarray, step_s: float = 1.0) -> dict:
    """The spectrogram as numbers: third-octave-ish columns every `step_s` seconds, whole dB."""
    n = db.shape[0]
    per = max(1, int(round(n / max(seconds, 1e-9) * step_s)))
    keep = [i for i, f in enumerate(fc) if 25 <= f <= 18000][:: max(1, len(fc) // 31)]
    rows = [[round(float(t * step_s), 1)] + [int(round(float(np.max(db[t * per:(t + 1) * per, i]))))
                                               for i in keep]
            for t in range(int(np.ceil(n / per)))]
    return {"columns": ["s"] + [round(float(fc[i])) for i in keep], "rows": rows,
            "note": f"max level per {step_s:g} s and band, dB"}
