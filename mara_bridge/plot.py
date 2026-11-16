"""Small PNG pictures for the agent: a spectrum (lines) and a spectrogram (heat map). No plotting library:
numpy draws into an RGB array, zlib packs the PNG. Claude and other multimodal clients look at them like at an
analyser window; the numbers come along as text anyway."""
from __future__ import annotations

import struct
import zlib

import numpy as np

W, H = 900, 360
MARGIN_L, MARGIN_B, MARGIN_T, MARGIN_R = 46, 24, 10, 10
BG, GRID, AXIS = (18, 18, 22), (48, 48, 56), (120, 120, 132)
COLOURS = [(250, 196, 70), (90, 180, 250), (230, 90, 120), (120, 220, 150), (200, 140, 250), (240, 140, 60)]
FREQ_TICKS = [20, 50, 100, 200, 500, 1000, 2000, 5000, 10000, 20000]

# 3×5 pixel digits for axis labels (no font needed)
_DIGITS = {
    "0": "111101101101111", "1": "010110010010111", "2": "111001111100111", "3": "111001111001111",
    "4": "101101111001001", "5": "111100111001111", "6": "111100111101111", "7": "111001001001001",
    "8": "111101111101111", "9": "111101111001111", "k": "100101110101101", "-": "000000111000000",
    "s": "011100010001110", ".": "000000000000010",
}


def png(rgb: np.ndarray) -> bytes:
    h, w, _ = rgb.shape
    raw = b"".join(b"\x00" + rgb[y].astype(np.uint8).tobytes() for y in range(h))

    def chunk(kind, data):
        return struct.pack(">I", len(data)) + kind + data + struct.pack(">I", zlib.crc32(kind + data) & 0xFFFFFFFF)

    return (b"\x89PNG\r\n\x1a\n" + chunk(b"IHDR", struct.pack(">IIBBBBB", w, h, 8, 2, 0, 0, 0))
            + chunk(b"IDAT", zlib.compress(raw, 6)) + chunk(b"IEND", b""))


def _text(img, x, y, text, colour=AXIS, scale=2):
    for ch in text:
        bits = _DIGITS.get(ch)
        if bits:
            for i, b in enumerate(bits):
                if b == "1":
                    r, c = divmod(i, 3)
                    img[y + r * scale:y + (r + 1) * scale, x + c * scale:x + (c + 1) * scale] = colour
        x += 4 * scale


def _fx(f, w):
    return MARGIN_L + (np.log10(f) - np.log10(20)) / (np.log10(20000) - np.log10(20)) * (w - MARGIN_L - MARGIN_R)


def _frame(lo_db, hi_db, step=10):
    img = np.zeros((H, W, 3), dtype=np.uint8)
    img[:] = BG
    plot_h = H - MARGIN_T - MARGIN_B
    for f in FREQ_TICKS:
        x = int(_fx(f, W))
        img[MARGIN_T:H - MARGIN_B, x] = GRID
        _text(img, x - 6, H - MARGIN_B + 6, f"{f // 1000}k" if f >= 1000 else str(f))
    for d in range(int(np.ceil(lo_db / step) * step), int(hi_db) + 1, step):
        y = int(MARGIN_T + (hi_db - d) / (hi_db - lo_db) * plot_h)
        img[y, MARGIN_L:W - MARGIN_R] = GRID
        _text(img, 4, y - 5, str(d))
    return img


def _line(img, xs, ys, colour, thick=2):
    for (x0, y0), (x1, y1) in zip(zip(xs[:-1], ys[:-1]), zip(xs[1:], ys[1:])):
        n = int(max(abs(x1 - x0), abs(y1 - y0), 1))
        for t in np.linspace(0, 1, n + 1):
            x, y = int(round(x0 + (x1 - x0) * t)), int(round(y0 + (y1 - y0) * t))
            img[max(0, y - thick // 2):y + thick - thick // 2, max(0, x - thick // 2):x + thick - thick // 2] = colour


def spectrum(curves: list[tuple[list, list]], lo_db: float | None = None, hi_db: float | None = None) -> bytes:
    """`curves`: (frequencies, dB) per line, drawn in COLOURS order (yours first, then the references)."""
    allv = np.concatenate([np.asarray(v, dtype=float) for _, v in curves])
    allv = allv[np.isfinite(allv)]
    hi = hi_db if hi_db is not None else float(np.ceil((allv.max() + 3) / 10) * 10)
    lo = lo_db if lo_db is not None else max(hi - 90, float(np.floor((allv.min() - 3) / 10) * 10))
    img = _frame(lo, hi)
    plot_h = H - MARGIN_T - MARGIN_B
    for i, (f, v) in enumerate(curves):
        f, v = np.asarray(f, dtype=float), np.clip(np.asarray(v, dtype=float), lo, hi)
        _line(img, _fx(f, W), MARGIN_T + (hi - v) / (hi - lo) * plot_h, COLOURS[i % len(COLOURS)])
    return png(img)


def _heat(v):
    """0..1 → dark blue, violet, orange, yellow (like most analysers)."""
    stops = np.array([[10, 10, 30], [60, 20, 110], [200, 50, 90], [250, 150, 40], [255, 245, 170]], dtype=float)
    v = np.clip(v, 0, 1) * (len(stops) - 1)
    i = np.minimum(v.astype(int), len(stops) - 2)
    t = (v - i)[..., None]
    return stops[i] * (1 - t) + stops[i + 1] * t


def spectrogram(freqs, seconds, db: np.ndarray, range_db: float = 70.0) -> bytes:
    """`db`: time × frequency. Time runs left to right, frequency (log) bottom to top."""
    db = np.asarray(db, dtype=float)
    hi = float(np.nanmax(db))
    norm = (db - (hi - range_db)) / range_db
    img = np.zeros((H, W, 3), dtype=np.uint8)
    img[:] = BG
    plot_w, plot_h = W - MARGIN_L - MARGIN_R, H - MARGIN_T - MARGIN_B
    fy = (np.log10(np.asarray(freqs)) - np.log10(20)) / (np.log10(20000) - np.log10(20))
    rows = np.clip(((1 - fy) * plot_h).astype(int), 0, plot_h - 1)
    cols = np.linspace(0, db.shape[0] - 1, plot_w).astype(int)
    canvas = np.zeros((plot_h, plot_w))
    order = np.argsort(rows)
    ys = np.arange(plot_h)
    # every pixel row takes the nearest frequency bin
    nearest = np.array([order[np.argmin(np.abs(rows[order] - y))] for y in ys])
    canvas = norm[cols][:, nearest].T
    img[MARGIN_T:MARGIN_T + plot_h, MARGIN_L:MARGIN_L + plot_w] = _heat(canvas).astype(np.uint8)
    for f in FREQ_TICKS:
        y = int(MARGIN_T + (1 - (np.log10(f) - np.log10(20)) / (np.log10(20000) - np.log10(20))) * plot_h)
        _text(img, 4, min(H - MARGIN_B - 10, max(MARGIN_T, y - 5)), f"{f // 1000}k" if f >= 1000 else str(f))
    for s in range(0, int(seconds) + 1, max(1, int(seconds // 8) or 1)):
        x = int(MARGIN_L + s / max(seconds, 1e-9) * plot_w)
        _text(img, x, H - MARGIN_B + 6, f"{s}s")
    return png(img)
