"""Forecast skies for the observatory: NOAA GFS cloud and wind forecasts as two tiled images and a small JSON.

Source: NOAA GFS 0.25 degree (public domain), read through Unidata's THREDDS server (thredds.ucar.edu), whose NetCDF
Subset Service cuts out variables, levels, times and a coarser grid and returns compressed netCDF-4 (read with h5py),
so no GRIB decoder is needed. NOMADS' own OPeNDAP/GrADS service was retired in 2025 (NWS notice SCN 25-81).

Output, per docs/FORECAST-FORMAT.md (the contract with the world side):
  forecast.png  2048 x 1024, 16 tiles of 512 x 256 (4 across, 4 down), each a full-globe equirectangular map with
                column 0 at 180 W and row 0 at 90 N. Red: cloud opacity; green: cloud-top height / 16 km; blue 128.
  winds.png     the same tiling: the steering wind for clouds, red east and green north, -250..+250 km/h -> 0..1.
  forecast.json the cycle, the frames (hours after it) and the scales.

    python tools/forecast_layers.py --out folder
    python tools/forecast_layers.py --out folder --cycle 2026-11-06T00   # a given cycle (must still be on the server)
"""
import argparse
import concurrent.futures
import datetime
import io
import json
import os
import re
import sys
import time
import urllib.error
import urllib.parse
import urllib.request

import h5py
import numpy as np
from PIL import Image

UA = "iss-clouds-forecast/1.0"
THREDDS = "https://thredds.ucar.edu/thredds"
COLLECTION = "grib/NCEP/GFS/Global_0p25deg"
CATALOG_URL = f"{THREDDS}/catalog/{COLLECTION}/catalog.xml"

FRAMES = [0, 3, 6, 9, 12, 15, 18, 21, 24, 30, 36, 42, 48, 60, 72, 96]   # hours after the cycle, one per tile
GRID = (4, 4)
TILE_W, TILE_H = 512, 256
WIND_KMH = 250.0
TOP_KM = 16.0           # the same height scale as the live images (live_clouds.py)

# Time blocks, each one request per variable group: every 3 h to 24, every 6 h to 48, then 60/72/84/96 (84 unused).
BLOCKS = [(0, 24, 1), (30, 48, 2), (60, 96, 4)]   # first hour, last hour, stride in GFS's 3-hourly steps
CLOUD_VARS = {"low": "Low_cloud_cover_low_cloud", "mid": "Medium_cloud_cover_middle_cloud",
              "high": "High_cloud_cover_high_cloud", "total": "Total_cloud_cover_entire_atmosphere"}
U_VAR, V_VAR = "u-component_of_wind_isobaric", "v-component_of_wind_isobaric"
LEVELS = [850, 700, 500, 300, 250]   # hPa
CLOUD_STRIDE = 3        # 0.75 degree
WIND_STRIDE = 6         # 1.5 degree: the steering wind is smooth, and this keeps the five levels to ~12 MB in all
TIMEOUT = 90
RETRIES = 3


def log(msg):
    print(msg, file=sys.stderr)


def get(url, timeout=TIMEOUT):
    last = None
    for i in range(RETRIES):
        try:
            req = urllib.request.Request(url, headers={"User-Agent": UA})
            with urllib.request.urlopen(req, timeout=timeout) as r:
                return r.read()
        except urllib.error.HTTPError as e:
            if 400 <= e.code < 500:            # a bad request won't get better by asking again
                raise
            last = e
        except Exception as e:                  # timeouts, resets
            last = e
        time.sleep(2 + 4 * i)
    raise last


# ---------------------------------------------------------------------------------------------- the source

def cycles():
    """The runs on the server, newest first, as (datetime, dataset path)."""
    xml = get(CATALOG_URL).decode("utf-8", "replace")
    out = []
    for path in set(re.findall(r'urlPath="([^"]*GFS_Global_0p25deg_(\d{8}_\d{4})\.grib2)"', xml)):
        t = datetime.datetime.strptime(path[1], "%Y%m%d_%H%M").replace(tzinfo=datetime.timezone.utc)
        out.append((t, path[0]))
    out.sort(reverse=True)
    return out


def hours_present(path):
    """Forecast hours present for each variable we need (the server lists a run while it's still arriving, with a
    shorter time axis). Read from the OPeNDAP descriptor and the time coordinates, a few KB."""
    dds = get(f"{THREDDS}/dodsC/{path}.dds").decode("utf-8", "replace")
    dims = {}
    for var in list(CLOUD_VARS.values()) + [U_VAR, V_VAR]:
        m = re.search(r"Float32 " + re.escape(var) + r"\[(\w+) = \d+\]", dds)
        if not m:
            return {}
        dims[var] = m.group(1)
    hours = {}
    for tdim in set(dims.values()):
        text = get(f"{THREDDS}/dodsC/{path}.ascii?{tdim}").decode("ascii", "replace")
        vals = text.split("\n", 2)[-1] if "---" not in text else text.split("---")[-1].split("\n", 2)[-1]
        hours[tdim] = {round(float(v)) for v in re.findall(r"-?\d+(?:\.\d+)?", vals)}
    return {var: hours[d] for var, d in dims.items()}


def read_nc(data):
    """{name: (array, dimension names, units)} from a netCDF-4 (HDF5) or netCDF-3 file in memory."""
    out = {}
    if data[:4] == bytes([0x89]) + b"HDF":
        with h5py.File(io.BytesIO(data), "r") as f:
            for k, v in f.items():
                if not isinstance(v, h5py.Dataset):
                    continue
                dims = (tuple(f[d[0]].name.lstrip("/") for d in v.attrs["DIMENSION_LIST"])
                        if "DIMENSION_LIST" in v.attrs else (k,) * v.ndim)
                units = v.attrs.get("units", b"")
                out[k] = (v[()], dims, units.decode() if isinstance(units, bytes) else str(units))
        return out
    from scipy.io import netcdf_file
    with netcdf_file(io.BytesIO(data), "r", mmap=False) as f:
        for k, v in f.variables.items():
            units = getattr(v, "units", b"")
            out[k] = (v[:].copy(), tuple(v.dimensions), units.decode() if isinstance(units, bytes) else str(units))
    return out


def hours_after(values, units, init):
    """Forecast hours from a time coordinate ("Hour since 2026-11-06T00:00:00Z")."""
    m = re.match(r"\s*hours?\s+since\s+(\S+)", units, re.I)
    base = datetime.datetime.fromisoformat(m.group(1).replace("Z", "+00:00")) if m else init
    if base.tzinfo is None:
        base = base.replace(tzinfo=datetime.timezone.utc)
    shift = (base - init).total_seconds() / 3600.0
    return [round(float(h) + shift) for h in np.atleast_1d(values)]


def subset(path, init, names, block, stride, level=None):
    h0, h1, step = block
    q = [("var", n) for n in names] + [
        ("horizStride", stride), ("accept", "netcdf4"),
        ("time_start", f"{init + datetime.timedelta(hours=h0):%Y-%m-%dT%H:%M:%SZ}"),
        ("time_end", f"{init + datetime.timedelta(hours=h1):%Y-%m-%dT%H:%M:%SZ}"), ("timeStride", step)]
    if level is not None:
        q.append(("vertCoord", level * 100))     # Pa
    data = get(f"{THREDDS}/ncss/grid/{path}?{urllib.parse.urlencode(q)}", timeout=TIMEOUT)
    nc = read_nc(data)
    out = {}
    for n in names:
        a, dims, _ = nc[n]
        # The time coordinate's name varies from run to run (time, time1, ...): take the variable's own first one.
        tvals, _, tunits = nc[dims[0]]
        hours = hours_after(tvals, tunits, init)
        a = np.asarray(a, np.float32).reshape(len(hours), a.shape[-2], a.shape[-1])   # drop the single level
        out[n] = {h: a[i] for i, h in enumerate(hours)}
    lat, lon = nc[nc[names[0]][1][-2]][0], nc[nc[names[0]][1][-1]][0]
    return out, np.asarray(lat, np.float64), np.asarray(lon, np.float64), len(data)


def fetch(path, init):
    """Downloads everything for one cycle. Returns ({field: {hour: 2-D array}}, grids, bytes); fields are the cloud
    names and ('u', hPa) / ('v', hPa)."""
    jobs = [("cloud", None, b) for b in BLOCKS] + [("wind", lev, b) for lev in LEVELS for b in BLOCKS]

    def run(job):
        kind, lev, block = job
        try:
            if kind == "cloud":
                return job, subset(path, init, list(CLOUD_VARS.values()), block, CLOUD_STRIDE)
            return job, subset(path, init, [U_VAR, V_VAR], block, WIND_STRIDE, lev)
        except urllib.error.HTTPError as e:
            # A run still arriving answers 400 for hours it doesn't have yet: those frames get padded.
            if e.code == 400 and block != BLOCKS[0]:
                log(f"{kind} {lev or ''} f{block[0]:03d}-f{block[1]:03d}: not on the server yet")
                return job, None
            raise

    fields, grids, size = {}, {}, 0
    with concurrent.futures.ThreadPoolExecutor(max_workers=6) as ex:
        for (kind, lev, _), got in ex.map(run, jobs):
            if got is None:
                continue
            data, lat, lon, n = got
            size += n
            grids[kind] = (lat, lon)
            for name, by_hour in data.items():
                if kind == "cloud":
                    key = next(k for k, v in CLOUD_VARS.items() if v == name)
                else:
                    key = ("u" if name == U_VAR else "v", lev)
                fields.setdefault(key, {}).update(by_hour)
    return fields, grids, size


# ---------------------------------------------------------------------------------------------- maps

def smoothstep(a, b, x):
    t = np.clip((x - a) / (b - a), 0.0, 1.0)
    return t * t * (3.0 - 2.0 * t)


OUT_LAT = 90.0 - (np.arange(TILE_H) + 0.5) * 180.0 / TILE_H
OUT_LON = -180.0 + (np.arange(TILE_W) + 0.5) * 360.0 / TILE_W


def to_tile(field, lat, lon):
    """Bilinear resample of a GFS grid (rows from 90 N south, columns from 0 E eastward round the globe) to a
    TILE_W x TILE_H map with column 0 at 180 W and row 0 at 90 N."""
    field = np.nan_to_num(field, nan=0.0)
    if lat[0] < lat[-1]:                                    # make rows run north to south
        lat, field = lat[::-1], field[::-1]
    fy = np.interp(OUT_LAT, lat[::-1], np.arange(len(lat), dtype=np.float64)[::-1])
    dlon = 360.0 / len(lon)
    fx = ((OUT_LON - lon[0]) % 360.0) / dlon
    y0 = np.clip(np.floor(fy).astype(int), 0, len(lat) - 2)
    ty = np.clip(fy - y0, 0, 1)[:, None]
    x0 = np.floor(fx).astype(int) % len(lon)
    x1 = (x0 + 1) % len(lon)
    tx = (fx - np.floor(fx))[None, :]
    a = field[y0][:, x0] * (1 - tx) + field[y0][:, x1] * tx
    b = field[y0 + 1][:, x0] * (1 - tx) + field[y0 + 1][:, x1] * tx
    return a * (1 - ty) + b * ty


# Each layer's look in the live (infrared) images, which see warm low cloud only faintly and thin cirrus as half
# there: a full mid deck reads nearly opaque, a full low deck about 0.6, a full high deck (usually cirrus) under half;
# a column with all three full (deep convection) comes out near 1. Tuned so the f+0 tile's opacity histogram within
# 60 degrees matches the live images' (2026-11-06 00Z against the 05Z live map: 52/10/10/5/24 % in fifths of the
# range against 56/10/7/6/21 %, mean 0.36 against 0.33).
LAYER_OPACITY = {"low": 0.6, "mid": 0.85, "high": 0.45}
CURVE = (0.25, 0.9)     # the combined opacity's smoothstep: scattered cloud fades out, solid decks reach 1
# Selection of the layer whose top is seen: cover above about 15 % starts to count, 45 % is decisive.
SEEN = (0.15, 0.45)
# Steering weights over LEVELS (850, 700, 500, 300, 250 hPa) for the layer whose top is seen, and for clear sky.
STEER = {"low": [0.55, 0.35, 0.10, 0.0, 0.0], "mid": [0.10, 0.35, 0.40, 0.10, 0.05],
         "high": [0.0, 0.0, 0.15, 0.45, 0.40], "clear": [0.0, 0.5, 0.5, 0.0, 0.0]}
# Pressure thickness each level stands for (hPa), so the mean is pressure-weighted.
DP = np.array([150.0, 175.0, 200.0, 125.0, 50.0])


def cloud_maps(low, mid, high, total):
    """(opacity 0-1, top km, selection weights {layer: map}) from GFS layer cover in percent, on the output grid."""
    l, m, h = (np.clip(x / 100.0, 0.0, 1.0) for x in (low, mid, high))
    t = np.clip(total / 100.0, 0.0, 1.0)
    # Random overlap of the three layers, each with its own opacity, capped by GFS's own total cover, then a gentle
    # curve that keeps scattered cloud faint and lets solid decks reach 1.
    op = 1.0 - (1.0 - LAYER_OPACITY["low"] * l) * (1.0 - LAYER_OPACITY["mid"] * m) * (1.0 - LAYER_OPACITY["high"] * h)
    op = smoothstep(*CURVE, np.minimum(op, t * 1.05))
    # A deep column (thick high cloud over mid and low) is convective or frontal: as opaque as the live images show it.
    deep = h * m * np.maximum(l, m)
    op = np.maximum(op, smoothstep(0.3, 0.8, deep))
    # Which layer's top is seen: the highest with meaningful cover.
    sh, sm, sl = (smoothstep(*SEEN, x) for x in (h, m, l))
    w = {"high": sh, "mid": (1 - sh) * sm, "low": (1 - sh) * (1 - sm) * sl}
    w["clear"] = 1.0 - w["high"] - w["mid"] - w["low"]
    # Heights: high cloud tops follow the tropopause (about 12 km in the tropics, 10 km poleward of 50 degrees), mid
    # about 5.5 km, low 2.5 km in the tropics to 1.5 km near the poles. A deep tropical column tops out a little
    # higher, like the cold anvils in the live images (kept modest: the live heights, from infrared, run lower).
    lat = np.abs(OUT_LAT)[:, None]
    hi_km = 10.0 + 2.0 * smoothstep(50.0, 20.0, lat) + 1.0 * smoothstep(0.4, 0.9, deep) * smoothstep(40.0, 20.0, lat)
    lo_km = 1.5 + 1.0 * smoothstep(60.0, 15.0, lat)
    top = w["high"] * hi_km + w["mid"] * 5.5 + (w["low"] + w["clear"]) * lo_km
    return np.clip(op, 0.0, 1.0), top, w


def steering(winds, w):
    """Pressure-weighted mean wind (km/h) over LEVELS, weighted toward the seen layer's levels."""
    u = np.zeros_like(w["clear"])
    v = np.zeros_like(w["clear"])
    norm = np.zeros_like(w["clear"])
    for layer in ("low", "mid", "high", "clear"):
        k = np.array(STEER[layer]) * DP
        k = k / k.sum()
        for i, lev in enumerate(LEVELS):
            if k[i] == 0.0:
                continue
            u += w[layer] * k[i] * winds[("u", lev)]
            v += w[layer] * k[i] * winds[("v", lev)]
        norm += w[layer]
    return u / np.maximum(norm, 1e-6) * 3.6, v / np.maximum(norm, 1e-6) * 3.6


def nearest(by_hour, hour):
    return by_hour[min(by_hour, key=lambda h: (abs(h - hour), h))]


def build(fields, grids):
    """The two 2048 x 1024 images, as float arrays (H, W, 2), and the number of frames padded."""
    cw, ch = GRID
    cloud_img = np.zeros((ch * TILE_H, cw * TILE_W, 2), np.float32)
    wind_img = np.zeros_like(cloud_img)
    padded = 0
    rng = np.random.default_rng(0)
    for i, hour in enumerate(FRAMES):
        missing = [k for k, v in fields.items() if hour not in v]
        padded += bool(missing)
        if missing:
            log(f"f{hour:03d}: missing {missing}, repeating the nearest frame")
        c = {k: to_tile(nearest(fields[k], hour), *grids["cloud"]) for k in CLOUD_VARS}
        op, top, w = cloud_maps(c["low"], c["mid"], c["high"], c["total"])
        winds = {key: to_tile(nearest(fields[key], hour), *grids["wind"]) for key in fields if isinstance(key, tuple)}
        u, v = steering(winds, w)
        r, col = divmod(i, cw)
        ys, xs = slice(r * TILE_H, (r + 1) * TILE_H), slice(col * TILE_W, (col + 1) * TILE_W)
        # Dither the cloud channels so 8 bits don't band, as the live images do.
        cloud_img[ys, xs, 0] = op + (rng.random(op.shape) - 0.5) / 255.0
        cloud_img[ys, xs, 1] = top / TOP_KM + (rng.random(op.shape) - 0.5) / 255.0
        wind_img[ys, xs, 0] = u / (2.0 * WIND_KMH) + 0.5
        wind_img[ys, xs, 1] = v / (2.0 * WIND_KMH) + 0.5
    return cloud_img, wind_img, padded


def save(img, path):
    px = np.zeros(img.shape[:2] + (3,), np.uint8)
    px[..., :2] = (np.clip(img, 0, 1) * 255 + 0.5).astype(np.uint8)
    px[..., 2] = 128   # the sRGB-decode marker the sky shader checks
    Image.fromarray(px).save(path, optimize=True)


# ---------------------------------------------------------------------------------------------- main

def choose(runs, wanted_cycle=None):
    """The newest run with every frame present; else the one with the most frames (to be padded)."""
    best = None
    for t, path in runs[:4]:
        if wanted_cycle and t != wanted_cycle:
            continue
        try:
            have = hours_present(path)
        except Exception as e:
            log(f"{t:%Y-%m-%d %H}Z: can't read its time axis ({type(e).__name__}: {e})")
            continue
        if not have:
            log(f"{t:%Y-%m-%d %H}Z: variables missing")
            continue
        got = min(sum(h in hs for h in FRAMES) for hs in have.values())
        if got == len(FRAMES):
            return t, path, got
        log(f"{t:%Y-%m-%d %H}Z: incomplete ({got} of {len(FRAMES)} frames so far)")
        if got and (best is None or got > best[2]):
            best = (t, path, got)
    if best is None:
        raise SystemExit("no usable GFS cycle on the server")
    return best


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--out", required=True, help="folder to write forecast.png, winds.png and forecast.json into")
    ap.add_argument("--cycle", help="use this cycle (YYYY-MM-DDTHH) instead of the newest complete one")
    a = ap.parse_args()
    t0 = time.time()
    wanted = None
    if a.cycle:
        wanted = datetime.datetime.strptime(a.cycle, "%Y-%m-%dT%H").replace(tzinfo=datetime.timezone.utc)
    init, path, got = choose(cycles(), wanted)
    fields, grids, size = None, None, 0
    for attempt in range(2):
        try:
            fields, grids, size = fetch(path, init)
            break
        except Exception as e:
            if attempt:
                raise
            log(f"download failed ({type(e).__name__}: {e}); trying once more")
            time.sleep(10)
    cloud_img, wind_img, padded = build(fields, grids)
    os.makedirs(a.out, exist_ok=True)
    save(cloud_img, os.path.join(a.out, "forecast.png"))
    save(wind_img, os.path.join(a.out, "winds.png"))
    doc = {"init": f"{init:%Y-%m-%dT%H:%M}Z", "frames": FRAMES, "grid": list(GRID), "tile": [TILE_W, TILE_H],
           "wind_kmh": round(WIND_KMH), "top_km": round(TOP_KM)}
    with open(os.path.join(a.out, "forecast.json"), "w") as f:
        json.dump(doc, f, separators=(", ", ": "))
        f.write("\n")
    mid = np.abs(np.tile(OUT_LAT, GRID[1])) < 60
    print(f"GFS {init:%Y-%m-%d %H}Z, {len(FRAMES)} frames to f{FRAMES[-1]:03d} ({padded} padded), "
          f"{size / 1e6:.1f} MB in {time.time() - t0:.0f} s, cloud cover {cloud_img[mid, :, 0].mean():.2f} within 60 degrees")


if __name__ == "__main__":
    main()
