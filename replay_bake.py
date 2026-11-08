"""Bake a famous storm's whole life into seek-bar frames for the observatory (docs/HISTORY-FORMAT.md, "replays").

Source: NOAA's GridSat-B1 climate data record (s3://noaa-cdr-gridsat-b1-pds, public, no account): global
geostationary infrared every 3 hours since 1980, 0.07 degrees, 70 N to 70 S. Each frame becomes a 1024 x 512 global
map in the live images' encoding (red opacity, green cloud-top height / 16 km, blue the 128 marker), with the same
cloud-cover curve as tools/live_clouds.py, so a replay looks like the live clouds. The clear-sky reference is each
pixel's warm end over the whole replay (its 90th percentile), which suits a week of frames.

Writes <out>/<name>/replay.json and atlas_0.png, atlas_1.png, ... (2048 x 2048, 8 frames each, 2 across, 4 down,
oldest first). Downloads are cached in --cache, so a re-bake with other settings is quick. Run once per storm; the
iss-clouds job copies the results into the site.

  python replay_bake.py --name katrina2005 --title "Hurricane Katrina, 2005" --start 2005-08-24T00 --end 2005-08-30T12
"""
import argparse
import ctypes
import datetime
import json
import os
import sys
import urllib.request

import h5py
import numpy as np
from PIL import Image
from scipy import ndimage

BUCKET = "https://noaa-cdr-gridsat-b1-pds.s3.amazonaws.com"
TW, TH, PER = 1024, 512, 8
TOP_KM = 16.0


def below_normal():
    if sys.platform == "win32":
        ctypes.windll.kernel32.SetPriorityClass(ctypes.windll.kernel32.GetCurrentProcess(), 0x00004000)
    else:
        os.nice(10)


def fetch(cache, t):
    name = f"GRIDSAT-B1.{t:%Y.%m.%d.%H}.v02r01.nc"
    path = os.path.join(cache, name)
    if not os.path.exists(path):
        tmp = path + ".part"
        urllib.request.urlretrieve(f"{BUCKET}/data/{t:%Y}/{name}", tmp)
        os.replace(tmp, path)
    return path


def brightness(path):
    """Infrared brightness temperature (K) on the 0.07-degree grid, NaN where missing; and the grid's lat, lon."""
    with h5py.File(path, "r") as f:
        d = f["irwin_cdr"]
        raw = d[0].astype(np.float32)
        bt = raw * float(d.attrs["scale_factor"][0]) + float(d.attrs["add_offset"][0])
        bt[raw == int(d.attrs["_FillValue"][0])] = np.nan
        return bt, f["lat"][:], f["lon"][:]


def to_frame(op, top, lat, lon):
    """0.07-degree grid (70 N to 70 S) -> the 1024 x 512 global equirectangular frame, 90 N at the top."""
    out = np.zeros((TH, TW, 3), np.float32)
    # Each output pixel's centre, looked up on the source grid (area-mean first: 5 x 5 source cells per pixel).
    k = 5
    hh, ww = (op.shape[0] // k) * k, (op.shape[1] // k) * k
    op_s = np.nanmean(op[:hh, :ww].reshape(hh // k, k, ww // k, k), axis=(1, 3))
    top_s = np.nanmean(top[:hh, :ww].reshape(hh // k, k, ww // k, k), axis=(1, 3))
    lat_s = lat[:hh].reshape(-1, k).mean(axis=1)
    lon_s = lon[:ww].reshape(-1, k).mean(axis=1)
    oy = 90.0 - (np.arange(TH) + 0.5) * 180.0 / TH
    ox = -180.0 + (np.arange(TW) + 0.5) * 360.0 / TW
    ri = np.interp(oy, lat_s, np.arange(len(lat_s)), left=-1, right=-1)
    ci = np.interp(ox, lon_s, np.arange(len(lon_s)))
    rows = np.clip(np.round(ri).astype(int), 0, len(lat_s) - 1)
    cols = np.clip(np.round(ci).astype(int), 0, len(lon_s) - 1)
    o = op_s[rows][:, cols]
    tp = top_s[rows][:, cols]
    outside = (ri < 0)[:, None] | (np.abs(oy) > 69.9)[:, None]
    o = np.where(outside, 0.0, np.nan_to_num(o))
    tp = np.where(outside, 0.0, np.nan_to_num(tp))
    rng = np.random.default_rng(0)
    out[..., 0] = o + (rng.random(o.shape) - 0.5) / 255.0
    out[..., 1] = tp / TOP_KM + (rng.random(o.shape) - 0.5) / 255.0
    px = np.zeros((TH, TW, 3), np.uint8)
    px[..., :2] = (np.clip(out[..., :2], 0, 1) * 255 + 0.5).astype(np.uint8)
    px[..., 2] = 128
    return px


def smoothstep(a, b, x):
    t = np.clip((x - a) / (b - a), 0.0, 1.0)
    return t * t * (3.0 - 2.0 * t)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--name", required=True)
    p.add_argument("--title", required=True)
    p.add_argument("--start", required=True, help="first frame, UTC, YYYY-MM-DDTHH (a multiple of 3 h)")
    p.add_argument("--end", required=True)
    p.add_argument("--go", nargs=3, type=float, metavar=("LAT", "LON", "ALT_KM"), help="where Go takes you")
    p.add_argument("--cache", default="gridsat-cache")
    p.add_argument("--out", default="replays")
    a = p.parse_args()
    below_normal()
    parse = lambda s: datetime.datetime.strptime(s, "%Y-%m-%dT%H").replace(tzinfo=datetime.timezone.utc)
    t0, t1 = parse(a.start), parse(a.end)
    hours = []
    t = t0
    while t <= t1:
        hours.append(t)
        t += datetime.timedelta(hours=3)
    n = len(hours)
    hours += [hours[-1]] * ((-n) % PER)   # fill the last atlas with the final frame
    os.makedirs(a.cache, exist_ok=True)
    print(f"{a.name}: {n} frames, {len(hours) // PER} atlases")
    # Pass 1: download, and each frame's brightness on a 5 x 5 coarser grid for the clear-sky reference (the full grid
    # for a week of frames would need gigabytes).
    k = 5
    small = []
    paths = []
    for i, t in enumerate(hours[:n]):
        path = fetch(a.cache, t)
        paths.append(path)
        bt, lat, lon = brightness(path)
        hh, ww = (bt.shape[0] // k) * k, (bt.shape[1] // k) * k
        small.append(np.nanmean(bt[:hh, :ww].reshape(hh // k, k, ww // k, k), axis=(1, 3)).astype(np.float32))
        print(f"  {i + 1}/{n} {t:%Y-%m-%d %H}Z", flush=True)
    clear_s = np.nanpercentile(np.stack(small), 90, axis=0)   # each place's warm (clear-sky) end over the replay
    clear_s = ndimage.uniform_filter(np.nan_to_num(clear_s, nan=290.0), 3)
    del small
    # Pass 2: each frame at full resolution against the clear sky.
    frames = []
    for path in paths:
        bt, lat, lon = brightness(path)
        clear = np.kron(clear_s, np.ones((k, k), np.float32))
        clear = np.pad(clear, ((0, bt.shape[0] - clear.shape[0]), (0, bt.shape[1] - clear.shape[1])), mode="edge")
        rise = clear - bt                                  # kelvin colder than the clear-sky surface
        # live_clouds.py's curve in kelvin: counts 8-60 of rise = about 5-36 K; coldness counts 110-200 = 269-214 K.
        op = smoothstep(5.0, 36.0, rise) * (0.35 + 0.65 * smoothstep(269.0, 214.0, bt))
        top = np.clip(np.maximum(rise / 6.5, (295.0 - bt) / 8.0), 0.5, TOP_KM)
        op = np.where(np.isnan(bt), 0.0, op)
        top = np.where(np.isnan(bt), 0.5, top)
        op = ndimage.gaussian_filter(op, 1.0)
        frames.append(to_frame(op, top, lat, lon))
    frames += [frames[-1]] * (len(hours) - n)
    od = os.path.join(a.out, a.name)
    os.makedirs(od, exist_ok=True)
    atlases = []
    for k in range(len(hours) // PER):
        atlas = np.zeros((TH * 4, TW * 2, 3), np.uint8)
        for j in range(PER):
            row, col = divmod(j, 2)
            atlas[row * TH:(row + 1) * TH, col * TW:(col + 1) * TW] = frames[k * PER + j]
        name = f"atlas_{k}.png"
        Image.fromarray(atlas).save(os.path.join(od, name), optimize=True)
        atlases.append(name)
    doc = {"name": a.name, "title": a.title, "frames": [f"{t:%Y-%m-%dT%H}:00Z" for t in hours[:n]], "step_h": 3,
           "grid": [2, 4], "tile": [TW, TH], "atlases": atlases,
           "source": "NOAA GridSat-B1 CDR (infrared), via AWS Open Data"}
    if a.go:
        doc["go"] = {"lat": a.go[0], "lon": a.go[1], "alt_km": a.go[2]}
    with open(os.path.join(od, "replay.json"), "w") as f:
        json.dump(doc, f, indent=1)
    print(f"wrote {od}: {len(atlases)} atlases")


if __name__ == "__main__":
    main()
