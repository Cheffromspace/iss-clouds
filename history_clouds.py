"""The last 48 hours of satellite clouds for the observatory's seek bar (docs/HISTORY-FORMAT.md).

Run by the hourly iss-clouds job right after live_clouds.py has written west.png, east.png and time.txt into --out.
Keeps 16 frames, every 3 hours (the newest at the latest 3-hourly hour at or before the live image), each 1024 x 512
in the live images' encoding (red opacity, green cloud-top height / 16 km, blue the 128 marker). Frames already
published are fetched back from the site (each run replaces the whole site, so they have to be carried forward); the
newest comes from the live image when it is on the hour, and anything else missing is rebuilt from NOAA's GMGSI archive,
at most --backfill a run so the job stays inside its time limit. Frames still missing repeat the nearest one.

Writes hist/YYYYMMDDTHH.png (the frames, for the next run), hist_a.png and hist_b.png (2048 x 2048 atlases, 8 frames
each, 2 across and 4 down, oldest first) and hist.json.
"""
import argparse
import datetime
import io
import json
import os
import tempfile
import urllib.error
import urllib.request

import numpy as np
from PIL import Image

import live_clouds as lc

FRAMES, STEP_H = 16, 3
TW, TH = 1024, 512


def frame_from_live(out):
    west = np.asarray(Image.open(os.path.join(out, "west.png")).convert("RGB"))
    east = np.asarray(Image.open(os.path.join(out, "east.png")).convert("RGB"))
    return shrink(np.concatenate([west, east], axis=1))


def shrink(px):
    """4096 x 2048 -> 1024 x 512, averaging opacity and height (area mean), blue back to the 128 marker."""
    h, w = px.shape[:2]
    a = px[..., :2].astype(np.float32).reshape(TH, h // TH, TW, w // TW, 2).mean(axis=(1, 3))
    out = np.zeros((TH, TW, 3), np.uint8)
    out[..., :2] = np.clip(a + 0.5, 0, 255).astype(np.uint8)
    out[..., 2] = 128
    return out


def fetch_published(site, t):
    url = f"{site.rstrip('/')}/hist/{t:%Y%m%dT%H}.png"
    try:
        with urllib.request.urlopen(url, timeout=60) as r:
            px = np.asarray(Image.open(io.BytesIO(r.read())).convert("RGB"))
        return px if px.shape == (TH, TW, 3) else None
    except (urllib.error.URLError, OSError, ValueError):
        return None


def rebuild(t, past_days):
    """The frame for hour t from the GMGSI archive, as live_clouds.py would have made it then (infrared only)."""
    with tempfile.TemporaryDirectory() as tmp:
        got = lc.fetch_hour(tmp, t, past_days)
        if got is None:
            return None
        now_path, past_paths = got
        now, lat, lon = lc.load(now_path)
        past = [lc.load(p)[0] for p in past_paths]
        op, top = lc.clouds(now, past)
        op, top = lc.deseam(op, top)
        px, _ = lc.encode(op, top, lat, lon)
    return shrink(px)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--out", default="site", help="the site folder live_clouds.py wrote into")
    p.add_argument("--site", default="https://cheffromspace.github.io/iss-clouds", help="the published site")
    p.add_argument("--backfill", type=int, default=16, help="most frames to rebuild from the archive this run")
    p.add_argument("--past-days", type=int, default=3, help="days of the same hour for the clear-sky reference")
    a = p.parse_args()
    with open(os.path.join(a.out, "time.txt")) as f:
        live = datetime.datetime.strptime(f.read().strip(), "%Y-%m-%dT%H:%MZ").replace(tzinfo=datetime.timezone.utc)
    newest = live.replace(hour=live.hour - live.hour % STEP_H)
    hours = [newest - datetime.timedelta(hours=STEP_H * k) for k in range(FRAMES)][::-1]   # oldest first
    frames, budget = {}, a.backfill
    for t in reversed(hours):   # newest first, so a short budget fills the recent end
        px = frame_from_live(a.out) if t == live else fetch_published(a.site, t)
        if px is None and budget > 0:
            budget -= 1
            try:
                px = rebuild(t, a.past_days)
            except Exception as e:   # one bad archive file shouldn't lose the rest
                print(f"rebuild {t:%Y-%m-%d %H}Z failed: {e}")
        if px is not None:
            frames[t] = px
    if not frames:
        raise SystemExit("no history frames")
    have = sorted(frames)
    os.makedirs(os.path.join(a.out, "hist"), exist_ok=True)
    for t in have:
        Image.fromarray(frames[t]).save(os.path.join(a.out, "hist", f"{t:%Y%m%dT%H}.png"), optimize=True)
    # Atlases: every slot filled, a missing hour repeating the nearest frame we have; hist.json says which are real.
    names, real = [], []
    for t in hours:
        near = min(have, key=lambda h: abs((h - t).total_seconds()))
        names.append(near)
        real.append(near == t)
    for k, name in enumerate(("hist_a.png", "hist_b.png")):
        atlas = np.zeros((TH * 4, TW * 2, 3), np.uint8)
        for j in range(8):
            row, col = divmod(j, 2)
            atlas[row * TH:(row + 1) * TH, col * TW:(col + 1) * TW] = frames[names[k * 8 + j]]
        Image.fromarray(atlas).save(os.path.join(a.out, name), optimize=True)
    with open(os.path.join(a.out, "hist.json"), "w") as f:
        json.dump({"frames": [f"{t:%Y-%m-%dT%H}:00Z" for t in hours], "real": real, "step_h": STEP_H,
                   "grid": [2, 4], "tile": [TW, TH], "atlases": ["hist_a.png", "hist_b.png"]}, f)
    print(f"history: {sum(real)} of {FRAMES} frames, {newest:%Y-%m-%d %H}Z back {STEP_H * (FRAMES - 1)} h")


if __name__ == "__main__":
    main()
