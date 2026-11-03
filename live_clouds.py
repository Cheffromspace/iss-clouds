"""Turn NOAA's hourly global infrared mosaic (GMGSI) into cloud images the ISS world downloads.

Source: NOAA GMGSI longwave infrared, s3://noaa-gmgsi-pds/GMGSI_LW/YYYY/MM/DD/HH/*.nc (public, no account).
It's an hourly blend of the GOES, Meteosat and Himawari geostationary satellites on a 0.0722 degree grid
from 72.7 N to 72.7 S. NOAA data is open for any use.

Infrared sees cloud tops as colder (brighter) than the ground. Cold clear ground (Siberia in winter, high
mountains) is bright too, so each pixel is compared with its clear-sky value: the warmest reading at the same
hour over the previous days. Pass those files as --past.

Output: two 2048 x 2048 RGB PNGs, west.png (180 W to 0) and east.png (0 to 180 E), each covering the full
90 N to 90 S. Together they make a 4096 x 2048 equirectangular map, close to the source's own resolution, within
VRChat's 2048 x 2048 limit per downloaded image. Red is cloud opacity (1 = opaque); green is cloud-top height,
0-16 km. Blue is unused. Beyond the mosaic's latitude
range the image is black and the world falls back to its fixed map.

    python tools/live_clouds.py --fetch --out folder       # latest hour, straight from NOAA
    python tools/live_clouds.py now.nc --past day1.nc ... --out folder
"""
import argparse
import datetime
import os
import re
import tempfile
import urllib.request

import h5py
import numpy as np
from PIL import Image

W, H = 4096, 2048
BUCKET = "https://noaa-gmgsi-pds.s3.amazonaws.com"
PAST_DAYS = 7


def keys(prefix):
    with urllib.request.urlopen(f"{BUCKET}/?list-type=2&prefix={prefix}", timeout=60) as r:
        return sorted(re.findall(r"<Key>([^<]+\.nc)</Key>", r.read().decode()))


def fetch(folder):
    """Downloads the newest hourly file and the same hour on each of the previous days. Returns (now, past, hour)."""
    t = datetime.datetime.now(datetime.timezone.utc)
    for _ in range(6):   # the newest hour appears about 40 minutes after it ends
        found = keys(t.strftime("GMGSI_LW/%Y/%m/%d/%H/"))
        if found:
            break
        t -= datetime.timedelta(hours=1)
    else:
        raise SystemExit("no GMGSI file in the last 6 hours")
    paths = []
    for day in range(PAST_DAYS + 1):
        k = found if day == 0 else keys((t - datetime.timedelta(days=day)).strftime("GMGSI_LW/%Y/%m/%d/%H/"))
        if not k:
            continue
        path = os.path.join(folder, f"{day}.nc")
        urllib.request.urlretrieve(f"{BUCKET}/{k[-1]}", path)
        paths.append(path)
    print(f"{t:%Y-%m-%d %H}:00 UTC with {len(paths) - 1} past days")
    return paths[0], paths[1:], t


def load(path):
    with h5py.File(path, "r") as f:
        d = f["data"][0].astype(np.float32)
        lat = f["lat"][:, 0]
        lon = f["lon"][0]
    d[d < 0] = np.nan
    return d, lat, lon


def smoothstep(a, b, x):
    t = np.clip((x - a) / (b - a), 0.0, 1.0)
    return t * t * (3.0 - 2.0 * t)


# Cloud-top height from how much colder the top is than the clear-sky ground. Clear tropical ocean reads about 67 and
# the coldest storm tops about 232, roughly 295 K and 195 K, so about 1.65 counts per kelvin. Air cools about 6.5 K
# per km, which makes about 10.5 counts per km.
COUNTS_PER_KM = 10.5
TOP_KM = 16.0   # the height stored as 1.0; the sky shader uses the same scale


def clouds(now, past):
    """Returns (opacity 0-1, cloud-top height in km)."""
    # Counts run 0-255 with colder (higher) as brighter. Clear sky is the lowest count seen at this hour lately.
    clear = np.nanmin(np.stack(past + [now]), axis=0) if past else np.full_like(now, 70.0)
    rise = now - clear
    # A rise over clear sky means cloud. Very cold tops are cloud whatever the clear-sky value says.
    op = np.maximum(smoothstep(6.0, 45.0, rise), smoothstep(150.0, 205.0, now))
    top = np.clip(np.maximum(rise, (now - 70.0) * 0.8) / COUNTS_PER_KM, 0.5, TOP_KM)
    return np.nan_to_num(op, nan=0.0), np.nan_to_num(top, nan=0.5)


def to_equirect(op, lat, lon):
    # Resample the source grid to a full-globe equirectangular map: column 0 at 180 W, row 0 at 90 N.
    rows = H
    out_lat = 90.0 - (np.arange(rows) + 0.5) * 180.0 / rows
    out_lon = -180.0 + (np.arange(W) + 0.5) * 360.0 / W
    # Source rows run north to south at a fixed step; source columns start at 180 and wrap round.
    dlat = (lat[0] - lat[-1]) / (len(lat) - 1)
    dlon = 360.0 / len(lon)
    fy = (lat[0] - out_lat) / dlat
    fx = ((out_lon - lon[0]) % 360.0) / dlon
    inside = (fy >= 0) & (fy <= len(lat) - 1)
    y0 = np.clip(np.floor(fy).astype(int), 0, len(lat) - 2)
    ty = np.clip(fy - y0, 0, 1)[:, None]
    x0 = np.floor(fx).astype(int) % len(lon)
    x1 = (x0 + 1) % len(lon)
    tx = (fx - np.floor(fx))[None, :]
    a = op[y0][:, x0] * (1 - tx) + op[y0][:, x1] * tx
    b = op[y0 + 1][:, x0] * (1 - tx) + op[y0 + 1][:, x1] * tx
    img = a * (1 - ty) + b * ty
    img[~inside] = 0.0
    return img


def main():
    p = argparse.ArgumentParser()
    p.add_argument("now", nargs="?")
    p.add_argument("--fetch", action="store_true", help="download the inputs from NOAA")
    p.add_argument("--past", nargs="*", default=[])
    p.add_argument("--out", default=".")
    p.add_argument("--preview", action="store_true", help="also write full.png, the whole 4096 x 2048 map")
    a = p.parse_args()
    tmp = tempfile.TemporaryDirectory()
    hour = None
    if a.fetch:
        a.now, a.past, hour = fetch(tmp.name)
    now, lat, lon = load(a.now)
    past = [load(f)[0] for f in a.past]
    op, top = clouds(now, past)
    img = to_equirect(op, lat, lon)
    height = to_equirect(top / TOP_KM, lat, lon)
    px = np.zeros((H, W, 3), np.uint8)
    px[..., 0] = (np.clip(img, 0, 1) * 255 + 0.5).astype(np.uint8)
    px[..., 1] = (np.clip(height, 0, 1) * 255 + 0.5).astype(np.uint8)
    os.makedirs(a.out, exist_ok=True)
    Image.fromarray(px[:, : W // 2]).save(os.path.join(a.out, "west.png"), optimize=True)
    Image.fromarray(px[:, W // 2 :]).save(os.path.join(a.out, "east.png"), optimize=True)
    if a.preview:
        Image.fromarray(px).save(os.path.join(a.out, "full.png"))
    if hour is not None:
        with open(os.path.join(a.out, "time.txt"), "w") as f:
            f.write(f"{hour:%Y-%m-%dT%H}:00Z\n")
    tmp.cleanup()
    print(f"cloud cover {img[np.abs(90.0 - (np.arange(H) + 0.5) * 180.0 / H) < 60].mean():.2f} within 60 degrees")


if __name__ == "__main__":
    main()
