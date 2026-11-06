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
0-16 km. Blue is a constant 128, a marker for the sRGB check in the sky shader. Beyond the mosaic's latitude
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
from scipy import ndimage

W, H = 4096, 2048
BUCKET = "https://noaa-gmgsi-pds.s3.amazonaws.com"
PAST_DAYS = 7


def keys(prefix):
    with urllib.request.urlopen(f"{BUCKET}/?list-type=2&prefix={prefix}", timeout=60) as r:
        return sorted(re.findall(r"<Key>([^<]+\.nc)</Key>", r.read().decode()))


def fetch(folder, with_vis=False):
    """Downloads the newest hourly infrared (and, if asked, visible) files and the same hour on each of the previous days.

    Returns (ir_now, ir_past, vis_now, vis_past, hour, ir_prev_hour); vis_now and ir_prev_hour are None if missing.
    """
    t = datetime.datetime.now(datetime.timezone.utc)
    for _ in range(6):   # the newest hour appears about 40 minutes after it ends
        found = keys(t.strftime("GMGSI_LW/%Y/%m/%d/%H/"))
        if found:
            break
        t -= datetime.timedelta(hours=1)
    else:
        raise SystemExit("no GMGSI file in the last 6 hours")
    paths = {"LW": [], "VIS": []} if with_vis else {"LW": []}
    for day in range(PAST_DAYS + 1):
        for product in paths:
            k = keys((t - datetime.timedelta(days=day)).strftime(f"GMGSI_{product}/%Y/%m/%d/%H/"))
            if not k:
                continue
            path = os.path.join(folder, f"{product}_{day}.nc")
            urllib.request.urlretrieve(f"{BUCKET}/{k[-1]}", path)
            paths[product].append((day, path))
    # The hour before, for cloud motion (flow.png).
    k = keys((t - datetime.timedelta(hours=1)).strftime("GMGSI_LW/%Y/%m/%d/%H/"))
    prev = None
    if k:
        prev = os.path.join(folder, "LW_prev.nc")
        urllib.request.urlretrieve(f"{BUCKET}/{k[-1]}", prev)
    ir = dict(paths["LW"])
    vis = dict(paths.get("VIS", []))
    print(f"{t:%Y-%m-%d %H}:00 UTC with {len(ir) - 1} past infrared days, {len(vis)} visible files")
    return ir[0], [p for d, p in sorted(ir.items()) if d], vis.get(0), [p for d, p in sorted(vis.items()) if d], t, prev


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


def sun_cos(lat, lon, t):
    """Cosine of the Sun's zenith angle over the grid at UTC time t (declination and hour angle; no equation of time)."""
    day = t.timetuple().tm_yday
    decl = np.radians(-23.44) * np.cos(2.0 * np.pi * (day + 10) / 365.0)
    hour_angle = np.radians((t.hour + t.minute / 60.0 - 12.0) * 15.0 + lon)[None, :]
    la = np.radians(lat)[:, None]
    return np.sin(la) * np.sin(decl) + np.cos(la) * np.cos(decl) * np.cos(hour_angle)


def clouds_vis(vis, vis_past, cosz):
    """Opacity from the visible mosaic, which sees cloud as a camera does: by how much brighter than clear sky it is.

    Clear sky is the darkest value seen at this hour over the previous days: the Sun stands at the same angle each
    day, so the ocean glint, deserts and snow are in that baseline and cancel out. Returns (opacity, valid), with
    valid false on the night side and where no visible data exists.
    """
    stack = []
    for v in vis_past + [vis]:
        v = v.copy()
        v[v <= 0.0] = np.nan
        stack.append(v)
    clear = np.nanmin(np.stack(stack), axis=0)
    rise = (stack[-1] - clear) / np.maximum(cosz, 0.15)
    # Compared against true-colour imagery: a solid low deck rises about 60 counts over the ocean, a thick storm top
    # about 150.
    op = smoothstep(12.0, 110.0, rise)
    valid = np.isfinite(rise) & (cosz > 0.1)
    return np.nan_to_num(op, nan=0.0), valid


def blend(op_ir, top, op_vis, valid, cosz):
    """Visible opacity where the Sun is up, infrared elsewhere, with the infrared calibrated against the visible
    where both exist (per cloud-top height band) so the night side matches the day side."""
    overlap = valid & (cosz > 0.35)
    edges = [0.0, 3.0, 6.0, 9.0, 12.0, 17.0]
    centres, scales = [], []
    for a, b in zip(edges[:-1], edges[1:]):
        m = overlap & (top >= a) & (top < b) & ((op_ir > 0.05) | (op_vis > 0.05))
        if m.sum() < 2000:
            continue
        centres.append(0.5 * (a + b))
        scales.append(float(np.clip(op_vis[m].mean() / max(op_ir[m].mean(), 1e-3), 0.3, 1.5)))
    if centres:
        scale = np.interp(top, centres, scales)
        print("infrared opacity scaled to visible by top band: " + ", ".join(f"{c:.1f} km x{s:.2f}" for c, s in zip(centres, scales)))
    else:
        scale = 1.0
    op_night = np.clip(op_ir * scale, 0.0, 1.0)
    w = smoothstep(0.12, 0.35, cosz) * valid
    op = op_night * (1.0 - w) + op_vis * w
    # Warm low cloud (fog, stratus) is invisible to infrared but plain in daylight: give it a low top.
    top = np.where(w * op_vis > op_night + 0.2, np.maximum(top, 1.5), top)
    return op, top


def deseam(op, top):
    """Hides the joins between satellites in the mosaic.

    Each satellite sees the same cloud top at a different angle, so where the mosaic switches from one to the next
    (about 137 W, 22 E and 93 E) the heights, and sometimes the opacity, step by up to a kilometre along a straight
    meridian. In the world that draws as a crease across every cloud deck it crosses. Finds those columns from the
    data itself (a column where cloud tops jump between neighbours in far more rows than usual), then blends each
    field across the join: the two sides are shifted to meet, and the shift fades out over about 60 km each way.
    """
    cloudy = op > 0.3
    both = cloudy[:, 1:] & cloudy[:, :-1]
    rows = np.maximum(both.sum(axis=0), 1)
    for field, least in ((top, 0.3), (op, 0.08)):
        # The mean signed step between neighbouring columns over rows where both are cloudy. Pixel noise averages
        # out; a join keeps its sign down the whole column. Real cloud edges don't line up along a meridian.
        mean_step = (np.diff(field, axis=1) * both).sum(axis=0) / rows
        mean_step[rows < 200] = 0.0
        size = np.abs(mean_step)
        floor = max(15.0 * np.median(size[rows >= 200]), least)
        seams = [c for c in np.argsort(size)[::-1][:12] if size[c] > floor and size[c] >= size[max(c - 5, 0):c + 6].max()]
        for c in seams:
            # The mosaic already blends the two satellites over about 25 columns, so the join is an S-curve, not a
            # step. Take the mean profile across it over rows cloudy on both sides, and subtract whatever departs
            # from a straight line between the two plateaus: the 2 km drop then spreads over 650 km, which is flat.
            lo, hi = max(c - 45, 0), min(c + 46, field.shape[1])
            ok = (op[:, lo:lo + 20].mean(axis=1) > 0.3) & (op[:, hi - 20:hi].mean(axis=1) > 0.3)
            if ok.sum() < 100:
                continue
            profile = field[ok][:, lo:hi].mean(axis=0)
            line = np.linspace(profile[:5].mean(), profile[-5:].mean(), hi - lo)
            field[:, lo:hi] -= (profile - line)[None, :]
            print(f"seam at column {c} (mean step {mean_step[c]:+.2f}, {profile[:5].mean() - profile[-5:].mean():+.2f} across) flattened")
    return np.clip(op, 0.0, 1.0), top


def to_equirect(op, lat, lon):
    # Resample the source grid to a full-globe equirectangular map: column 0 at 180 W, row 0 at 90 N.
    rows = H
    out_lat = 90.0 - (np.arange(rows) + 0.5) * 180.0 / rows
    out_lon = -180.0 + (np.arange(W) + 0.5) * 360.0 / W
    # Source rows run north to south on a Mercator grid: the step is 0.072 degrees at the equator and 0.021 near
    # 72 N. Treating it as even put every cloud too close to the equator (a storm at 47 N drew at 36 N), so each
    # output row is looked up in the source's own latitudes. Source columns are even, start at 180 and wrap round.
    dlon = 360.0 / len(lon)
    fy = np.interp(out_lat, lat[::-1], np.arange(len(lat), dtype=np.float64)[::-1])
    fx = ((out_lon - lon[0]) % 360.0) / dlon
    inside = (out_lat <= lat[0]) & (out_lat >= lat[-1])
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


FLOW_W, FLOW_H = 512, 256
FLOW_KMH = 150.0   # the speed stored as 0 or 1 (0.5 is still); jet-level cloud rarely moves faster


def cloud_flow(now_raw, prev_raw, now_op, lat, lon):
    """How the clouds moved over the last hour, in km/h east and north, on a FLOW_W x FLOW_H equirectangular grid.

    Block matching, the way weather services track cloud motion: on a 2048 x 1024 map of infrared brightness (about
    20 km a pixel), each patch of this hour is compared with the last hour's shifted by every offset up to about 150
    km, and the best match wins (refined to a fraction of a pixel). Clear sky has nothing to track, so the result is
    weighted by cloud and texture and spread smoothly into the gaps.
    """
    w2, h2 = 2048, 1024

    def eq(f):
        img = to_equirect(np.nan_to_num(f, nan=0.0), lat, lon)
        return img.reshape(h2, H // h2, w2, W // w2).mean(axis=(1, 3))

    a, b, cloud = eq(now_raw), eq(prev_raw), eq(now_op)
    wrap = ("nearest", "wrap")
    # Keep the cloud texture, drop the slow changes (the day heating the ground, a deck thickening).
    a = a - ndimage.gaussian_filter(a, 12, mode=wrap)
    b = b - ndimage.gaussian_filter(b, 12, mode=wrap)
    R, block = 7, 11   # search radius and patch size, pixels
    n = (2 * R + 1) ** 2
    costs = np.empty((n,) + a.shape, np.float32)
    offs = []
    for dy in range(-R, R + 1):
        for dx in range(-R, R + 1):
            # The last hour's cloud at (x - dx, y - dy) moved here: compare with the previous image shifted by (dx, dy).
            sh = np.roll(np.roll(b, dy, axis=0), dx, axis=1)
            costs[len(offs)] = ndimage.uniform_filter((a - sh) ** 2, block, mode=wrap)
            offs.append((dx, dy))
    offs = np.array(offs)
    k = np.argmin(costs, axis=0)
    best = np.take_along_axis(costs, k[None], 0)[0]
    bx, by = offs[k, 0].astype(np.float32), offs[k, 1].astype(np.float32)

    # Sub-pixel: a parabola through the best cost and its neighbours each way (when they're inside the search).
    def cost(ox, oy):
        nx, ny = offs[k, 0] + ox, offs[k, 1] + oy
        ok = (np.abs(nx) <= R) & (np.abs(ny) <= R)
        idx = np.clip((ny + R) * (2 * R + 1) + (nx + R), 0, n - 1)
        return np.where(ok, np.take_along_axis(costs, idx[None], 0)[0], np.nan)

    def refine(base, minus, plus):
        den = minus - 2.0 * best + plus
        step = np.where(np.isfinite(den) & (den > 1e-6), 0.5 * (minus - plus) / np.where(den > 1e-6, den, 1.0), 0.0)
        return base + np.clip(step, -0.5, 0.5)

    fx = refine(bx, cost(-1, 0), cost(1, 0))
    fy = refine(by, cost(0, -1), cost(0, 1))
    # A clear winner: the best is well below the typical cost at this pixel.
    typical = np.median(costs[:: max(1, n // 25)], axis=0)
    del costs
    distinct = np.clip(1.0 - best / np.maximum(typical, 1e-6), 0.0, 1.0)
    texture = ndimage.uniform_filter(a * a, block, mode=wrap)
    inside = (np.abs(bx) < R) & (np.abs(by) < R)
    wgt = smoothstep(0.2, 0.6, cloud) * smoothstep(4.0, 40.0, texture) * smoothstep(0.2, 0.6, distinct) * inside
    out_lat = 90.0 - (np.arange(h2) + 0.5) * 180.0 / h2
    km_x = 2.0 * np.pi * 6371.0 / w2 * np.cos(np.radians(out_lat))[:, None]
    km_y = np.pi * 6371.0 / h2
    u = fx * km_x   # km/h east (one hour between the images)
    v = -fy * km_y  # rows run south, so a positive row shift is southward
    # A few bad matches survive; the median of the neighbourhood throws them out before smoothing.
    u = ndimage.median_filter(u, 5, mode="wrap")
    v = ndimage.median_filter(v, 5, mode="wrap")

    # Spread into the gaps: normalised smoothing at about 150 km, and a much wider pass for big clear areas.
    def spread(f, sigma):
        num = ndimage.gaussian_filter(f * wgt, sigma, mode=wrap)
        den = ndimage.gaussian_filter(wgt, sigma, mode=wrap)
        return num / np.maximum(den, 1e-6), den

    nu, d1 = spread(u, 7)
    nv, _ = spread(v, 7)
    wu, d2 = spread(u, 40)
    wv, _ = spread(v, 40)
    near = smoothstep(0.02, 0.15, d1)
    far = smoothstep(0.0005, 0.01, d2)
    u = (nu * near + wu * (1 - near)) * far
    v = (nv * near + wv * (1 - near)) * far
    # Beyond the satellites' view nothing is known: still.
    edge = smoothstep(72.0, 62.0, np.abs(out_lat))[:, None]
    u, v = u * edge, v * edge

    def shrink(f):
        return f.reshape(FLOW_H, h2 // FLOW_H, FLOW_W, w2 // FLOW_W).mean(axis=(1, 3))

    u, v = shrink(u), shrink(v)
    speed = np.hypot(u, v)
    mid = np.abs(90.0 - (np.arange(FLOW_H) + 0.5) * 180.0 / FLOW_H) < 60
    print(f"cloud motion: median {np.median(speed[mid]):.0f} km/h, 95th percentile {np.percentile(speed[mid], 95):.0f} km/h")
    return u, v


def save_flow(u, v, path):
    px = np.zeros((FLOW_H, FLOW_W, 3), np.uint8)
    px[..., 0] = (np.clip(u / (2.0 * FLOW_KMH) + 0.5, 0, 1) * 255 + 0.5).astype(np.uint8)
    px[..., 1] = (np.clip(v / (2.0 * FLOW_KMH) + 0.5, 0, 1) * 255 + 0.5).astype(np.uint8)
    px[..., 2] = 128   # the same sRGB marker as the cloud images
    Image.fromarray(px).save(path, optimize=True)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("now", nargs="?")
    p.add_argument("--fetch", action="store_true", help="download the inputs from NOAA")
    p.add_argument("--past", nargs="*", default=[])
    p.add_argument("--vis", help="visible mosaic for the same hour")
    p.add_argument("--vis-past", nargs="*", default=[])
    p.add_argument("--vis-blend", action="store_true",
                   help="experimental: with --fetch, also download the visible mosaics and blend them in by day")
    p.add_argument("--time", help="UTC hour of the data as YYYY-MM-DDTHH, for the Sun's position (default: now)")
    p.add_argument("--prev", help="infrared mosaic for the hour before, for cloud motion (flow.png)")
    p.add_argument("--out", default=".")
    p.add_argument("--preview", action="store_true", help="also write full.png, the whole 4096 x 2048 map")
    a = p.parse_args()
    tmp = tempfile.TemporaryDirectory()
    hour = None
    if a.fetch:
        a.now, a.past, a.vis, a.vis_past, hour, a.prev = fetch(tmp.name, a.vis_blend)
    now, lat, lon = load(a.now)
    past = [load(f)[0] for f in a.past]
    op, top = clouds(now, past)
    if a.vis:
        t = hour or (datetime.datetime.strptime(a.time, "%Y-%m-%dT%H").replace(tzinfo=datetime.timezone.utc) if a.time
                     else datetime.datetime.now(datetime.timezone.utc))
        cosz = sun_cos(lat, ((lon + 180.0) % 360.0) - 180.0, t)
        op_vis, valid = clouds_vis(load(a.vis)[0], [load(f)[0] for f in a.vis_past], cosz)
        op, top = blend(op, top, op_vis, valid, cosz)
    op, top = deseam(op, top)
    flow = None
    if a.prev:
        try:
            flow = cloud_flow(now, load(a.prev)[0], op, lat, lon)
        except Exception as e:   # motion is a nicety: never lose the clouds over it
            print(f"cloud motion skipped: {e}")
    # Cover is nearly all-or-nothing from one source pixel to the next, so an edge falls inside a single pixel and the
    # sky shader's threshold traces the pixel staircase (right angles). A slight blur spreads each edge over a few
    # pixels, so the shader's noise shapes it instead of the grid.
    op = ndimage.gaussian_filter(op, 1.3)
    img = to_equirect(op, lat, lon)
    # Heights come in whole counts (about 100 m), which show as contour rings where lighting changes fast with height
    # (at sunset). Smooth them, and dither both channels so 8 bits don't band.
    top = ndimage.gaussian_filter(top, 1.5)
    height = to_equirect(top / TOP_KM, lat, lon)
    rng = np.random.default_rng(0)
    img = img + (rng.random(img.shape) - 0.5) / 255.0
    height = height + (rng.random(height.shape) - 0.5) / 255.0
    px = np.zeros((H, W, 3), np.uint8)
    px[..., 0] = (np.clip(img, 0, 1) * 255 + 0.5).astype(np.uint8)
    px[..., 1] = (np.clip(height, 0, 1) * 255 + 0.5).astype(np.uint8)
    # Blue is a constant 0.5 so the sky shader can tell whether VRChat decoded the image as sRGB (it then reads ~0.21)
    # and undo it.
    px[..., 2] = 128
    os.makedirs(a.out, exist_ok=True)
    Image.fromarray(px[:, : W // 2]).save(os.path.join(a.out, "west.png"), optimize=True)
    Image.fromarray(px[:, W // 2 :]).save(os.path.join(a.out, "east.png"), optimize=True)
    if flow is not None:
        save_flow(*flow, os.path.join(a.out, "flow.png"))
    if a.preview:
        Image.fromarray(px).save(os.path.join(a.out, "full.png"))
    if hour is not None:
        with open(os.path.join(a.out, "time.txt"), "w") as f:
            f.write(f"{hour:%Y-%m-%dT%H}:00Z\n")
    tmp.cleanup()
    print(f"cloud cover {img[np.abs(90.0 - (np.arange(H) + 0.5) * 180.0 / H) < 60].mean():.2f} within 60 degrees")


if __name__ == "__main__":
    main()
