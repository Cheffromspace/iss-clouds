"""Build events.json, the observatory world's "Now" list, from free public feeds and our own cloud maps.

Meant to run in the hourly iss-clouds job after live_clouds.py and be published next to west.png/east.png.

    python tools/events_feed.py --out site                    # cloud maps downloaded from github.io
    python tools/events_feed.py --out site --clouds-dir site  # cloud maps just written by live_clouds.py

Sources (all free, no key):
  GDACS tropical cyclones (EU JRC / UN), every basin
  NHC CurrentStorms.json, detail for Atlantic and east/central Pacific storms
  NASA EONET v3: wildfires, volcanoes, sea ice (icebergs), dust/haze, and storms GDACS doesn't list yet
  USGS M4.5+ past day: M6+ or significant only
  NOAA SWPC planetary Kp and OVATION nowcast: the "aurora" block the sky draws every hour, and an event when Kp >= 5
  Our own cold-top finder on the published cloud maps (red = opacity, green = cloud-top height 0-16 km)
  With fire or volcano kinds (fire_volcano.py, kept next to this file): NASA FIRMS VIIRS fires (24 h), AWC volcanic
  ash SIGMETs, Smithsonian GVP summits. See hot_events() for the fields they add.

Any source that fails is skipped with a note on stderr; the file is always written. The output is kept small and
flat for Udon's VRCJson parser: one list of flat objects, plain numbers, ASCII strings, under 16 KB.

Uses only the standard library plus numpy and Pillow.
"""
import argparse
import collections
import datetime
import io
import json
import math
import os
import re
import sys
import unicodedata
import urllib.request

import numpy as np
from PIL import Image

UA = "iss-clouds-events/1.0"
NOW = datetime.datetime.now(datetime.timezone.utc)
MAX_EVENTS = 20
MAX_BYTES = 16 * 1024

GDACS_URL = "https://www.gdacs.org/gdacsapi/api/events/geteventlist/SEARCH?eventlist=TC"
NHC_URL = "https://www.nhc.noaa.gov/CurrentStorms.json"
# Two narrower queries instead of the whole open list (about 9 MB, mostly years-old US fire reports).
EONET_RECENT_URL = ("https://eonet.gsfc.nasa.gov/api/v3/events/geojson?status=open&days=21"
                    "&category=wildfires,severeStorms,dustHaze")
EONET_STANDING_URL = "https://eonet.gsfc.nasa.gov/api/v3/events/geojson?status=open&category=volcanoes,seaLakeIce"
USGS_URL = "https://earthquake.usgs.gov/earthquakes/feed/v1.0/summary/4.5_day.geojson"
KP_URL = "https://services.swpc.noaa.gov/products/noaa-planetary-k-index.json"
OVATION_URL = "https://services.swpc.noaa.gov/json/ovation_aurora_latest.json"
CLOUDS_URL = "https://cheffromspace.github.io/iss-clouds/"

CREDITS = {
    "GDACS": "GDACS (EU JRC / UN)",
    "NHC": "NOAA National Hurricane Center",
    "EONET": "NASA EONET",
    "USGS": "USGS Earthquake Hazards Program",
    "SWPC": "NOAA Space Weather Prediction Center",
    "GMGSI": "NOAA GMGSI infrared mosaic (cold-top finder)",
    "FIRMS": "NASA FIRMS (VIIRS 375 m)",
    "AWC": "NOAA Aviation Weather Center / VAACs",
    "GVP": "Smithsonian GVP, Volcanoes of the World",
}

# Viewing altitudes, matching the console's presets (10, 35, 100, 400, 1000, 5000, 36000 km).
ALT = {"cyclone": 1000, "storm": 400, "fire": 35, "volcano": 35, "ice": 100, "quake": 400, "aurora": 400,
       "dust": 1000}

# Kinds the observatory can show today: storms appear in its live cloud maps, and the sky draws the aurora oval
# from the "aurora" block below. Fires, smoke, dust, volcanoes, sea ice and quakes have no visuals yet (owner,
# 2026-11-05: list only what the world can show), so they're left out unless asked for with --kinds.
SHOWN_KINDS = ("cyclone", "storm", "aurora")
ALL_KINDS = tuple(ALT)

STANDING = [
    {"id": "standing-sahara", "kind": "dust", "name": "Sahara dust belt",
     "line": "Dust blown west over the Atlantic", "lat": 18.0, "lon": -20.0, "alt_km": 1000},
    {"id": "standing-itcz", "kind": "storm", "name": "ITCZ over the Pacific",
     "line": "The equator's band of thunderstorms", "lat": 7.0, "lon": -140.0, "alt_km": 5000},
    {"id": "standing-antarctic-ice", "kind": "ice", "name": "Antarctic sea ice edge",
     "line": "Pack ice meets the Southern Ocean", "lat": -62.0, "lon": -30.0, "alt_km": 1000},
]


def log(msg):
    print(msg, file=sys.stderr)


def get(url, timeout=60):
    req = urllib.request.Request(url, headers={"User-Agent": UA, "Accept": "*/*"})
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return r.read()


def get_json(url):
    return json.loads(get(url).decode("utf-8", "replace"))


def ascii_text(s, limit=60):
    """Plain ASCII: accents folded, anything else dropped, whitespace collapsed."""
    s = unicodedata.normalize("NFKD", str(s or "")).encode("ascii", "ignore").decode()
    s = re.sub(r"[\x00-\x1f\"\\]", " ", s)
    s = re.sub(r"\s+", " ", s).strip()
    return s if len(s) <= limit else s[: limit - 3].rstrip() + "..."


def parse_time(s):
    """ISO string (with or without zone, assumed UTC), or epoch milliseconds. None if unreadable."""
    if s is None:
        return None
    if isinstance(s, (int, float)):
        return datetime.datetime.fromtimestamp(s / 1000.0, datetime.timezone.utc)
    s = str(s).strip().replace(" ", "T")
    try:
        t = datetime.datetime.fromisoformat(s.replace("Z", "+00:00"))
    except ValueError:
        return None
    return t if t.tzinfo else t.replace(tzinfo=datetime.timezone.utc)


def iso(t):
    return t.astimezone(datetime.timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ") if t else ""


def hours_ago(t):
    return (NOW - t).total_seconds() / 3600.0 if t else 1e9


def km_between(lat1, lon1, lat2, lon2):
    p1, p2 = math.radians(lat1), math.radians(lat2)
    a = math.sin((p2 - p1) / 2) ** 2 + math.cos(p1) * math.cos(p2) * math.sin(math.radians(lon2 - lon1) / 2) ** 2
    return 2 * 6371.0 * math.asin(min(1.0, math.sqrt(a)))


def compass(deg):
    return ["N", "NNE", "NE", "ENE", "E", "ESE", "SE", "SSE", "S", "SSW", "SW", "WSW", "W", "WNW", "NW",
            "NNW"][int((deg % 360) / 22.5 + 0.5) % 16]


def bearing(lat1, lon1, lat2, lon2):
    p1, p2, dl = math.radians(lat1), math.radians(lat2), math.radians(lon2 - lon1)
    y = math.sin(dl) * math.cos(p2)
    x = math.cos(p1) * math.sin(p2) - math.sin(p1) * math.cos(p2) * math.cos(dl)
    return math.degrees(math.atan2(y, x)) % 360


def num(v):
    """A float from a number or a string like "75" or "24.6N"; None if there isn't one."""
    if v is None:
        return None
    if isinstance(v, (int, float)):
        return float(v)
    m = re.search(r"-?\d+(\.\d+)?", str(v))
    return float(m.group()) if m else None


def latlon_text(lat, lon):
    return f"{abs(lat):.0f}{'N' if lat >= 0 else 'S'} {abs(lon):.0f}{'E' if lon >= 0 else 'W'}"


def event(kind, eid, name, line, lat, lon, start, updated, source, score):
    return {"id": ascii_text(eid, 40), "kind": kind, "name": ascii_text(name, 40), "line": ascii_text(line, 60),
            "lat": round(float(lat), 2), "lon": round(((float(lon) + 180.0) % 360.0) - 180.0, 2),
            "alt_km": ALT[kind], "start": iso(start), "updated": iso(updated), "source": source,
            "standing": False, "_score": score}


# ---------------------------------------------------------------------------------------------- cyclones

STORM_WORDS = r"\b(SUPER|POST|POTENTIAL|TROPICAL|SUBTROPICAL|EXTRATROPICAL|CYCLONE|TYPHOON|HURRICANE|STORM|" \
              r"DEPRESSION|SEVERE|VERY|INTENSE|CYCLONIC|REMNANTS OF)\b"


def storm_key(name):
    """MURASU from "Tropical Cyclone MURASU-26", "Super Typhoon In-fa" -> INFA."""
    s = ascii_text(name, 80).upper()
    s = re.sub(r"-\d{2}$", "", s)
    s = re.sub(STORM_WORDS, " ", s)
    return re.sub(r"[^A-Z0-9]", "", s)


def storm_name(name):
    """Display name: "Super Typhoon In-fa" -> "In-fa", "Tropical Cyclone MURASU-26" -> "Murasu"."""
    s = re.sub(r"-\d{2}$", "", ascii_text(name, 80))
    s = re.sub(STORM_WORDS, " ", s, flags=re.I)
    s = re.sub(r"\s+", " ", s).strip(" -")
    return s.title() if s.isupper() or s.islower() else s


def category(kt):
    if kt is None:
        return ""
    for lim, label in ((137, "Cat 5"), (113, "Cat 4"), (96, "Cat 3"), (83, "Cat 2"), (64, "Cat 1"),
                       (34, "Tropical storm")):
        if kt >= lim:
            return label
    return "Tropical depression"


def fetch_gdacs():
    out = []
    for f in get_json(GDACS_URL).get("features", []):
        p = f.get("properties") or {}
        g = f.get("geometry") or {}
        if str(p.get("iscurrent", "")).lower() != "true" or g.get("type") != "Point":
            continue
        updated = parse_time(p.get("todate"))
        if hours_ago(updated) > 48:
            continue
        sev = (p.get("severitydata") or {})
        peak = None
        if str(sev.get("severityunit", "")).lower().startswith("km"):
            peak = num(sev.get("severity"))
            peak = peak / 1.852 if peak else None
        name = p.get("eventname") or p.get("name") or "Cyclone"
        out.append({"key": storm_key(name), "name": storm_name(name),
                    "lat": g["coordinates"][1], "lon": g["coordinates"][0], "start": parse_time(p.get("fromdate")),
                    "updated": updated, "alert": str(p.get("alertlevel") or ""), "peak_kt": peak,
                    "id": f"gdacs-tc-{p.get('eventid')}", "sources": ["GDACS"]})
    return out


def fetch_nhc():
    out = []
    for s in get_json(NHC_URL).get("activeStorms") or []:
        lat, lon = num(s.get("latitudeNumeric")), num(s.get("longitudeNumeric"))
        if lat is None or lon is None:
            lat, lon = num(s.get("latitude")), num(s.get("longitude"))
            if lat is None or lon is None:
                continue
            lat = -abs(lat) if "S" in str(s.get("latitude")) else lat
            lon = -abs(lon) if "W" in str(s.get("longitude")) else lon
        mdir, mspd = num(s.get("movementDir")), num(s.get("movementSpeed"))
        out.append({"key": storm_key(s.get("name", "")), "name": storm_name(s.get("name") or ""),
                    "lat": lat, "lon": lon, "kt": num(s.get("intensity")), "cls": str(s.get("classification") or ""),
                    "move": compass(mdir) if mdir is not None else "", "move_mph": mspd,
                    "updated": parse_time(s.get("lastUpdate")), "id": f"nhc-{s.get('id') or storm_key(s.get('name'))}"})
    return out


def eonet_tracks(features):
    """{id: [Point features sorted by date]}; the LineString summaries some events carry are dropped."""
    tracks = collections.defaultdict(list)
    for f in features:
        p, g = f.get("properties") or {}, f.get("geometry") or {}
        if g.get("type") == "Point" and p.get("id") and parse_time(p.get("date")):
            tracks[p["id"]].append(f)
    for v in tracks.values():
        v.sort(key=lambda f: parse_time(f["properties"]["date"]))
    return tracks


def category_of(f):
    cats = (f.get("properties") or {}).get("categories") or [{}]
    return cats[0].get("id", "")


def eonet_storms(tracks):
    out = []
    for eid, pts in tracks.items():
        if category_of(pts[0]) != "severeStorms":
            continue
        last = pts[-1]["properties"]
        updated = parse_time(last.get("date"))
        if hours_ago(updated) > 36:
            continue
        lon, lat = pts[-1]["geometry"]["coordinates"][:2]
        move = ""
        if len(pts) >= 2:
            lon0, lat0 = pts[-2]["geometry"]["coordinates"][:2]
            dt = (updated - parse_time(pts[-2]["properties"]["date"])).total_seconds() / 3600.0
            d = km_between(lat0, lon0, lat, lon)
            if dt > 0 and d > 10:
                move = compass(bearing(lat0, lon0, lat, lon))
        kt = num(last.get("magnitudeValue")) if str(last.get("magnitudeUnit", "")).startswith("kt") else None
        out.append({"key": storm_key(last.get("title", "")), "name": storm_name(last.get("title", "")), "lat": lat, "lon": lon,
                    "kt": kt, "move": move, "start": parse_time(pts[0]["properties"]["date"]), "updated": updated,
                    "id": eid})
    return out


def cyclones(gdacs, nhc, eon):
    """Merge the three lists by name, or by position within 300 km. NHC's numbers beat EONET's beat GDACS's peak."""
    merged = []

    def find(c):
        for m in merged:
            if (c["key"] and c["key"] == m["key"]) or km_between(c["lat"], c["lon"], m["lat"], m["lon"]) < 300:
                return m
        return None

    for c in gdacs:
        merged.append(dict(c))
    for c in eon:
        m = find(c)
        if m is None:
            merged.append(dict(c, alert="", peak_kt=None, sources=["EONET"]))
            continue
        if c["updated"] and (m["updated"] is None or c["updated"] >= m["updated"]):
            m.update(lat=c["lat"], lon=c["lon"], updated=c["updated"])
        m.update(kt=c["kt"], move=c["move"] or m.get("move", ""), start=m.get("start") or c["start"])
        m["sources"].append("EONET")
    for c in nhc:
        m = find(c)
        if m is None:
            merged.append(dict(c, alert="", peak_kt=None, start=c["updated"], sources=["NHC"]))
            continue
        m.update(lat=c["lat"], lon=c["lon"], kt=c["kt"] or m.get("kt"), move=c["move"] or m.get("move", ""),
                 move_mph=c.get("move_mph"), cls=c["cls"], updated=c["updated"] or m["updated"], name=c["name"])
        m["sources"].append("NHC")

    out = []
    for m in merged:
        kt = m.get("kt")
        if kt:
            line = f"{category(kt)}, {kt:.0f} kt"
        elif m.get("peak_kt"):
            line = f"{m['alert'] + ' alert, ' if m.get('alert') else ''}peak {m['peak_kt']:.0f} kt"
        else:
            line = "Tropical cyclone"
        if m.get("move"):
            line += f", moving {m['move']}"
            if m.get("move_mph"):
                line += f" {m['move_mph'] * 0.869:.0f} kt"
        age = hours_ago(m["updated"])
        if age > 18:
            line += f", last fix {age:.0f} h ago"
        alert = {"red": 30, "orange": 15}.get(m.get("alert", "").lower(), 0)
        score = 100 + (kt or (m.get("peak_kt") or 40) * 0.6) + alert - max(0.0, age - 12) * 0.8
        word = "Cyclone"
        if m["lat"] > 0 and m["lon"] < -20:
            word = "Hurricane"
        elif m["lat"] > 0 and m["lon"] > 100:
            word = "Typhoon"
        if kt is not None and kt < 64:
            word = "Tropical storm" if kt >= 34 else "Depression"
        name = f"{word} {m['name']}" if m["name"] else word
        src = "/".join(dict.fromkeys(m["sources"]))
        out.append(event("cyclone", m["id"], name, line, m["lat"], m["lon"], m.get("start"), m["updated"], src,
                         score))
    return out


# ---------------------------------------------------------------------------------------------- EONET others

PRESCRIBED = re.compile(r"prescrib|\bRX\b|\bPB\b|\bBB\b|\bCPT\b|pile|burn block|slash", re.I)


def eonet_others(tracks, limit=True):
    fires, volcs, ice, dust = [], [], [], []
    for eid, pts in tracks.items():
        cat = category_of(pts[0])
        p = pts[-1]["properties"]
        lon, lat = pts[-1]["geometry"]["coordinates"][:2]
        start, updated = parse_time(pts[0]["properties"]["date"]), parse_time(p.get("date"))
        title = ascii_text(p.get("title"), 80)
        age_d = hours_ago(updated) / 24.0
        mag, unit = num(p.get("magnitudeValue")), str(p.get("magnitudeUnit") or "")
        if cat == "wildfires":
            if age_d > 14 or PRESCRIBED.search(title):
                continue
            name = re.sub(r"\s*\bWildfire\b\s*", " ", title).strip(" ,")
            name = re.sub(r"^(Incident Complex|Wildfire in)\s+", "", name, flags=re.I) or "Wildfire"
            line = f"Wildfire, {mag:,.0f} acres" if mag and unit == "acres" else "Wildfire"
            score = 30 + (math.log10(mag) * 3 if mag and mag > 1 else 0) - age_d
            fires.append(event("fire", eid, name, line, lat, lon, start, updated, "EONET", score))
        elif cat == "volcanoes":
            if age_d > 365:
                continue
            name = re.sub(r"\s*\bVolcano\b", "", title)
            line = f"Erupting, listed since {start:%b %Y}"
            volcs.append(event("volcano", eid, name, line, lat, lon, start, updated, "EONET", 40 - age_d / 15))
        elif cat == "seaLakeIce":
            if age_d > 45 or not mag:
                continue
            area = mag * 3.43 if unit.startswith("NM") else mag   # square nautical miles to km2
            line = f"Drifting, {area:,.0f} km2"
            ice.append(event("ice", eid, title, line, lat, lon, start, updated, "EONET",
                             15 + math.log10(max(area, 1)) * 3))
        elif cat == "dustHaze":
            if age_d > 5:
                continue
            dust.append(event("dust", eid, title, "Dust and haze", lat, lon, start, updated, "EONET", 35 - age_d))
    fires.sort(key=lambda e: -e["_score"])
    volcs.sort(key=lambda e: -e["_score"])
    ice.sort(key=lambda e: -e["_score"])
    if not limit:                                   # hot_events() checks them all first, then trims
        return fires + volcs + ice[:2] + dust[:3]
    return fires[:10] + volcs[:3] + ice[:2] + dust[:3]


# ---------------------------------------------------------------------------------------------- fire, volcano detail

FIRE_MATCH_KM = 25.0      # an EONET fire's point to the edge of a FIRMS fire cluster
VOLC_MATCH_KM = 20.0      # an EONET volcano's point to an ash SIGMET's vent


def hot_events(others, kinds, ok):
    """Checks fires and volcanoes against what satellites and aviation see now, and adds what the sky needs to draw
    them. Only things really happening stay (owner, 2026-11-05): a fire must have VIIRS detections in the last 24 h,
    a volcano an ash SIGMET in force or a hot vent; when the checking source fails, the EONET entry stays as is.

    Fire fields: frp_mw (fire radiative power summed over the 24 h of VIIRS passes, MW), night_mw (the night passes'
      part, which is what glows), fire_km (radius of the burning area), seen (newest detection, ISO time).
      Big fire clusters EONET doesn't list (over 1,000 MW from 10+ pixels) are added as their own events.
    Volcano fields: summit_km (GVP), glow_mw (night FRP at the vent, 5 MW or more over 24 h; 0 = no glow seen),
      plume_top_km (ash cloud top above sea level), plume_dir (degrees, the way the ash extends from the vent),
      plume_km (vent to the far end of the observed ash), plume_kt (forecast drift speed). -1 = no ash reported.
    """
    import fire_volcano as fv
    fires = [e for e in others if e["kind"] == "fire"]
    volcs = [e for e in others if e["kind"] == "volcano"]
    rest = [e for e in others if e["kind"] not in ("fire", "volcano")]
    det, ok["FIRMS"] = attempt("FIRMS VIIRS 24 h", fv.firms)
    clusters = fv.fire_clusters(det) if ok["FIRMS"] else []
    plumes, gvp = {}, {}
    if "volcano" in kinds:
        plumes, ok["AWC"] = attempt("AWC ash SIGMETs", fv.ash_plumes, NOW)
        gvp, ok["GVP"] = attempt("GVP volcanoes", fv.gvp_volcanoes)

    out_v, used_plumes = [], set()
    for e in volcs:
        plume = None
        for k, p in plumes.items():
            if k not in used_plumes and p["vent"] and km_between(e["lat"], e["lon"], *p["vent"]) < VOLC_MATCH_KM:
                plume = p
                used_plumes.add(k)
                break
        hot = fv.vent_heat(clusters, e["lat"], e["lon"]) if clusters else None
        if not plume and not hot and ok["FIRMS"] and ok.get("AWC"):
            continue                                      # listed, but nothing to see today
        g = next((v for v in gvp.values() if km_between(e["lat"], e["lon"], v[1], v[2]) < 5), None)
        e.update(fv.volcano_fields(plume, hot, g[3] if g else 0.0))
        e["line"] = ascii_text(fv.volcano_line(plume, hot), 60)
        e["source"] = "/".join(["EONET"] + (["AWC"] if plume else []) + (["FIRMS"] if hot else []) +
                               (["GVP"] if g else []))
        e["_score"] = 45 + (plume["top_km"] * 2 if plume else 0) + (math.log10(1 + hot["frp_mw"]) * 3 if hot else 0)
        out_v.append(e)
    for k, p in plumes.items():                           # erupting now but not (yet) in EONET
        if k in used_plumes:
            continue
        g = gvp.get(k)
        lat, lon = (g[1], g[2]) if g else (p["vent"] or (None, None))
        if lat is None:
            continue
        hot = fv.vent_heat(clusters, lat, lon) if clusters else None
        e = event("volcano", f"va-{k.lower()}", g[0] if g else p["name"], fv.volcano_line(p, hot), lat, lon,
                  p["obs"], p["obs"], "/".join(["AWC"] + (["FIRMS"] if hot else []) + (["GVP"] if g else [])),
                  45 + p["top_km"] * 2)
        e.update(fv.volcano_fields(p, hot, g[3] if g else 0.0))
        out_v.append(e)

    def fire_fields(c):
        return {"frp_mw": round(c["frp_mw"]), "night_mw": round(c["night_mw"]), "fire_km": round(c["radius_km"], 1),
                "seen": iso(c["last"])}

    vents = [(e["lat"], e["lon"]) for e in out_v]
    out_f, taken = [], set()
    for e in fires:
        c = fv.nearest_cluster(clusters, e["lat"], e["lon"], FIRE_MATCH_KM) if clusters else None
        if c is None and ok["FIRMS"]:
            continue                                      # no VIIRS detection in 24 h: out, or too small to see
        if c:
            taken.add(id(c))
            e.update(fire_fields(c))
            e["source"] = "EONET/FIRMS"
            e["_score"] += math.log10(1 + c["frp_mw"]) * 3
        out_f.append(e)
    for c in clusters:
        if c["frp_mw"] < fv.MIN_FIRE_MW:
            break
        if id(c) in taken or c["pixels"] < fv.MIN_FIRE_PIXELS:
            continue
        if any(km_between(c["lat"], c["lon"], a, b) < c["radius_km"] + fv.VENT_KM for a, b in vents):
            continue                                      # a volcano's own heat
        line = f"{c['frp_mw']:,.0f} MW of fire in 24 h, {2 * c['radius_km']:.0f} km across"
        e = event("fire", f"firms-{c['lat']:+.1f}{c['lon']:+.1f}", f"Fires {latlon_text(c['lat'], c['lon'])}", line,
                  c["lat"], c["lon"], c["last"], c["last"], "FIRMS", 28 + math.log10(c["frp_mw"]) * 3)
        e.update(fire_fields(c))
        out_f.append(e)
    out_f.sort(key=lambda e: -e["_score"])
    out_v.sort(key=lambda e: -e["_score"])
    log(f"fires: {len(fires)} from EONET, {len(out_f)} kept or added; volcanoes: {len(volcs)} from EONET, "
        f"{len(plumes)} ash SIGMETs, {len(out_v)} kept")
    return out_f[:10] + out_v[:5] + rest


# ---------------------------------------------------------------------------------------------- quakes, aurora

def quakes():
    out = []
    for f in get_json(USGS_URL).get("features", []):
        p, g = f.get("properties") or {}, f.get("geometry") or {}
        mag, sig = num(p.get("mag")) or 0.0, num(p.get("sig")) or 0.0
        alert = str(p.get("alert") or "")
        if not (mag >= 6.0 or sig >= 600 or alert in ("yellow", "orange", "red")):
            continue
        lon, lat = g["coordinates"][:2]
        depth = g["coordinates"][2] if len(g["coordinates"]) > 2 else None
        t = parse_time(p.get("time"))
        line = f"M{mag:.1f}" + (f", {depth:.0f} km deep" if depth is not None else "")
        if p.get("tsunami"):
            line += ", tsunami flag"
        place = re.sub(r"^\d+\s*km\s+[NSEW]{1,3}\s+of\s+", "near ", ascii_text(p.get("place"), 60))
        score = 60 + (mag - 6.0) * 15 + (10 if alert in ("orange", "red") else 0) - hours_ago(t) * 0.3
        out.append(event("quake", f"usgs-{f.get('id')}", f"Earthquake {place}", line, lat, lon, t,
                         parse_time(p.get("updated")) or t, "USGS", score))
    out.sort(key=lambda e: -e["_score"])
    return out[:5]


# Geomagnetic north pole (IGRF-14 dipole, 2025): used to place the aurora oval in geographic coordinates.
DIPOLE_LAT, DIPOLE_LON = 80.8, -72.6


def geomag_lat(lat, lon):
    a, b = math.radians(lat), math.radians(DIPOLE_LAT)
    return math.degrees(math.asin(math.sin(a) * math.sin(b) +
                                  math.cos(a) * math.cos(b) * math.cos(math.radians(lon - DIPOLE_LON))))


def kp_rows():
    data = get_json(KP_URL)
    rows = []
    for r in data:
        if isinstance(r, dict):
            t, kp = r.get("time_tag"), r.get("Kp", r.get("kp", r.get("kp_index")))
        elif isinstance(r, list) and len(r) >= 2:
            t, kp = r[0], r[1]
            if t == "time_tag":
                continue                                       # old format: header row
        else:
            continue
        t, kp = parse_time(t), num(kp)
        if t and kp is not None:
            rows.append((t, kp))
    if not rows:
        raise ValueError("no Kp rows")
    rows.sort()
    return rows


def kp_band(kp):
    """Night-side oval from Kp alone, in degrees geomagnetic latitude: (equatorward edge, poleward edge).
    The same formula as the world's fallback (AuroraFeed.cs), so the two agree."""
    return 66.0 - 2.0 * kp, 73.0 - 0.9 * kp


def ovation_bands():
    """Summarise SWPC's OVATION nowcast (a 1-degree grid of aurora probability, about 920 KB) to what the world
    draws: per hemisphere, the night-side (21-03 magnetic local time) oval's equatorward and poleward edges in
    geomagnetic latitude where the probability is at least 10 %, and its peak probability."""
    data = get_json(OVATION_URL)
    g = np.asarray(data["coordinates"], dtype=float)
    lon, lat, p = np.radians(g[:, 0]), np.radians(g[:, 1]), g[:, 2]
    # Geographic -> dipole frame: z along the geomagnetic north pole, x toward its meridian.
    b, l0 = math.radians(DIPOLE_LAT), math.radians(DIPOLE_LON)
    x, y, z = np.cos(lat) * np.cos(lon - l0), np.cos(lat) * np.sin(lon - l0), np.sin(lat)
    mx, mz = x * math.sin(b) - z * math.cos(b), x * math.cos(b) + z * math.sin(b)
    mlat, mlon = np.degrees(np.arcsin(np.clip(mz, -1, 1))), np.degrees(np.arctan2(y, mx))
    # Magnetic local time from the subsolar point's magnetic longitude (declination and equation of time ignored:
    # an hour of MLT is 15 degrees, far coarser than either).
    t = parse_time(data.get("Forecast Time")) or parse_time(data.get("Observation Time")) or NOW
    slon = math.radians(180.0 - (t.hour + t.minute / 60.0) * 15.0) - l0
    smlon = math.degrees(math.atan2(math.sin(slon), math.cos(slon) * math.sin(b)))
    mlt = (12.0 + (mlon - smlon) / 15.0) % 24.0
    night = (mlt >= 21.0) | (mlt <= 3.0)
    out = {}
    for hemi, sign in (("n", 1.0), ("s", -1.0)):
        m = night & (sign * mlat > 40.0)
        seen = m & (p >= 10.0)
        if seen.sum() < 5:
            out[hemi] = None                                   # nothing visible on that night side
            continue
        a = sign * mlat[seen]
        out[hemi] = (float(np.percentile(a, 5)), float(np.percentile(a, 95)), float(p[m].max()))
    return out, t


def aurora_state(rows, ov):
    """The world's aurora settings: Kp, the time they're for, the night-side oval per hemisphere and its strength.
    Published every hour whatever Kp is, because a quiet oval is still real (faint, near the poles)."""
    kp_now = rows[-1][1]
    eq, pol = kp_band(kp_now)
    st = {"kp": round(kp_now, 2), "kp_time": iso(rows[-1][0]), "unix": int(NOW.timestamp()),
          "n_eq": round(eq, 1), "n_pol": round(pol, 1), "s_eq": round(eq, 1), "s_pol": round(pol, 1),
          "n_peak": -1, "s_peak": -1, "source": "kp"}
    if ov:
        bands, t = ov
        st["source"], st["unix"] = "ovation", int(t.timestamp())
        for h in ("n", "s"):
            if bands[h] is None:
                st[h + "_peak"] = 0
            else:
                st[h + "_eq"], st[h + "_pol"], st[h + "_peak"] = round(bands[h][0], 1), round(bands[h][1], 1),                     int(round(bands[h][2]))
    return st


def aurora(rows):
    recent = [kp for t, kp in rows if hours_ago(t) <= 6] or [rows[-1][1]]
    kp = max(recent)
    log(f"Kp now {rows[-1][1]:.2f} ({rows[-1][0]:%Y-%m-%d %H}Z), max {kp:.2f} in the last 6 h")
    if kp < 5:
        return []
    # Equatorward edge of the auroral oval on the night side (kp_band: Kp 5 about 56, Kp 9 about 48).
    edge = kp_band(kp)[0]
    # Go to the northern oval at local midnight, a few degrees poleward of the edge.
    lon = ((180.0 - (NOW.hour + NOW.minute / 60.0) * 15.0 + 180.0) % 360.0) - 180.0
    lat = next((la for la in np.arange(30.0, 89.0, 0.25) if geomag_lat(la, lon) >= edge + 4.0), 65.0)
    g = "G" + str(min(5, max(1, int(kp + 1e-6) - 4)))
    line = f"Kp {kp:.1f} ({g} storm), oval to {edge:.0f} deg magnetic"
    return [event("aurora", f"aurora-{rows[-1][0]:%Y%m%d}", "Aurora storm", line, lat, lon,
                  next((t for t, k in rows if k >= 5 and hours_ago(t) <= 24), rows[-1][0]), rows[-1][0], "SWPC",
                  90 + (kp - 5) * 8)]


# ---------------------------------------------------------------------------------------------- cold tops

TOP_KM = 16.0
COLD_KM = 12.0          # cloud tops above this, about -55 C in the tropics
MIN_OPACITY = 0.75
MIN_AREA_KM2 = 30000.0  # a mesoscale convective system's cold shield
BLOCK = 4               # 4096 x 2048 -> 1024 x 512, about 39 km cells at the equator
# Only the tropics and subtropics: poleward of this the tropopause is low and the heights live_clouds.py estimates
# from infrared run high, so big frontal cloud shields in the storm tracks would pass as "thunderstorms".
MAX_LAT = 35.0
CLOUD_TIME = []


def load_clouds(folder):
    imgs = []
    for side in ("west", "east"):
        if folder:
            img = Image.open(os.path.join(folder, f"{side}.png"))
        else:
            img = Image.open(io.BytesIO(get(CLOUDS_URL + f"{side}.png", timeout=120)))
        imgs.append(np.asarray(img.convert("RGB"))[..., :2])
    try:   # the hour the maps show, written by live_clouds.py
        text = open(os.path.join(folder, "time.txt")).read() if folder else get(CLOUDS_URL + "time.txt").decode()
        CLOUD_TIME.append(parse_time(text.strip()))
    except Exception as e:
        log(f"cloud map time unknown ({type(e).__name__})")
    return np.concatenate(imgs, axis=1)


def cold_tops(rg, named):
    h, w = rg.shape[:2]
    op = rg[..., 0].astype(np.float32) / 255.0
    top = rg[..., 1].astype(np.float32) / 255.0 * TOP_KM
    cold = (op >= MIN_OPACITY) & (top >= COLD_KM)
    hb, wb = h // BLOCK, w // BLOCK
    frac = cold[: hb * BLOCK, : wb * BLOCK].reshape(hb, BLOCK, wb, BLOCK).mean(axis=(1, 3))
    tops = top[: hb * BLOCK, : wb * BLOCK].reshape(hb, BLOCK, wb, BLOCK).max(axis=(1, 3))
    mask = frac >= 0.5
    lat_c = 90.0 - (np.arange(hb) + 0.5) * 180.0 / hb
    lon_c = -180.0 + (np.arange(wb) + 0.5) * 360.0 / wb
    mask[np.abs(lat_c) > MAX_LAT] = False
    cell_km2 = (2 * math.pi * 6371.0 / wb) * (math.pi * 6371.0 / hb) * np.cos(np.radians(lat_c))
    seen = np.zeros_like(mask)
    found = []
    for y0, x0 in zip(*np.nonzero(mask)):
        if seen[y0, x0]:
            continue
        seen[y0, x0] = True
        stack, cells = [(y0, x0)], []
        while stack:                               # flood fill, 8-connected, wrapping round in longitude
            y, x = stack.pop()
            cells.append((y, x))
            for dy in (-1, 0, 1):
                for dx in (-1, 0, 1):
                    ny, nx = y + dy, (x + dx) % wb
                    if 0 <= ny < hb and mask[ny, nx] and not seen[ny, nx]:
                        seen[ny, nx] = True
                        stack.append((ny, nx))
        ys = np.array([c[0] for c in cells])
        xs = np.array([c[1] for c in cells])
        wts = cell_km2[ys] * frac[ys, xs]
        area = float(wts.sum())
        if area < MIN_AREA_KM2:
            continue
        lat = float((lat_c[ys] * wts).sum() / wts.sum())
        ang = np.radians(lon_c[xs])                # circular mean, safe across the date line
        lon = math.degrees(math.atan2((np.sin(ang) * wts).sum(), (np.cos(ang) * wts).sum()))
        found.append((area, lat, lon, float(tops[ys, xs].max())))
    found.sort(reverse=True)
    t = CLOUD_TIME[-1] if CLOUD_TIME else None
    out = []
    for area, lat, lon, peak in found:
        if any(km_between(lat, lon, c["lat"], c["lon"]) < 500 for c in named):
            continue
        line = f"Cold tops over {area / 1000:,.0f},000 km2, to {peak:.0f} km"
        out.append(event("storm", f"coldtop-{lat:+.0f}{lon:+.0f}", f"Thunderstorm complex {latlon_text(lat, lon)}",
                         line, lat, lon, t, t, "GMGSI", 50 + min(area / 20000.0, 25)))
        if len(out) >= 4:
            break
    log(f"cold-top finder: {len(found)} regions over {MIN_AREA_KM2:,.0f} km2, {len(out)} kept")
    return out


# ---------------------------------------------------------------------------------------------- main

def attempt(name, fn, *args):
    try:
        r = fn(*args)
        log(f"{name}: ok, {len(r)} item(s)")
        return r, True
    except Exception as e:  # any failure: skip the source, keep the file
        log(f"{name}: FAILED ({type(e).__name__}: {e})")
        return [], False


def build(clouds_dir, kinds=SHOWN_KINDS):
    kinds = set(kinds)
    ok = {}
    gd, ok["GDACS"] = attempt("GDACS", fetch_gdacs)
    nh, ok["NHC"] = attempt("NHC", fetch_nhc)
    recent, ok_r = attempt("EONET recent", lambda: get_json(EONET_RECENT_URL).get("features", []))
    standing, ok_s = [], True
    if kinds & {"volcano", "ice"}:
        standing, ok_s = attempt("EONET volcanoes/ice", lambda: get_json(EONET_STANDING_URL).get("features", []))
    ok["EONET"] = ok_r or ok_s
    tracks = eonet_tracks(recent + standing)
    eon_storms, _ = attempt("EONET storms", eonet_storms, tracks)
    events, _ = attempt("cyclone merge", cyclones, gd, nh, eon_storms)
    hot = bool(kinds & {"fire", "volcano"})
    others, _ = attempt("EONET events", eonet_others, tracks, not hot)
    if hot:
        others = hot_events(others, kinds, ok)
    qk, au = [], []
    if "quake" in kinds:
        qk, ok["USGS"] = attempt("USGS", quakes)
    # Kp and the OVATION oval are published whatever the kinds: the world draws the aurora from them all the time.
    rows, ok["SWPC"] = attempt("SWPC Kp", kp_rows)
    ov, ok_ov = attempt("SWPC OVATION", lambda: [ovation_bands()])
    aurora_doc = aurora_state(rows, ov[0] if ok_ov else None) if rows else None
    if "aurora" in kinds and rows:
        au, _ = attempt("aurora event", aurora, rows)
    rg, ok["GMGSI"] = attempt("cloud maps", lambda: load_clouds(clouds_dir))
    ct = []
    if ok["GMGSI"] and len(rg):
        ct, ok["GMGSI"] = attempt("cold tops", cold_tops, rg, events)
    events = [e for e in events + au + qk + ct + others if e["kind"] in kinds]
    events.sort(key=lambda e: -e["_score"])
    events = events[:MAX_EVENTS]
    used = {s for e in events for s in e["source"].split("/")}
    if aurora_doc:
        used.add("SWPC")
    if len(events) < 3:
        for s in (s for s in STANDING if s["kind"] in kinds):
            events.append(dict(s, start="", updated="", source="", standing=True))
    for e in events:
        e.pop("_score", None)
    doc = {"generated": iso(NOW), "quiet": not any(not e["standing"] for e in events), "events": events,
           "credits": [CREDITS[k] for k in CREDITS if k in used] or [CREDITS["GDACS"]],
           "failed": [k for k, v in ok.items() if not v]}
    if aurora_doc:
        doc["aurora"] = aurora_doc
    # Trim from the least important end until it fits.
    while len(dump(doc)) > MAX_BYTES and doc["events"]:
        doc["events"].pop()
    return doc


def dump(doc):
    return json.dumps(doc, ensure_ascii=True, separators=(",", ":")).encode("ascii")


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--out", required=True, help="folder to write events.json into")
    ap.add_argument("--clouds-dir", help="folder holding west.png and east.png (default: download from github.io)")
    ap.add_argument("--kinds", default=",".join(SHOWN_KINDS),
                    help=f"comma-separated kinds to list (default: what the world can show). All: {','.join(ALL_KINDS)}")
    a = ap.parse_args()
    kinds = [k.strip() for k in a.kinds.split(",") if k.strip()]
    unknown = [k for k in kinds if k not in ALL_KINDS]
    if unknown:
        ap.error(f"unknown kind(s): {', '.join(unknown)}")
    doc = build(a.clouds_dir, kinds)
    os.makedirs(a.out, exist_ok=True)
    data = dump(doc)
    path = os.path.join(a.out, "events.json")
    with open(path, "wb") as f:
        f.write(data)
    log(f"wrote {path}: {len(doc['events'])} events, {len(data)} bytes")


if __name__ == "__main__":
    main()
