"""PathSnap evaluator UI (runs on Laptop 2).
    pip install -r ui/requirements_ui.txt
    streamlit run ui/app.py
The UI holds no data and no model and never receives ground truth: it asks the API (Laptop 1) for the trip list
(the evaluator_test trips live on the API side), shows the predicted route and the metrics the API computed."""
import json
import math
import os
import time

import branca.colormap as bcm
import folium
import pandas as pd
import requests
import streamlit as st
from streamlit_folium import st_folium

TRIPS_PATH = os.environ.get("PATHSNAP_TRIPS_PATH", "/trips")        # change here if the API uses other route names
PREDICT_PATH = os.environ.get("PATHSNAP_PREDICT_PATH", "/predict")
RATIO_WORDS = ("top1", "accuracy", "f1", "continuity", "precision", "recall", "rate")

st.set_page_config(page_title="PathSnap", page_icon="🛰️", layout="wide")
st.markdown("""
<style>
.block-container {padding-top: 1.4rem; max-width: 1300px;}
.hero {background: linear-gradient(120deg,#0f2027 0%,#203a43 45%,#2c5364 100%); padding: 1.3rem 1.7rem;
       border-radius: 16px; margin-bottom: 1.1rem; border: 1px solid rgba(255,255,255,.08);}
.hero h1 {margin: 0; font-size: 2rem; letter-spacing: .02em; color: #fff;}
.hero p {margin: .25rem 0 .7rem; color: rgba(255,255,255,.75);}
.badge {display:inline-block; padding:.18rem .7rem; border-radius:99px; font-size:.78rem; font-weight:600; margin-right:.4rem;}
.ok {background:rgba(34,197,94,.18); color:#4ade80;} .bad {background:rgba(239,68,68,.18); color:#f87171;}
.info {background:rgba(148,163,184,.18); color:#cbd5e1;}
.kpi {background: rgba(148,163,184,.08); border-left: 5px solid var(--c); padding: .75rem 1rem; border-radius: 12px; margin-bottom:.7rem;}
.kpi .k {font-size:.72rem; opacity:.7; text-transform:uppercase; letter-spacing:.06em;}
.kpi .v {font-size:1.7rem; font-weight:700; line-height:1.25;}
</style>""", unsafe_allow_html=True)


def accent(key):
    k = key.lower()
    if any(w in k for w in ("top1", "accuracy", "f1")):
        return "#22c55e"
    if "continuity" in k:
        return "#3b82f6"
    if "ms" in k.split("_") or "time" in k or "round" in k or "second" in k:
        return "#f59e0b"
    return "#94a3b8"


def fmt(key, v):
    if isinstance(v, bool) or not isinstance(v, (int, float)):
        return str(v)
    if isinstance(v, float) and 0 <= v <= 1 and any(w in key.lower() for w in RATIO_WORDS):
        return f"{v:.2%}"
    return f"{v:,.2f}" if isinstance(v, float) else f"{v:,}"


def kpi_row(metrics):
    items = list(metrics.items())
    for i in range(0, len(items), 4):
        for col, (k, v) in zip(st.columns(4), items[i:i + 4]):
            col.markdown(f'<div class="kpi" style="--c:{accent(k)}"><div class="k">{k.replace("_", " ")}</div>'
                         f'<div class="v">{fmt(k, v)}</div></div>', unsafe_allow_html=True)


def api_get(base, path):
    r = requests.get(f"{base}{path}", timeout=15)
    r.raise_for_status()
    return r.json()


def call_predict(base, trip_id):
    t0 = time.time()
    r = requests.post(f"{base}{PREDICT_PATH}", json={"trip_id": trip_id}, timeout=300)
    r.raise_for_status()
    res = r.json()
    res["_roundtrip_s"] = round(time.time() - t0, 2)
    return res


def build_map(res):
    gps, segs = res.get("gps", []), res.get("predicted_route", [])
    pts = [(p["lat"], p["lon"]) for p in gps] or [pt for s in segs for pt in s]
    m = folium.Map(location=pts[0] if pts else [0, 0], zoom_start=14, tiles=None, control_scale=True)
    folium.TileLayer("Esri WorldStreetMap", name="Light").add_to(m)
    folium.TileLayer("Esri WorldGrayCanvas", name="Dark", show=False).add_to(m)
    folium.TileLayer("https://server.arcgisonline.com/ArcGIS/rest/services/World_Imagery/MapServer/tile/{z}/{y}/{x}",
                     attr="Esri World Imagery", name="Satellite", show=False).add_to(m)
    if pts:
        m.fit_bounds([[min(p[0] for p in pts), min(p[1] for p in pts)], [max(p[0] for p in pts), max(p[1] for p in pts)]])

    g_gps = folium.FeatureGroup("Raw GPS (coloured by time)", show=True)
    if gps:
        cmap = bcm.LinearColormap(["#2dd4bf", "#3b82f6", "#a855f7"], vmin=0, vmax=max(len(gps) - 1, 1),
                                  caption="GPS point order (start to end)")
        step = max(1, math.ceil(len(gps) / 1500))                        # keep the page light on long trips
        for i in range(0, len(gps), step):
            p = gps[i]
            folium.CircleMarker([p["lat"], p["lon"]], radius=3.5, color=cmap(i), weight=0, fill=True, fill_color=cmap(i),
                                fill_opacity=0.9, tooltip=f"obs {p.get('obs_idx', i)}").add_to(g_gps)
        cmap.add_to(m)
    g_gps.add_to(m)

    g_pred = folium.FeatureGroup("Predicted route", show=True)
    if segs:
        folium.PolyLine(segs, color="#ff3b30", weight=11, opacity=0.22).add_to(g_pred)    # glow
        folium.PolyLine(segs, color="#ff3b30", weight=4, opacity=1.0).add_to(g_pred)
    g_pred.add_to(m)
    if gps:
        for p, name, col in ((gps[0], "Start", "#16a34a"), (gps[-1], "End", "#111827")):
            folium.CircleMarker([p["lat"], p["lon"]], radius=9, color="white", weight=3, fill=True, fill_color=col,
                                fill_opacity=1, tooltip=name).add_to(m)
    folium.LayerControl(collapsed=False).add_to(m)
    return m


def route_geojson(res):
    coords = [[[p[1], p[0]] for p in seg] for seg in res.get("predicted_route", [])]
    return json.dumps({"type": "Feature", "properties": {"trip_id": res.get("trip_id")},
                       "geometry": {"type": "MultiLineString", "coordinates": coords}})


# ---------------- connection ----------------
st.sidebar.title("PathSnap")
api = st.sidebar.text_input("API address (Laptop 1)", os.environ.get("PATHSNAP_API", "http://localhost:8000")).rstrip("/")
if st.sidebar.button("Connect / refresh trips") or "trips" not in st.session_state:
    try:
        st.session_state["trips"] = api_get(api, TRIPS_PATH)["trips"]
        st.session_state["conn_error"] = None
    except Exception as e:
        st.session_state["trips"], st.session_state["conn_error"] = [], str(e)
err, trips = st.session_state.get("conn_error"), st.session_state["trips"]
if err:
    st.sidebar.error(f"Cannot reach the API at {api}{TRIPS_PATH}\n\n{err}")
st.sidebar.caption("Trips are read by the API from data/evaluator_test on Laptop 1.")

badge = (f'<span class="badge bad">API offline</span>' if err else
         f'<span class="badge ok">API online</span><span class="badge info">{len(trips)} evaluation trips</span>')
st.markdown(f'<div class="hero"><h1>PathSnap</h1><p>Distributed map-matching and route reconstruction '
            f'from noisy GPS</p>{badge}<span class="badge info">{api}</span></div>', unsafe_allow_html=True)
if not trips:
    st.info("No trips available. Check the API address in the sidebar and that the API is running on Laptop 1.")
    st.stop()

# ---------------- controls ----------------
c1, c2, c3 = st.columns([3, 1.2, 1.2])
trip_id = c1.selectbox("Evaluation trip", trips)
if c2.button("Run PathSnap", type="primary", use_container_width=True):
    try:
        with st.spinner("API is processing the trip..."):
            st.session_state["result"] = call_predict(api, trip_id)
    except Exception as e:
        st.session_state.pop("result", None)
        st.error(f"Request failed: {e}")
if c3.button("Run all trips", use_container_width=True):
    rows, bar = [], st.progress(0.0, text="Running all trips...")
    for i, t in enumerate(trips):
        try:
            r = call_predict(api, t)
            rows.append({"trip_id": t, **r.get("metrics", {}), "round_trip_s": r["_roundtrip_s"]})
        except Exception as e:
            rows.append({"trip_id": t, "error": str(e)[:80]})
        bar.progress((i + 1) / len(trips), text=f"{i + 1}/{len(trips)} trips")
    bar.empty()
    st.session_state["batch"] = pd.DataFrame(rows)

res, batch = st.session_state.get("result"), st.session_state.get("batch")
if res:
    st.subheader(f"Trip {res.get('trip_id', trip_id)}")
    kpi_row({**res.get("metrics", {}), "UI round trip (s)": res["_roundtrip_s"]})

tab_map, tab_acc, tab_det = st.tabs(["Map", "Accuracy", "Details"])
with tab_map:
    if res:
        st_folium(build_map(res), height=620, use_container_width=True, returned_objects=[])
    else:
        st.info("Pick a trip and press **Run PathSnap** to see the reconstructed route.")
with tab_acc:
    if batch is not None:
        st.markdown("**All evaluation trips**")
        num = batch.select_dtypes("number")
        st.dataframe(pd.concat([batch, pd.DataFrame([{"trip_id": "MEAN", **num.mean().to_dict()}])], ignore_index=True),
                     use_container_width=True, hide_index=True)
        top = [c for c in num.columns if "top1" in c.lower()]
        if top:
            st.bar_chart(batch.set_index("trip_id")[top] * 100, y_label="top-1 accuracy (%)")
    elif res:
        top = {k: v * 100 for k, v in res.get("metrics", {}).items() if "top1" in k.lower() and isinstance(v, (int, float))}
        if top:
            st.bar_chart(pd.Series(top, name="top-1 accuracy (%)"))
        else:
            st.info("The API returned no top-1 metrics for this trip.")
    else:
        st.info("Run a trip, or use **Run all trips** for a summary table.")
with tab_det:
    if res:
        d1, d2 = st.columns(2)
        d1.download_button("Download predicted route (GeoJSON)", route_geojson(res), f"{res.get('trip_id', 'trip')}_route.geojson",
                           use_container_width=True)
        d2.download_button("Download GPS points (CSV)", pd.DataFrame(res.get("gps", [])).to_csv(index=False),
                           f"{res.get('trip_id', 'trip')}_gps.csv", use_container_width=True)
        st.json({k: v for k, v in res.items() if k not in ("gps", "predicted_route")})
    else:
        st.info("Nothing to show yet.")
