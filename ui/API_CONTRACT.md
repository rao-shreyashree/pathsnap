# API contract (UI on Laptop 2 <-> API on Laptop 1)

Start the API on Laptop 1 so other machines can reach it:
    uvicorn api.main:app --host 0.0.0.0 --port 8000
(uvicorn prints one log line per request: show this to the evaluator.)

## GET /trips
    {"trips": ["<trip_id>", ... 10 evaluation trip ids ...]}

## POST /predict      body: {"trip_id": "<trip_id>"}
    {
      "trip_id": "...",
      "gps":   [{"obs_idx": 0, "lat": 50.05, "lon": 8.58, "ts": 0}, ...],            # raw GPS of the trip
      "predicted_route": [[[lat, lon], [lat, lon]], ...],                              # one [from, to] segment per predicted arc, in order
      "metrics": {"n_obs": 1234, "top1_classifier_only": 0.78, "top1_with_hmm": 0.91,
                  "route_continuity": 0.99, "inference_ms": 420.5}                    # any numeric/str entries, UI shows them all
    }
Errors: 404 for an unknown trip_id, 500 with a message otherwise.

## Rules (from the guide)
- Ground truth stays on the server: do NOT put the reference route or labels in the response.
- Metrics are computed server-side from the API's own reference data.
- `predicted_route` segments can be built with `_arc_coords(trip, arc_ids)` from src/folium_map.py
  (returns [[lat, lon], [lat, lon]] per arc).
