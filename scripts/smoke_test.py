"""Quick API check.  python scripts/smoke_test.py [http://127.0.0.1:8000]"""
import json
import sys
import urllib.error
import urllib.request

BASE = sys.argv[1] if len(sys.argv) > 1 else "http://127.0.0.1:8000"


def call(path, body=None):
    req = urllib.request.Request(BASE + path, method="POST" if body is not None else "GET",
                                 data=None if body is None else json.dumps(body).encode(),
                                 headers={"Content-Type": "application/json"})
    try:
        with urllib.request.urlopen(req, timeout=300) as r:
            return r.status, json.load(r)
    except urllib.error.HTTPError as e:
        return e.code, json.load(e)


print("health", call("/health"))
trips = call("/trips")[1]["trips"]
print(len(trips), "trips:", trips)
assert call("/predict", {"trip_id": "no_such_trip"})[0] == 404
assert call("/predict", {"trip_id": ""})[0] == 422
for t in trips:
    code, out = call("/predict", {"trip_id": t})
    assert code == 200 and out["trip_id"] == t, (t, code, out)
    seg = out["predicted_route"][0]
    assert len(seg) == 2 and len(seg[0]) == 2, "route must be [[lat,lon],[lat,lon]] segments"
    assert out["gps"] and {"obs_idx", "lat", "lon", "ts"} <= set(out["gps"][0])
    m = out["metrics"]
    for k in ("n_obs", "top1_classifier_only", "top1_with_hmm", "route_continuity", "inference_ms"):
        assert k in m, k
    text = json.dumps(out).lower()
    assert "reference" not in text and "label" not in text and "ref_arc" not in text
    print(t, m)
print("ALL OK")
