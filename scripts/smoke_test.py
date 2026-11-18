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
    assert out["predicted_route"]["type"] in ("LineString", "MultiLineString")
    m = out["metrics"]
    assert 0 <= m["pointwise_accuracy"] <= 1 and m["correct_observations"] <= m["scored_observations"]
    assert "reference" not in json.dumps(out).lower() and "label" not in json.dumps(out).lower()
    print(t, m, out["details"]["method"], out["details"]["elapsed_seconds"], "s")
print("ALL OK")
