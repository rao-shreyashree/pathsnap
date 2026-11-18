"""Pure helper functions (no I/O) so they can be unit-tested."""
from itertools import groupby

import numpy as np
import pandas as pd


def as_array(x):
    """DataFrame / ndarray / list -> ndarray."""
    return x.to_numpy() if hasattr(x, "to_numpy") else np.asarray(x)


def arcs_to_geojson(arc_ids, nodes, arcs):
    """Ordered arc ids -> GeoJSON LineString, or MultiLineString if the route has gaps.

    nodes: row i = (longitude, latitude).  arcs: row i = (from_node, to_node).
    Consecutive duplicates are dropped. A new line starts whenever an arc does not
    begin where the previous one ended, so the UI never draws a line across a gap.
    """
    nodes, arcs = as_array(nodes), as_array(arcs)
    seq = [int(k) for k, _ in groupby(arc_ids)]
    lines, cur, last_to = [], [], None
    for a in seq:
        u, v = int(arcs[a][0]), int(arcs[a][1])
        pu = [float(nodes[u][0]), float(nodes[u][1])]
        pv = [float(nodes[v][0]), float(nodes[v][1])]
        if cur and last_to == u:
            cur.append(pv)
        else:
            if cur:
                lines.append(cur)
            cur = [pu, pv]
        last_to = v
    if cur:
        lines.append(cur)
    if not lines:
        return {"type": "LineString", "coordinates": []}
    if len(lines) == 1:
        return {"type": "LineString", "coordinates": lines[0]}
    return {"type": "MultiLineString", "coordinates": lines}


def pointwise_score(pred_by_obs, ref_by_obs):
    """(correct, scored). Only observations that have a valid reference arc are scored;
    an observation with no prediction counts as wrong."""
    scored = [k for k, v in ref_by_obs.items() if pd.notna(v) and v >= 0]
    correct = sum(1 for k in scored if pred_by_obs.get(k) == int(ref_by_obs[k]))
    return correct, len(scored)


def continuity(pred_by_obs, arcs):
    """Share of consecutive predictions that are the same arc or connect end-to-start."""
    arcs = as_array(arcs)
    seq = [pred_by_obs[k] for k in sorted(pred_by_obs)]
    if len(seq) < 2:
        return None
    ok = sum(a == b or arcs[a][1] == arcs[b][0] for a, b in zip(seq, seq[1:]))
    return ok / (len(seq) - 1)
