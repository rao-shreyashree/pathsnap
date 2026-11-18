"""Generate review-only candidate tables from the supplied development ZIP."""

import argparse
from contextlib import nullcontext
from pathlib import Path
from zipfile import ZipFile

import numpy as np
import pandas as pd
from scipy.spatial import cKDTree


def load_trip(archive, base, suffix, dtype=float):
    path = f"{base}.{suffix}"
    source_file = archive.open(path) if archive is not None else Path(path).open("rb")
    with source_file as source:
        return np.loadtxt(source, dtype=dtype, ndmin=2)


def segment_geometry(points, starts, vectors, lengths_squared):
    offsets = points - starts
    fraction = np.sum(offsets * vectors, axis=-1) / np.maximum(lengths_squared, 1e-12)
    fraction = np.clip(fraction, 0, 1)
    displacement = offsets - fraction[..., None] * vectors
    return np.linalg.norm(displacement, axis=-1), fraction


def align_reference(points, starts, vectors, lengths_squared, reference):
    """Find minimum total distance to route positions in nondecreasing order."""
    starts, vectors, lengths_squared = starts[reference], vectors[reference], lengths_squared[reference]
    positions = np.arange(len(reference), dtype=np.int32)
    parents = np.empty((len(points), len(reference)), dtype=np.int32)
    costs = segment_geometry(points[0], starts, vectors, lengths_squared)[0]
    for i, point in enumerate(points[1:], start=1):
        prefix_best = np.minimum.accumulate(costs)
        improved = np.r_[True, prefix_best[1:] < prefix_best[:-1]]
        parents[i] = np.maximum.accumulate(np.where(improved, positions, 0))
        costs = prefix_best + segment_geometry(point, starts, vectors, lengths_squared)[0]
    aligned = np.empty(len(points), dtype=np.int32)
    aligned[-1] = np.argmin(costs)
    for i in range(len(points) - 1, 0, -1):
        aligned[i - 1] = parents[i, aligned[i]]
    return reference[aligned]


def make_candidates(archive, base, trip_id):
    track = load_trip(archive, base, "track")
    nodes = load_trip(archive, base, "nodes")
    arcs = load_trip(archive, base, "arcs", np.int64)
    reference = load_trip(archive, base, "route", np.int64).ravel()
    if not np.isfinite(track).all() or not np.isfinite(nodes).all():
        raise ValueError(f"{trip_id}: nonfinite coordinates or timestamps")
    if np.any(arcs < 0) or np.any(arcs >= len(nodes)) or np.any(reference < 0) or np.any(reference >= len(arcs)):
        raise ValueError(f"{trip_id}: invalid node or arc IDs")
    track = track[np.argsort(track[:, 2], kind="stable")]
    origin = np.median(track[:, :2], axis=0)
    scale = 6371008.8 * np.pi / 180 * np.array([np.cos(np.deg2rad(origin[1])), 1])
    points = (track[:, :2] - origin) * scale
    nodes = (nodes - origin) * scale
    starts = nodes[arcs[:, 0]]
    vectors = nodes[arcs[:, 1]] - starts
    lengths_squared = np.sum(vectors ** 2, axis=1)

    # Shortlist via a midpoint index, then rank by point-to-segment distance.
    shortlist_size = min(32, len(arcs))
    _, shortlist = cKDTree(starts + vectors / 2).query(points, k=np.arange(1, shortlist_size + 1))
    distances, fractions = segment_geometry(points[:, None, :], starts[shortlist],
                                            vectors[shortlist], lengths_squared[shortlist])
    top = np.argsort(distances, axis=1, kind="stable")[:, :5]
    candidates = np.take_along_axis(shortlist, top, axis=1)
    distances = np.take_along_axis(distances, top, axis=1)
    fractions = np.take_along_axis(fractions, top, axis=1)

    movement = np.diff(points, axis=0, prepend=points[:1])
    speed = np.linalg.norm(movement, axis=1)
    gaps = np.diff(track[:, 2], prepend=track[0, 2])
    bearings = np.arctan2(movement[:, 1], movement[:, 0])
    arc_bearings = np.arctan2(vectors[candidates, 1], vectors[candidates, 0])
    heading_error = np.abs(np.angle(np.exp(1j * (arc_bearings - bearings[:, None]))))
    heading_error[(speed < 1) | (gaps <= 0) | (gaps > 5)] = np.pi / 2

    # Reference arcs supply labels only. Candidate IDs and features are already fixed.
    labels = align_reference(points, starts, vectors, lengths_squared, reference)
    width = candidates.shape[1]
    return pd.DataFrame({
        "trip_id": trip_id,
        "observation_id": np.repeat(np.arange(len(points)), width),
        "arc_id": candidates.ravel(),
        "reference_arc_id": np.repeat(labels, width),
        "from_node": arcs[candidates, 0].ravel(),
        "to_node": arcs[candidates, 1].ravel(),
        "distance_m": distances.ravel(),
        "heading_error_rad": heading_error.ravel(),
        "segment_fraction": fractions.ravel(),
    })


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("input", type=Path, help="Student ZIP or directory containing train and student_test")
    parser.add_argument("output", type=Path)
    parser.add_argument("--train-limit", type=int)
    parser.add_argument("--test-limit", type=int)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    is_directory = args.input.is_dir()
    source = nullcontext() if is_directory else ZipFile(args.input)
    with source as archive:
        for split, limit in [("train", args.train_limit), ("student_test", args.test_limit)]:
            if is_directory:
                names = sorted(path.as_posix() for path in (args.input / split).glob("*/*.track"))
            else:
                names = sorted(name for name in archive.namelist()
                               if name.endswith(".track") and name.split("/")[-3] == split)
            selected = names[:limit]
            if not selected:
                parser.error(f"No trips selected for {split}")
            tables = []
            for name in selected:
                trip_id = name.split("/")[-2]
                tables.append(make_candidates(archive, name.removesuffix(".track"), trip_id))
            table = pd.concat(tables, ignore_index=True)
            table.to_csv(args.output / f"{split}.csv", index=False)
            print(f"{split}: {len(selected)} trips, {len(table):,} candidates", flush=True)


if __name__ == "__main__":
    main()
