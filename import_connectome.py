"""
Import neuPrint connectome assets into data/connectome/ for the fly twin.

This script persists the full connectivity scaffold plus optional task-interface
subsets and skeleton SWC files. It is intentionally staged so the full graph is
kept as the substrate while morphology export can remain limited to the first
task-relevant populations.
"""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

import numpy as np
import pandas as pd
from neuprint import Client, NeuronCriteria as NC, fetch_adjacencies, fetch_neurons
from scipy import sparse


DEFAULT_SERVER = "neuprint.janelia.org"
DEFAULT_DATASET = "male-cns:v0.9"
DEFAULT_OUT_DIR = Path("data/connectome")


def _resolve_token(explicit_token: str | None, token_env: str) -> str:
    if explicit_token:
        return explicit_token

    for env_name in (token_env, "NEUPRINT_TOKEN", "NEUPRINT_APPLICATION_CREDENTIALS"):
        token = os.environ.get(env_name)
        if token:
            return token

    raise RuntimeError(
        "No neuPrint token found. Pass --token or set one of: "
        f"{token_env}, NEUPRINT_TOKEN, NEUPRINT_APPLICATION_CREDENTIALS"
    )


def _make_client(server: str, dataset: str | None, token: str) -> Client:
    return Client(server, dataset=dataset, token=token)


def list_datasets(server: str, token: str) -> None:
    client = _make_client(server, dataset=None, token=token)
    datasets = client.fetch_datasets(reload_cache=True)

    if not datasets:
        print("No datasets visible for this account.")
        return

    print("Available datasets:")
    for dataset_name in sorted(datasets):
        print(f"  {dataset_name}")


def _full_neuron_criteria() -> NC:
    return NC(status="Traced", cropped=False)


def _export_table(df: pd.DataFrame, out_path: Path) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out_path, index=False)


def _filter_interface_neurons(
    neuron_df: pd.DataFrame,
    type_patterns: list[str],
    instance_patterns: list[str],
) -> pd.DataFrame:
    if neuron_df.empty:
        return neuron_df.copy()

    mask = pd.Series(False, index=neuron_df.index)

    if type_patterns:
        type_series = neuron_df.get("type", pd.Series("", index=neuron_df.index)).fillna("")
        mask = mask | type_series.str.contains("|".join(f"(?:{pattern})" for pattern in type_patterns), regex=True)

    if instance_patterns:
        instance_series = neuron_df.get("instance", pd.Series("", index=neuron_df.index)).fillna("")
        mask = mask | instance_series.str.contains(
            "|".join(f"(?:{pattern})" for pattern in instance_patterns),
            regex=True,
        )

    if not type_patterns and not instance_patterns:
        return neuron_df.iloc[0:0].copy()

    return neuron_df.loc[mask].copy()


def _export_skeletons(
    client: Client,
    body_ids: list[int],
    skeleton_dir: Path,
    heal_distance: float | None,
    limit: int | None,
) -> dict[str, int]:
    import time
    from tqdm import tqdm

    skeleton_dir.mkdir(parents=True, exist_ok=True)
    selected_ids = body_ids[:limit] if limit is not None else body_ids
    total = len(selected_ids)

    print(f"Exporting {total} skeleton(s) to {skeleton_dir} ...")

    exported = 0
    failed = 0
    t0 = time.perf_counter()

    with tqdm(selected_ids, unit="skeleton", dynamic_ncols=True) as pbar:
        for body_id in pbar:
            out_path = skeleton_dir / f"{body_id}.swc"
            t_item = time.perf_counter()
            try:
                heal = heal_distance if heal_distance is not None else False
                client.fetch_skeleton(
                    int(body_id),
                    heal=heal,
                    format="swc",
                    export_path=str(out_path),
                )
                exported += 1
                elapsed_item = time.perf_counter() - t_item
                pbar.set_postfix(
                    ok=exported,
                    fail=failed,
                    last_s=f"{elapsed_item:.1f}s",
                )
            except Exception as exc:
                failed += 1
                tqdm.write(f"  FAILED bodyId={body_id}: {exc}")
                pbar.set_postfix(ok=exported, fail=failed)

    total_s = time.perf_counter() - t0
    avg_s = total_s / max(exported + failed, 1)
    print(
        f"Done: {exported} exported, {failed} failed "
        f"in {total_s:.1f}s  (avg {avg_s:.1f}s/skeleton)"
    )

    return {
        "requested": total,
        "exported": exported,
        "failed": failed,
    }


def run_skeleton_export(
    server: str,
    dataset: str,
    token: str,
    out_dir: Path,
    interface_type_patterns: list[str],
    interface_instance_patterns: list[str],
    skeleton_limit: int | None,
    heal_distance: float | None,
    quiet: bool,
) -> None:
    """
    Export SWC skeletons from already-cached neuron data without re-fetching
    the full adjacency matrix.

    Reads ``neurons.csv.gz`` from *out_dir*, applies the same interface-neuron
    filter used by ``run_import``, then fetches SWC files from neuPrint.

    If no interface patterns are given all traced neurons in the cache are
    eligible (subject to ``--skeleton-limit``).
    """
    neurons_path = out_dir / "neurons.csv.gz"
    if not neurons_path.exists():
        raise FileNotFoundError(
            f"Cached neuron table not found: {neurons_path}\n"
            "Run the full import first (without --skeleton-only)."
        )

    if not quiet:
        print(f"Loading cached neuron table from {neurons_path} ...")
    neuron_df = pd.read_csv(neurons_path)

    if interface_type_patterns or interface_instance_patterns:
        interface_df = _filter_interface_neurons(
            neuron_df,
            type_patterns=interface_type_patterns,
            instance_patterns=interface_instance_patterns,
        )
        if not quiet:
            print(
                f"Interface filter matched {len(interface_df)} / {len(neuron_df)} neurons."
            )
    else:
        # No patterns → use all cached neurons (limit strongly recommended).
        interface_df = neuron_df.copy()
        if not quiet:
            n = len(interface_df) if skeleton_limit is None else min(len(interface_df), skeleton_limit)
            print(
                f"No interface patterns given — exporting up to {n} skeletons "
                f"from all {len(neuron_df)} cached neurons."
            )

    if interface_df.empty:
        print("No neurons matched the interface filter. No skeletons exported.")
        return

    client = _make_client(server, dataset=dataset, token=token)

    if not quiet:
        print("Exporting skeletons ...")
    summary = _export_skeletons(
        client=client,
        body_ids=interface_df["bodyId"].astype(int).tolist(),
        skeleton_dir=out_dir / "skeletons",
        heal_distance=heal_distance,
        limit=skeleton_limit,
    )

    if not quiet:
        print("Skeleton export complete.")
        print(json.dumps(summary, indent=2))


def run_import(
    server: str,
    dataset: str,
    token: str,
    out_dir: Path,
    min_weight: int,
    interface_type_patterns: list[str],
    interface_instance_patterns: list[str],
    export_interface_skeletons: bool,
    skeleton_limit: int | None,
    heal_distance: float | None,
    quiet: bool,
) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    client = _make_client(server, dataset=dataset, token=token)
    criteria = _full_neuron_criteria()

    if not quiet:
        print(f"Fetching neurons from dataset '{dataset}'...")
    neuron_df, roi_counts_df = fetch_neurons(criteria, client=client)

    if not quiet:
        print("Fetching adjacencies for traced, uncropped neurons...")
    _, conn_df = fetch_adjacencies(criteria, criteria, client=client)

    if min_weight > 1:
        conn_df = conn_df.loc[conn_df["weight"] >= min_weight].copy()

    if not quiet:
        print("Building connectivity matrix...")
    # Build a sparse matrix directly from the edge list to avoid the
    # dense pivot that overflows with the full male-CNS neuron count.
    all_ids = np.union1d(
        conn_df["bodyId_pre"].to_numpy(dtype=np.int64),
        conn_df["bodyId_post"].to_numpy(dtype=np.int64),
    )
    # Include neurons that have no edges (they exist in neuron_df but may not
    # appear in conn_df after weight filtering).
    if not neuron_df.empty and "bodyId" in neuron_df.columns:
        all_ids = np.union1d(all_ids, neuron_df["bodyId"].to_numpy(dtype=np.int64))
    body_ids = np.sort(all_ids)
    id_to_idx: dict[int, int] = {int(bid): i for i, bid in enumerate(body_ids)}

    row_indices = conn_df["bodyId_pre"].map(id_to_idx).to_numpy(dtype=np.int32)
    col_indices = conn_df["bodyId_post"].map(id_to_idx).to_numpy(dtype=np.int32)
    weights = conn_df["weight"].to_numpy(dtype=np.float32)
    n = len(body_ids)
    matrix_sparse = sparse.csr_matrix(
        (weights, (row_indices, col_indices)), shape=(n, n), dtype=np.float32
    )

    _export_table(neuron_df, out_dir / "neurons.csv.gz")
    _export_table(roi_counts_df, out_dir / "roi_counts.csv.gz")
    _export_table(conn_df, out_dir / "connections.csv.gz")
    np.save(out_dir / "body_ids.npy", body_ids)
    sparse.save_npz(out_dir / "adjacency_weights_csr.npz", matrix_sparse)

    interface_df = _filter_interface_neurons(
        neuron_df,
        type_patterns=interface_type_patterns,
        instance_patterns=interface_instance_patterns,
    )
    if not interface_df.empty:
        _export_table(interface_df, out_dir / "interface_neurons.csv.gz")

    skeleton_summary = None
    if export_interface_skeletons and not interface_df.empty:
        if not quiet:
            print("Exporting interface skeletons...")
        skeleton_summary = _export_skeletons(
            client=client,
            body_ids=interface_df["bodyId"].astype(int).tolist(),
            skeleton_dir=out_dir / "skeletons",
            heal_distance=heal_distance,
            limit=skeleton_limit,
        )

    summary = {
        "server": server,
        "dataset": dataset,
        "neuron_count": int(len(neuron_df)),
        "roi_count_rows": int(len(roi_counts_df)),
        "connection_count": int(len(conn_df)),
        "matrix_shape": [int(matrix_sparse.shape[0]), int(matrix_sparse.shape[1])],
        "matrix_nonzero": int(matrix_sparse.nnz),
        "min_weight": int(min_weight),
        "interface_type_patterns": interface_type_patterns,
        "interface_instance_patterns": interface_instance_patterns,
        "interface_neuron_count": int(len(interface_df)),
        "skeleton_export": skeleton_summary,
    }

    with (out_dir / "import_summary.json").open("w", encoding="utf-8") as handle:
        json.dump(summary, handle, indent=2)

    if not quiet:
        print("Import complete.")
        print(json.dumps(summary, indent=2))


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Import neuPrint connectome assets into data/connectome")
    parser.add_argument("--server", default=DEFAULT_SERVER, help="neuPrint server hostname")
    parser.add_argument("--dataset", default=DEFAULT_DATASET, help="neuPrint dataset name to import (default: male-cns:v0.9)")
    parser.add_argument("--token", default=None, help="neuPrint token (optional if provided via environment)")
    parser.add_argument("--token-env", default="NEUPRINT_TOKEN", help="Primary environment variable to read the token from")
    parser.add_argument("--out-dir", default=str(DEFAULT_OUT_DIR), help="Output directory for connectome assets")
    parser.add_argument("--list-datasets", action="store_true", help="List datasets visible to your account and exit")
    parser.add_argument("--min-weight", type=int, default=1, help="Minimum synaptic weight to keep in exported connections")
    parser.add_argument(
        "--interface-type",
        action="append",
        default=[],
        help="Regex over neuron type used to mark task-interface populations; repeatable",
    )
    parser.add_argument(
        "--interface-instance",
        action="append",
        default=[],
        help="Regex over neuron instance used to mark task-interface populations; repeatable",
    )
    parser.add_argument(
        "--export-interface-skeletons",
        action="store_true",
        help="Export SWC skeletons for interface neurons only",
    )
    parser.add_argument(
        "--skeleton-limit",
        type=int,
        default=None,
        help="Optional cap on the number of interface skeletons to export",
    )
    parser.add_argument(
        "--heal-distance",
        type=float,
        default=None,
        help="Optional skeleton healing distance passed to neuPrint fetch_skeleton",
    )
    parser.add_argument("--quiet", action="store_true", help="Suppress progress output")
    parser.add_argument(
        "--skeleton-only",
        action="store_true",
        help=(
            "Export SWC skeletons from already-cached neurons.csv.gz without "
            "re-fetching the full adjacency matrix. Useful after a completed "
            "import when you only need (more) skeletons."
        ),
    )
    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()

    token = _resolve_token(args.token, args.token_env)

    if args.list_datasets:
        list_datasets(server=args.server, token=token)
        return

    if args.skeleton_only:
        run_skeleton_export(
            server=args.server,
            dataset=args.dataset,
            token=token,
            out_dir=Path(args.out_dir),
            interface_type_patterns=args.interface_type,
            interface_instance_patterns=args.interface_instance,
            skeleton_limit=args.skeleton_limit,
            heal_distance=args.heal_distance,
            quiet=args.quiet,
        )
        return

    run_import(
        server=args.server,
        dataset=args.dataset,
        token=token,
        out_dir=Path(args.out_dir),
        min_weight=args.min_weight,
        interface_type_patterns=args.interface_type,
        interface_instance_patterns=args.interface_instance,
        export_interface_skeletons=args.export_interface_skeletons,
        skeleton_limit=args.skeleton_limit,
        heal_distance=args.heal_distance,
        quiet=args.quiet,
    )


if __name__ == "__main__":
    main()