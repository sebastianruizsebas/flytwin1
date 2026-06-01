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
from neuprint.utils import connection_table_to_matrix
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
    skeleton_dir.mkdir(parents=True, exist_ok=True)
    selected_ids = body_ids[:limit] if limit is not None else body_ids

    exported = 0
    failed = 0
    for body_id in selected_ids:
        out_path = skeleton_dir / f"{body_id}.swc"
        try:
            heal = heal_distance if heal_distance is not None else False
            client.fetch_skeleton(
                int(body_id),
                heal=heal,
                format="swc",
                export_path=str(out_path),
            )
            exported += 1
        except Exception as exc:
            failed += 1
            print(f"Failed to export skeleton for bodyId={body_id}: {exc}")

    return {
        "requested": len(selected_ids),
        "exported": exported,
        "failed": failed,
    }


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
    matrix_df = connection_table_to_matrix(conn_df, group_cols="bodyId", weight_col="weight", sort_by="bodyId")
    body_ids = matrix_df.index.to_numpy(dtype=np.int64)
    matrix_sparse = sparse.csr_matrix(matrix_df.to_numpy(dtype=np.float32))

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
    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()

    token = _resolve_token(args.token, args.token_env)

    if args.list_datasets:
        list_datasets(server=args.server, token=token)
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