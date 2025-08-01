# SPDX-License-Identifier: BSD-3-Clause
"""Run FastTENET with permutation testing to build a GRN.

This script loads all parameters from a YAML configuration file. In addition to
standard FastTENET options, you can specify the following keys at the root level
of the config file:

PERMUTATIONS: number of permutations to run (default: 100)
SHUFFLE: which data to shuffle for permutations. Either 'trajectory' or
         'expression' (default: 'trajectory').
FDR: FDR threshold used after permutation p‑value computation (default: 0.05).
SPATH_GRN: path to save the resulting GRN (optional)
SPATH_TRIMMED_GRN: path to save trimmed GRN (optional)

Example
-------
python permutation_runner.py --config configs/config_tuck_sub.yml
"""
import argparse
from pathlib import Path
from typing import List

import numpy as np
from omegaconf import OmegaConf
import statsmodels.stats.multitest as smm

import fasttenet as fte


def load_config(path: str):
    conf = OmegaConf.load(path)
    conf.setdefault("PERMUTATIONS", 100)
    conf.setdefault("SHUFFLE", "trajectory")
    conf.setdefault("FDR", 0.05)
    return conf


def build_worker(conf: OmegaConf) -> fte.FastTENET:
    return fte.FastTENET(config=conf)


def run_fasttenet(worker: fte.FastTENET, conf: OmegaConf) -> np.ndarray:
    return worker.run(config=conf)


def permute_data(worker: fte.FastTENET, conf: OmegaConf) -> List[np.ndarray]:
    perm_mats = []
    n_perm = int(conf.PERMUTATIONS)
    shuffle = str(conf.SHUFFLE)

    for _ in range(n_perm):
        if shuffle == "expression":
            exp = worker._exp_data.copy()
            for row in exp:
                np.random.shuffle(row)
            aligned = fte.align_data(exp, worker._trajectory, worker._branch)
        else:  # trajectory
            trj = np.random.permutation(worker._trajectory)
            aligned = fte.align_data(worker._exp_data, trj, worker._branch)

        perm_worker = fte.FastTENET(aligned_data=aligned,
                                    node_name=worker._node_name,
                                    tfs=worker._tf)
        perm_mat = run_fasttenet(perm_worker, conf)
        perm_mats.append(perm_mat)
    return perm_mats


def compute_pvals(original: np.ndarray, perms: List[np.ndarray]) -> np.ndarray:
    perm_stack = np.stack(perms, axis=0)
    greater = (perm_stack >= original[None, :, :]).sum(axis=0)
    pvals = (greater + 1) / (perm_stack.shape[0] + 1)
    return pvals


def filter_matrix_by_pval(te_matrix: np.ndarray, pvals: np.ndarray, conf: OmegaConf) -> np.ndarray:
    pflat = pvals.reshape(-1)
    _, fdr_p, _, _ = smm.multipletests(pflat, alpha=float(conf.FDR), method="fdr_bh")
    fdr_mask = (fdr_p.reshape(pvals.shape) < float(conf.FDR)).astype(te_matrix.dtype)
    return te_matrix * fdr_mask


def save_grn(grn: np.ndarray, path: Path):
    np.savetxt(path, grn, delimiter="\t", fmt="%s")
    print(f"save grn in {path}")


def main():
    parser = argparse.ArgumentParser(description="FastTENET permutation test")
    parser.add_argument("--config", required=True, help="config file path")
    args = parser.parse_args()

    conf = load_config(args.config)
    worker = build_worker(conf)

    print("[Run FastTENET on original data]")
    orig_matrix = run_fasttenet(worker, conf)

    print("[Running permutations]")
    perm_mats = permute_data(worker, conf)

    print("[Computing p-values]")
    pvals = compute_pvals(orig_matrix, perm_mats)

    filtered_mat = filter_matrix_by_pval(orig_matrix, pvals, conf)

    gene_names = worker._node_name
    tf = worker._tf

    weaver = fte.NetWeaver(result_matrix=filtered_mat,
                           gene_names=gene_names,
                           tfs=tf,
                           fdr=1.0,
                           links=0,
                           is_trimming=True,
                           trim_threshold=float(conf.get("TRIM_THRESHOLD", 0.0)),
                           dtype=np.float32)

    grn, trimmed = weaver.run(backend=conf.get("BACKEND", "cpu"),
                              device_ids=conf.get("DEVICE_IDS", 0),
                              batch_size=int(conf.get("BATCH_SIZE", 0)))

    if conf.get("SPATH_GRN"):
        save_grn(grn, Path(conf.SPATH_GRN))
    if conf.get("SPATH_TRIMMED_GRN"):
        save_grn(trimmed, Path(conf.SPATH_TRIMMED_GRN))


if __name__ == "__main__":
    main()
