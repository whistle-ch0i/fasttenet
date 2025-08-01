import os
import os.path as osp
import re
from pathlib import Path
from typing import Optional, Dict, Tuple, List, Any

import numpy as np
from itertools import permutations

import mate
from fasttenet.utils import load_exp_data, load_time_data, get_device_list, align_data


# Precompile the regex once for efficiency
CHR_PEAK_PATTERN = re.compile(r"^chr[0-9XY]+-[0-9]+-[0-9]+$")


def load_gene_chr_mapping(filename: str) -> Dict[str, str]:
    """Load gene-to-chromosome mapping from file."""
    mapping: Dict[str, str] = {}
    with open(filename, "r") as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) != 2:
                continue  # skip malformed lines
            chr_, gene = parts
            mapping[gene] = chr_
    return mapping


def extract_gene_peaks_from_exp(node_names: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """Separate gene names and peak names based on naming pattern."""
    # Boolean mask for peaks
    is_peak = np.vectorize(lambda x: bool(CHR_PEAK_PATTERN.match(x)))(node_names)
    gene = node_names[~is_peak]
    peaks = node_names[is_peak]
    return gene, peaks


def extract_gene_peaks_idx_from_exp(node_names: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """Get indices of genes and peaks based on naming pattern."""
    is_peak_mask = np.vectorize(lambda x: bool(CHR_PEAK_PATTERN.match(x)))(node_names)
    gene_indices = np.nonzero(~is_peak_mask)[0].astype(np.int32)
    peak_indices = np.nonzero(is_peak_mask)[0].astype(np.int32)
    return gene_indices, peak_indices


class FastTENET:
    def __init__(
        self,
        dpath_exp_data: Optional[str] = None,
        dpath_trj_data: Optional[str] = None,
        dpath_branch_data: Optional[str] = None,
        dpath_tf_data: Optional[str] = None,
        spath_result_matrix: Optional[str] = None,
        make_binary: bool = False,
        config: Optional[Dict[str, Any]] = None,
        aligned_data: Optional[np.ndarray] = None,
        node_name: Optional[np.ndarray] = None,
        tfs: Optional[np.ndarray] = None,
    ):
        self._tf: Optional[np.ndarray] = None
        self._result_matrix: Optional[np.ndarray] = None
        self._refined_data: Optional[np.ndarray] = None
        self._mate: Optional[Any] = None

        self._node_name: Optional[np.ndarray] = None
        self._exp_data: Optional[np.ndarray] = None
        self._trajectory: Optional[np.ndarray] = None
        self._branch: Optional[np.ndarray] = None
        self._gene_name: Optional[np.ndarray] = None
        self._peak_name: Optional[np.ndarray] = None
        self._gene_idx: Optional[np.ndarray] = None
        self._peak_idx: Optional[np.ndarray] = None
        self._gene_chr_mapping: Optional[Dict[str, str]] = None
        self._spath_result_matrix: Optional[str] = None

        if config:
            inits = config["INIT"]
            droot = Path(inits["DROOT"])
            dpath_exp_data = str(droot / inits["FPATH_EXP"])
            dpath_trj_data = str(droot / inits["FPATH_TRJ"])
            dpath_branch_data = str(droot / inits["FPATH_BRANCH"])
            dpath_gene_chr_data = str(droot / inits["FPATH_GENE_CHR"])  # TENET+
            if "FPATH_TF" in inits:
                dpath_tf_data = str(droot / inits["FPATH_TF"])
            make_binary = bool(inits.get("MAKE_BINARY", False))

            self._node_name, self._exp_data = load_exp_data(dpath_exp_data, make_binary)
            self._trajectory = load_time_data(dpath_trj_data, dtype=np.float32)
            self._branch = load_time_data(dpath_branch_data, dtype=np.int32)

            # TENET+ additions
            self._gene_name, self._peak_name = extract_gene_peaks_from_exp(self._node_name)
            self._gene_idx, self._peak_idx = extract_gene_peaks_idx_from_exp(self._node_name)
            self._gene_chr_mapping = load_gene_chr_mapping(dpath_gene_chr_data)

            if dpath_tf_data:
                self._tf = np.loadtxt(dpath_tf_data, dtype=str)

            self._spath_result_matrix = str(droot / inits["SPATH_RESULT"])
        else:
            if aligned_data is not None:
                if node_name is None:
                    raise ValueError("node name must be provided when using refined_data directly")
                self._refined_data = aligned_data
                self._node_name = node_name
                self._tf = tfs
            else:
                if not (dpath_exp_data and dpath_trj_data and dpath_branch_data):
                    raise ValueError(
                        "dpath_exp_data, dpath_trj_data, and dpath_branch_data must be provided when not using config or aligned_data"
                    )
                if dpath_exp_data:
                    self._node_name, self._exp_data = load_exp_data(dpath_exp_data, make_binary)
                    self._gene_name, self._peak_name = extract_gene_peaks_from_exp(self._node_name)
                    self._gene_idx, self._peak_idx = extract_gene_peaks_idx_from_exp(self._node_name)
                if dpath_trj_data:
                    self._trajectory = load_time_data(dpath_trj_data, dtype=np.float32)
                if dpath_branch_data:
                    self._branch = load_time_data(dpath_branch_data, dtype=np.int32)
                if dpath_exp_data:
                    # assume gene_chr file path was intended to be provided; otherwise this will raise
                    # keep previous variable name for compatibility
                    try:
                        self._gene_chr_mapping = load_gene_chr_mapping(dpath_gene_chr_data)  # type: ignore[name-defined]
                    except NameError:
                        raise ValueError("dpath_gene_chr_data is required when not using config but used gene-chr logic")

                if dpath_tf_data:
                    self._tf = np.loadtxt(dpath_tf_data, dtype=str)

            self._spath_result_matrix = spath_result_matrix

        self._mate = None

    def save_result_matrix(self, spath_result_matrix: Optional[str] = None) -> None:
        """Save the result matrix to file, with header row/column."""
        target_path = spath_result_matrix or self._spath_result_matrix
        if not target_path:
            raise ValueError("Result matrix path is undefined")
        if self._result_matrix is None:
            raise ValueError("Result matrix is not computed yet")

        node_names = self._node_name  # type: ignore[attr-defined]
        tmp_rm = np.concatenate([node_names[:, None], self._result_matrix.astype(str)], axis=1)
        extended_nn = np.concatenate((["TE"], node_names))
        tmp_rm = np.concatenate([extended_nn[None, :], tmp_rm])

        np.savetxt(target_path, tmp_rm.T, delimiter="\t", fmt="%s")
        print(f"Save result matrix: {target_path}")

    def run(
        self,
        backend: Optional[str] = None,
        device_ids: Optional[Any] = None,
        procs_per_device: Optional[int] = None,
        batch_size: int = 0,
        num_kernels: int = 1,
        binning_method: str = "FSBW-L",
        kp: float = 0.5,
        binning_opt: Optional[dict] = None,
        smoothing_opt: Optional[dict] = None,
        dt: int = 1,
        config: Optional[Dict[str, Any]] = None,
    ) -> np.ndarray:
        # Determine backend
        if not backend:
            backend = config.get("BACKEND") if config else "cpu"

        # Device IDs resolution
        if device_ids is None:
            if config and "DEVICE_IDS" in config:
                ids = config["DEVICE_IDS"]
                device_ids = int(ids) if isinstance(ids, int) or (isinstance(ids, str) and ids.isdigit()) else list(ids)
            else:
                device_ids = [0] if "cpu" in backend.lower() else get_device_list()

        # Processes per device
        if procs_per_device is None:
            procs_per_device = int(config["PROCS_PER_DEVICE"]) if config and "PROCS_PER_DEVICE" in config else 1

        # Batch size determination
        if batch_size == 0 and backend.lower() != "tenet":
            if config and "BATCH_SIZE" in config:
                batch_string = config["BATCH_SIZE"].split("**")
                batch_size = int(batch_string[0]) ** int(batch_string[1]) if len(batch_string) > 1 else int(batch_string[0])
            else:
                raise ValueError("batch size must be defined when backend is not 'tenet'")

        # Override with config if present
        if config:
            binning_method = config.get("BINNING_METHOD", binning_method)
            kp = float(config.get("KP", kp))
            num_kernels = int(config.get("NUM_KERNELS", num_kernels))
            dt = int(config.get("DT", dt))
            binning_opt = config.get("BINNING_OPT", binning_opt)
            smoothing_opt = config.get("SMOOTHING_OPT", smoothing_opt)

        # Data preparation
        if self._refined_data is not None:
            arr = self._refined_data
        else:
            if self._exp_data is None or self._trajectory is None or self._branch is None:
                raise ValueError("Expression, trajectory, and branch data must be loaded before running")
            arr = align_data(data=self._exp_data, trj=self._trajectory, branch=self._branch)

        # Build pairs
        if self._tf is not None and self._node_name is not None:
            # Compute intersection once
            _, inds_source, _ = np.intersect1d(self._node_name, self._tf, return_indices=True)

            gene_idx = self._gene_idx if self._gene_idx is not None else np.array([], dtype=np.int32)
            peak_idx = self._peak_idx if self._peak_idx is not None else np.array([], dtype=np.int32)

            # Pairs from gene/peak to TFs (excluding self)
            pairs_list: List[Tuple[int, int]] = []
            for ix_t in np.concatenate((gene_idx, peak_idx)):
                # skip if ix_t overlaps with TF source index
                valid_sources = inds_source[inds_source != ix_t]
                for ix_s in valid_sources:
                    pairs_list.append((ix_t, ix_s))

            # Additional: gene-to-peak on same chromosome
            if self._gene_chr_mapping is not None and self._node_name is not None:
                # Precompute peak chromosome for quick lookup
                peak_chr_map: Dict[int, str] = {}
                for ix_s in peak_idx:
                    peak = self._node_name[ix_s]
                    peak_chr_map[ix_s] = peak.split("-")[0]

                for ix_t in gene_idx:
                    gene = self._node_name[ix_t]
                    gene_chr = self._gene_chr_mapping.get(gene)
                    if gene_chr is None:
                        continue
                    for ix_s, peak_chr in peak_chr_map.items():
                        if peak_chr == gene_chr and ix_t != ix_s:
                            pairs_list.append((ix_t, ix_s))

            if pairs_list:
                pairs = np.asarray(pairs_list, dtype=np.int32)
            else:
                # fallback to empty array with proper shape
                pairs = np.zeros((0, 2), dtype=np.int32)
        else:
            # Full permutation if no TFs defined
            pairs = np.asarray(tuple(permutations(range(arr.shape[0]), 2)), dtype=np.int32)

        # Run MATE / MATELightning depending on backend
        if backend.lower() in ("lightning", "gpu", "cuda"):
            self._mate = mate.MATELightning(
                arr=arr,
                pairs=pairs,
                kp=kp,
                num_kernels=num_kernels,
                binning_method=binning_method,
                binning_opt=binning_opt,
                smoothing_opt=smoothing_opt,
                dt=dt,
            )
            self._result_matrix = self._mate.run(
                backend="gpu", devices=device_ids, batch_size=batch_size, num_workers=procs_per_device
            )
        else:
            self._mate = mate.MATE(
                kp=kp,
                num_kernels=num_kernels,
                binning_method=binning_method,
                binning_opt=binning_opt,
                smoothing_opt=smoothing_opt,
            )
            self._result_matrix = self._mate.run(
                arr=arr,
                pairs=pairs,
                backend=backend,
                device_ids=device_ids,
                procs_per_device=procs_per_device,
                batch_size=batch_size,
                dt=dt,
            )

        if self._result_matrix is not None and self._spath_result_matrix is not None:
            self.save_result_matrix(self._spath_result_matrix)

        # Return transposed to match original API
        return self._result_matrix.T  # type: ignore[return-value]
