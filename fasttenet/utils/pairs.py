"""Utility functions for generating node pairs."""

from typing import Dict, Iterable, List, Tuple, Optional


def load_gene_chr_mapping(filename: str) -> Dict[str, str]:
    """Load mapping from gene name to chromosome."""
    mapping: Dict[str, str] = {}
    with open(filename, "r") as f:
        for line in f:
            if not line.strip():
                continue
            chromosome, gene = line.strip().split()
            if gene.startswith("chr"):
                # skip lines that start with chr as gene name
                continue
            mapping[gene] = chromosome
    return mapping


def load_file_to_list(filename: str) -> List[str]:
    """Load a file into a list, stripping trailing newlines."""
    with open(filename, "r") as f:
        return [line.strip() for line in f if line.strip()]


# Pair generation functions return 0-indexed pairs ---------------------------

def create_TENET_Plus_pairs(
    peaks: Iterable[str],
    genes_per_chromosome: Dict[str, List[str]],
    gene_names_dict: Dict[str, int],
    tf_list: Iterable[str],
) -> List[Tuple[int, int]]:
    """Create gene–gene pairs based on TENET+ logic."""
    gene_pairs: List[Tuple[int, int]] = []
    for peak in peaks:
        peak_chr = peak.split("-")[0]
        peak_idx = gene_names_dict.get(peak)
        if peak_idx is not None:
            for gene in genes_per_chromosome.get(peak_chr, []):
                if gene in gene_names_dict:
                    gene_pairs.append((peak_idx, gene_names_dict[gene]))

    tf_indices = [gene_names_dict[tf] for tf in tf_list if tf in gene_names_dict]
    for tf_idx in tf_indices:
        for gene, gidx in gene_names_dict.items():
            if gidx != tf_idx:
                gene_pairs.append((tf_idx, gidx))

    return gene_pairs


def create_tf_peak_pairs(
    gene_names_dict: Dict[str, int],
    only_peak_names_dict: Dict[str, int],
    tf_list: Iterable[str],
) -> List[Tuple[int, int]]:
    """Create TF->peak pairs."""
    peak_pairs: List[Tuple[int, int]] = []
    tf_indices = [gene_names_dict[tf] for tf in tf_list if tf in gene_names_dict]
    for tf_idx in tf_indices:
        for peak, pidx in only_peak_names_dict.items():
            if pidx != tf_idx:
                peak_pairs.append((tf_idx, pidx))
    return peak_pairs


def create_tf_gene_pairs(
    gene_names_dict: Dict[str, int],
    only_gene_names_dict: Dict[str, int],
    tf_list: Iterable[str],
) -> List[Tuple[int, int]]:
    """Create TF->gene pairs."""
    gene_pairs: List[Tuple[int, int]] = []
    tf_indices = [gene_names_dict[tf] for tf in tf_list if tf in gene_names_dict]
    for tf_idx in tf_indices:
        for gene, gidx in only_gene_names_dict.items():
            if gidx != tf_idx:
                gene_pairs.append((tf_idx, gidx))
    return gene_pairs


def create_tf_gene_peak_pairs(
    gene_names_dict: Dict[str, int],
    tf_list: Iterable[str],
) -> List[Tuple[int, int]]:
    """Create TF->gene and TF->peak pairs in one list."""
    gene_pairs: List[Tuple[int, int]] = []
    tf_indices = [gene_names_dict[tf] for tf in tf_list if tf in gene_names_dict]
    for tf_idx in tf_indices:
        for gene, gidx in gene_names_dict.items():
            if gidx != tf_idx:
                gene_pairs.append((tf_idx, gidx))
    return gene_pairs


def create_cis_peaksource_pairs(
    peaks: Iterable[str],
    genes_per_chromosome: Dict[str, List[str]],
    gene_names_dict: Dict[str, int],
) -> List[Tuple[int, int]]:
    """Create cis peak->gene pairs."""
    gene_pairs: List[Tuple[int, int]] = []
    for peak in peaks:
        peak_chr = peak.split("-")[0]
        peak_idx = gene_names_dict.get(peak)
        if peak_idx is not None:
            for gene in genes_per_chromosome.get(peak_chr, []):
                if gene in gene_names_dict:
                    gene_pairs.append((peak_idx, gene_names_dict[gene]))
    return gene_pairs


def create_peak_peak_pairs(
    peaks: Iterable[str],
    only_peak_names_dict: Dict[str, int],
    max_distance: Optional[int] = 1000000,
) -> List[Tuple[int, int]]:
    """Create directed cis peak->peak pairs within an optional distance limit."""
    peak_pairs: List[Tuple[int, int]] = []
    peaks_by_chr: Dict[str, List[Tuple[str, int, int]]] = {}
    for peak in peaks:
        parts = peak.split("-")
        if len(parts) < 3:
            continue
        chrom, start_str, end_str = parts[0], parts[1], parts[2]
        try:
            start, end = int(start_str), int(end_str)
        except ValueError:
            continue
        if peak in only_peak_names_dict:
            peaks_by_chr.setdefault(chrom, []).append((peak, start, end))

    for chrom, plist in peaks_by_chr.items():
        for i in range(len(plist)):
            p1, s1, e1 = plist[i]
            for j in range(len(plist)):
                if i == j:
                    continue
                p2, s2, e2 = plist[j]
                if e1 < s2:
                    dist = s2 - e1
                elif e2 < s1:
                    dist = s1 - e2
                else:
                    dist = 0
                if max_distance is None or dist <= max_distance:
                    idx1 = only_peak_names_dict[p1]
                    idx2 = only_peak_names_dict[p2]
                    peak_pairs.append((idx1, idx2))
    return peak_pairs


def load_sif_connections(
    filename: str,
    gene_names_dict: Dict[str, int],
) -> Tuple[List[Tuple[int, int]], List[Tuple[int, int]], List[Tuple[int, int]]]:
    """Load connections from a SIF file."""
    tf_gene_pairs: List[Tuple[int, int]] = []
    tf_peak_pairs: List[Tuple[int, int]] = []
    peak_gene_pairs: List[Tuple[int, int]] = []

    with open(filename, "r") as f:
        header = f.readline()
        for line_number, line in enumerate(f, start=2):
            parts = line.strip().split()
            if len(parts) < 7:
                continue
            TF, gene, peak, *_ = parts
            if TF in gene_names_dict and gene in gene_names_dict:
                tf_gene_pairs.append((gene_names_dict[TF], gene_names_dict[gene]))
            if TF in gene_names_dict and peak in gene_names_dict:
                tf_peak_pairs.append((gene_names_dict[TF], gene_names_dict[peak]))
            if peak in gene_names_dict and gene in gene_names_dict:
                peak_gene_pairs.append((gene_names_dict[peak], gene_names_dict[gene]))

    return tf_gene_pairs, tf_peak_pairs, peak_gene_pairs

