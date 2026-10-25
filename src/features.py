"""
Biological sequence encoding and feature transformation functions.
Supports: One-Hot, BLOSUM62 matrix scoring, and AAPC (Amino Acid Pair Composition).
"""

from typing import List, Dict
import numpy as np
import pandas as pd

AMINO_ACIDS = list("ACDEFGHIKLMNPQRSTVWY")
AA_TO_IDX = {aa: i for i, aa in enumerate(AMINO_ACIDS)}

# Standard BLOSUM62 Substitution Matrix
BLOSUM62: Dict[str, np.ndarray] = {
	'A': np.array([4, -1, -2, -2, 0, -1, -1, 0, -2, -1, -1, -1, -1, -2, -1, 1, 0, -3, -2, 0], dtype=np.float32),
	'R': np.array([-1, 5, 0, -2, -3, 1, 0, -2, 0, -3, -2, 2, -1, -3, -2, -1, -1, -3, -2, -3], dtype=np.float32),
	'N': np.array([-2, 0, 6, 1, -3, 0, 0, 0, 1, -3, -3, 0, -2, -3, -2, 1, 0, -4, -2, -3], dtype=np.float32),
	'D': np.array([-2, -2, 1, 6, -3, 0, 2, -1, -1, -3, -4, -1, -3, -3, -1, 0, -1, -4, -3, -3], dtype=np.float32),
	'C': np.array([0, -3, -3, -3, 9, -3, -4, -3, -3, -1, -1, -3, -1, -2, -3, -1, -1, -2, -2, -1], dtype=np.float32),
	'Q': np.array([-1, 1, 0, 0, -3, 5, 2, -2, 0, -3, -2, 1, 0, -3, -1, 0, -1, -2, -1, -2], dtype=np.float32),
	'E': np.array([-1, 0, 0, 2, -4, 2, 5, -2, 0, -3, -3, 1, -2, -3, -1, 0, -1, -3, -2, -2], dtype=np.float32),
	'G': np.array([0, -2, 0, -1, -3, -2, -2, 6, -2, -4, -4, -2, -3, -3, -2, 0, -2, -2, -3, -3], dtype=np.float32),
	'H': np.array([-2, 0, 1, -1, -3, 0, 0, -2, 8, -3, -3, -1, -2, -1, -2, -1, -2, -2, 2, -3], dtype=np.float32),
	'I': np.array([-1, -3, -3, -3, -1, -3, -3, -4, -3, 4, 2, -3, 1, 0, -3, -2, -1, -3, -1, 3], dtype=np.float32),
	'L': np.array([-1, -2, -3, -4, -1, -2, -3, -4, -3, 2, 4, -2, 2, 0, -3, -2, -1, -2, -1, 1], dtype=np.float32),
	'K': np.array([-1, 2, 0, -1, -3, 1, 1, -2, -1, -3, -2, 5, -1, -3, -1, 0, -1, -3, -2, -2], dtype=np.float32),
	'M': np.array([-1, -1, -2, -3, -1, 0, -2, -3, -2, 1, 2, -1, 5, 0, -2, -1, -1, -1, -1, 1], dtype=np.float32),
	'F': np.array([-2, -3, -3, -3, -2, -3, -3, -3, -1, 0, 0, -3, 0, 6, -4, -2, -2, 1, 3, -1], dtype=np.float32),
	'P': np.array([-1, -2, -2, -1, -3, -1, -1, -2, -2, -3, -3, -1, -2, -4, 7, -1, -1, -4, -3, -2], dtype=np.float32),
	'S': np.array([1, -1, 1, 0, -1, 0, 0, 0, -1, -2, -2, 0, -1, -2, -1, 4, 1, -3, -2, -2], dtype=np.float32),
	'T': np.array([0, -1, 0, -1, -1, -1, -1, -2, -2, -1, -1, -1, -1, -2, -1, 1, 5, -2, -2, 0], dtype=np.float32),
	'W': np.array([-3, -3, -4, -4, -2, -2, -3, -2, -2, -3, -2, -3, -1, 1, -4, -3, -2, 11, 2, -3], dtype=np.float32),
	'Y': np.array([-2, -2, -2, -3, -2, -1, -2, -3, 2, -1, -1, -2, -1, 3, -3, -2, -2, 2, 7, -1], dtype=np.float32),
	'V': np.array([0, -3, -3, -3, -1, -2, -2, -3, -3, 3, 1, -2, 1, -1, -2, -2, 0, -3, -1, 4], dtype=np.float32),
}


def sequence_to_blosum(sequence: str, max_length: int = 31) -> np.ndarray:
	"""Encodes a sequence into a 1D BLOSUM62 vector (max_length * 20 dimensions)."""
	blosum_matrix = np.zeros((max_length, 20), dtype=np.float32)
	for i, aa in enumerate(sequence[:max_length]):
		if aa in BLOSUM62:
			blosum_matrix[i, :] = BLOSUM62[aa]
	return blosum_matrix.flatten()


def sequence_to_aapc(sequence: str) -> np.ndarray:
	"""Encodes a sequence using Amino Acid Pair Composition (400 dipeptide features)."""
	aapc_vector = np.zeros(400, dtype=np.float32)
	if len(sequence) < 2:
		return aapc_vector

	total_pairs = len(sequence) - 1
	for i in range(total_pairs):
		aa1, aa2 = sequence[i], sequence[i + 1]
		if aa1 in AA_TO_IDX and aa2 in AA_TO_IDX:
			idx = AA_TO_IDX[aa1] * 20 + AA_TO_IDX[aa2]
			aapc_vector[idx] += 1.0

	if total_pairs > 0:
		aapc_vector /= total_pairs
	return aapc_vector
