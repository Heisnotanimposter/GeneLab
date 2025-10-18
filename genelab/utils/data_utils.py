"""Data processing utilities for GeneLab."""

import logging
from typing import Tuple
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer

logger = logging.getLogger(__name__)


def get_k_mers(sequence: str, k: int) -> str:
    """
    Divide a sequence into k-mers (overlapping substrings).

    Args:
        sequence: DNA sequence
        k: Size of k-mers

    Returns:
        String of space-separated k-mers
    """
    kmers = []
    for i in range(len(sequence) - k + 1):
        kmers.append(sequence[i : i + k])
    return " ".join(kmers)


def generate_training_data(sequences: list[str], k: int = 6) -> np.ndarray:
    """
    Generate k-mer training data from sequences.

    Args:
        sequences: List of DNA sequences
        k: Size of k-mers

    Returns:
        Array of k-mer strings
    """
    kmers_list = []
    for seq in sequences:
        kmers = get_k_mers(seq, k)
        kmers_list.append(kmers)

    return np.array(kmers_list)


def vectorize_features(
    sequences: np.ndarray, ngram_range: Tuple[int, int] = (4, 4), analyzer: str = "word"
) -> Tuple[np.ndarray, TfidfVectorizer]:
    """
    Vectorize sequences using TF-IDF.

    Args:
        sequences: Array of k-mer strings
        ngram_range: Range of n-grams to extract
        analyzer: Analyzer type ('word' or 'char')

    Returns:
        Tuple of (vectorized features, fitted vectorizer)
    """
    vectorizer = TfidfVectorizer(ngram_range=ngram_range, analyzer=analyzer)
    X_transformed = vectorizer.fit_transform(sequences)

    logger.info(
        f"Vectorized {len(sequences)} sequences with {X_transformed.shape[1]} features "
        f"(ngram_range={ngram_range}, analyzer={analyzer})"
    )

    return X_transformed, vectorizer


def pad_sequences(sequences: list[str], max_length: Optional[int] = None) -> np.ndarray:
    """
    Pad sequences to the same length.

    Args:
        sequences: List of sequences
        max_length: Maximum length (if None, uses longest sequence)

    Returns:
        Array of padded sequences
    """
    if max_length is None:
        max_length = max(len(seq) for seq in sequences)

    padded = []
    for seq in sequences:
        padded_seq = seq + "N" * (max_length - len(seq))
        padded.append(padded_seq)

    return np.array(padded)


def split_sequences(sequences: list[str], train_ratio: float = 0.7, val_ratio: float = 0.15) -> Tuple[list, list, list]:
    """
    Split sequences into train, validation, and test sets.

    Args:
        sequences: List of sequences
        train_ratio: Proportion for training set
        val_ratio: Proportion for validation set

    Returns:
        Tuple of (train, val, test) sequences
    """
    total = len(sequences)
    train_size = int(total * train_ratio)
    val_size = int(total * val_ratio)

    train = sequences[:train_size]
    val = sequences[train_size : train_size + val_size]
    test = sequences[train_size + val_size :]

    return train, val, test
