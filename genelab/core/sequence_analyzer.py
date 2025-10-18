"""Sequence analysis tools for genetic sequences."""

import logging
from typing import Dict, List, Optional
import numpy as np

logger = logging.getLogger(__name__)


class SequenceAnalyzer:
    """Analyze and compare genetic sequences."""

    @staticmethod
    def analyze_sequence(
        sequence: str, mutation_positions: List[int], expected_bases: Dict[int, str]
    ) -> List[str]:
        """
        Analyze a sequence for mutations at specific positions.

        Args:
            sequence: DNA sequence to analyze
            mutation_positions: List of positions to check
            expected_bases: Dictionary mapping positions to expected bases

        Returns:
            List of analysis results as strings
        """
        results = []
        for pos in mutation_positions:
            if pos < 0 or pos >= len(sequence):
                results.append(f"Position {pos} is out of range (sequence length: {len(sequence)}).")
                continue

            actual_base = sequence[pos]
            expected_base = expected_bases.get(pos, None)

            if expected_base is None:
                results.append(f"Position {pos}: {actual_base} (no expected base specified)")
            elif actual_base != expected_base:
                results.append(f"Mutation at {pos}: expected {expected_base}, found {actual_base}")
            else:
                results.append(f"No mutation at position {pos} (matches expected {expected_base})")

        return results

    @staticmethod
    def hamming_distance(seq1: str, seq2: str) -> int:
        """
        Calculate Hamming distance between two sequences.

        Args:
            seq1: First sequence
            seq2: Second sequence

        Returns:
            Hamming distance (number of differing positions)

        Raises:
            ValueError: If sequences have different lengths
        """
        if len(seq1) != len(seq2):
            raise ValueError(f"Sequences must have equal length. Got {len(seq1)} and {len(seq2)}")

        distance = sum(c1 != c2 for c1, c2 in zip(seq1, seq2))
        return distance

    @staticmethod
    def similarity_score(seq1: str, seq2: str) -> float:
        """
        Calculate similarity score between two sequences.

        Args:
            seq1: First sequence
            seq2: Second sequence

        Returns:
            Similarity score (0 to 1, where 1 is identical)
        """
        if len(seq1) == 0 or len(seq2) == 0:
            return 0.0

        hamming = SequenceAnalyzer.hamming_distance(seq1, seq2)
        similarity = 1 - (hamming / len(seq1))
        return similarity

    @staticmethod
    def find_mutations(seq1: str, seq2: str) -> List[Dict[str, any]]:
        """
        Find all mutations between two sequences.

        Args:
            seq1: Reference sequence
            seq2: Mutated sequence

        Returns:
            List of dictionaries with mutation details
        """
        if len(seq1) != len(seq2):
            logger.warning(f"Sequences have different lengths: {len(seq1)} vs {len(seq2)}")

        mutations = []
        min_len = min(len(seq1), len(seq2))

        for i in range(min_len):
            if seq1[i] != seq2[i]:
                mutations.append(
                    {
                        "position": i,
                        "reference": seq1[i],
                        "mutant": seq2[i],
                        "type": SequenceAnalyzer._get_mutation_type(seq1[i], seq2[i]),
                    }
                )

        return mutations

    @staticmethod
    def _get_mutation_type(ref: str, mut: str) -> str:
        """
        Determine the type of mutation.

        Args:
            ref: Reference base
            mut: Mutant base

        Returns:
            Mutation type (transition, transversion, or unknown)
        """
        purines = {"A", "G"}
        pyrimidines = {"T", "C"}

        if ref in purines and mut in purines:
            return "transition"
        elif ref in pyrimidines and mut in pyrimidines:
            return "transition"
        elif (ref in purines and mut in pyrimidines) or (ref in pyrimidines and mut in purines):
            return "transversion"
        else:
            return "unknown"

    @staticmethod
    def calculate_gc_content(sequence: str) -> float:
        """
        Calculate GC content of a sequence.

        Args:
            sequence: DNA sequence

        Returns:
            GC content as a percentage
        """
        if len(sequence) == 0:
            return 0.0

        gc_count = sequence.upper().count("G") + sequence.upper().count("C")
        return (gc_count / len(sequence)) * 100

    @staticmethod
    def find_repeats(sequence: str, min_repeat_length: int = 3) -> List[Dict[str, any]]:
        """
        Find repeat sequences in a DNA sequence.

        Args:
            sequence: DNA sequence to analyze
            min_repeat_length: Minimum length of repeats to find

        Returns:
            List of dictionaries with repeat information
        """
        repeats = []
        sequence = sequence.upper()

        for length in range(min_repeat_length, len(sequence) // 2 + 1):
            for i in range(len(sequence) - length * 2 + 1):
                repeat_seq = sequence[i : i + length]
                next_occurrence = sequence.find(repeat_seq, i + length)

                if next_occurrence != -1:
                    repeats.append(
                        {
                            "sequence": repeat_seq,
                            "length": length,
                            "first_position": i,
                            "second_position": next_occurrence,
                            "distance": next_occurrence - i,
                        }
                    )

        return repeats

    @staticmethod
    def reverse_complement(sequence: str) -> str:
        """
        Get the reverse complement of a DNA sequence.

        Args:
            sequence: DNA sequence

        Returns:
            Reverse complement sequence
        """
        complement_map = {"A": "T", "T": "A", "G": "C", "C": "G", "N": "N"}
        complement = "".join(complement_map.get(base, "N") for base in sequence.upper())
        return complement[::-1]
