"""DNA sequence parser with multiple encoding options."""

import logging
from pathlib import Path
from typing import List, Optional, Union
import numpy as np

logger = logging.getLogger(__name__)


class DNAParser:
    """Parse and encode DNA sequences with various encoding schemes."""

    def __init__(self, encoding: str = "one-hot"):
        """
        Initialize the DNA parser with a specified encoding.

        Args:
            encoding: Type of encoding to use ('one-hot', 'integer', or 'kmer')
        """
        self.encoding = encoding.lower()

        if self.encoding == "one-hot":
            self.encoding_map = {
                "A": [1, 0, 0, 0],
                "T": [0, 1, 0, 0],
                "G": [0, 0, 1, 0],
                "C": [0, 0, 0, 1],
            }
        elif self.encoding == "integer":
            self.encoding_map = {
                "A": 0,
                "T": 1,
                "G": 2,
                "C": 3,
            }
        elif self.encoding == "kmer":
            self.encoding_map = None  # K-mer encoding is handled differently
        else:
            raise ValueError(f"Unsupported encoding type: {encoding}. Choose 'one-hot', 'integer', or 'kmer'.")

        logger.info(f"DNAParser initialized with encoding: {encoding}")

    def parse_file(self, file_path: Union[str, Path]) -> List[str]:
        """
        Parse DNA sequences from a text file.

        Args:
            file_path: Path to the text file

        Returns:
            List of sequences as strings
        """
        file_path = Path(file_path)
        if not file_path.exists():
            raise FileNotFoundError(f"File not found: {file_path}")

        sequences = []
        with open(file_path, "r") as f:
            for line in f:
                seq = line.strip().upper()
                if seq:  # Skip empty lines
                    sequences.append(seq)

        logger.info(f"Parsed {len(sequences)} sequences from {file_path}")
        return sequences

    def parse_fasta(self, file_path: Union[str, Path]) -> List[tuple[str, str]]:
        """
        Parse sequences from a FASTA file.

        Args:
            file_path: Path to the FASTA file

        Returns:
            List of (header, sequence) tuples
        """
        from Bio import SeqIO

        file_path = Path(file_path)
        if not file_path.exists():
            raise FileNotFoundError(f"File not found: {file_path}")

        sequences = []
        for record in SeqIO.parse(file_path, "fasta"):
            sequences.append((record.id, str(record.seq).upper()))

        logger.info(f"Parsed {len(sequences)} sequences from FASTA file {file_path}")
        return sequences

    def encode_sequence(self, sequence: str) -> np.ndarray:
        """
        Encode a single DNA sequence.

        Args:
            sequence: DNA sequence string

        Returns:
            Encoded sequence as numpy array
        """
        if self.encoding == "one-hot":
            encoded = []
            for nucleotide in sequence:
                if nucleotide in self.encoding_map:
                    encoded.extend(self.encoding_map[nucleotide])
                else:
                    # Handle unknown nucleotides (e.g., 'N') with all zeros
                    encoded.extend([0, 0, 0, 0])
            return np.array(encoded, dtype=np.float32)

        elif self.encoding == "integer":
            encoded = []
            for nucleotide in sequence:
                if nucleotide in self.encoding_map:
                    encoded.append(self.encoding_map[nucleotide])
                else:
                    # Handle unknown nucleotides with -1
                    encoded.append(-1)
            return np.array(encoded, dtype=np.int32)

        elif self.encoding == "kmer":
            # K-mer encoding
            return self._encode_kmer(sequence)

        else:
            raise ValueError(f"Unknown encoding: {self.encoding}")

    def _encode_kmer(self, sequence: str, k: int = 3) -> np.ndarray:
        """
        Encode sequence using K-mer frequency.

        Args:
            sequence: DNA sequence
            k: K-mer size

        Returns:
            K-mer frequency vector
        """
        kmer_counts = {}
        for i in range(len(sequence) - k + 1):
            kmer = sequence[i : i + k]
            kmer_counts[kmer] = kmer_counts.get(kmer, 0) + 1

        # Normalize by sequence length
        total_kmers = len(sequence) - k + 1
        kmer_freq = np.array([kmer_counts.get(kmer, 0) / total_kmers for kmer in self._generate_all_kmers(k)])
        return kmer_freq.astype(np.float32)

    def _generate_all_kmers(self, k: int) -> List[str]:
        """Generate all possible K-mers of length k."""
        import itertools

        bases = ["A", "T", "G", "C"]
        return ["".join(kmer) for kmer in itertools.product(bases, repeat=k)]

    def encode_sequences(self, sequences: List[str]) -> np.ndarray:
        """
        Encode a list of DNA sequences.

        Args:
            sequences: List of DNA sequences

        Returns:
            Array of encoded sequences
        """
        encoded_sequences = []
        for seq in sequences:
            encoded = self.encode_sequence(seq)
            encoded_sequences.append(encoded)

        return np.array(encoded_sequences)

    def validate_sequence(self, sequence: str) -> bool:
        """
        Validate that a sequence contains only valid DNA nucleotides.

        Args:
            sequence: DNA sequence to validate

        Returns:
            True if sequence is valid, False otherwise
        """
        valid_bases = set("ATCGN")
        return all(base.upper() in valid_bases for base in sequence)

    def get_sequence_stats(self, sequence: str) -> dict:
        """
        Get statistics about a DNA sequence.

        Args:
            sequence: DNA sequence

        Returns:
            Dictionary with sequence statistics
        """
        stats = {
            "length": len(sequence),
            "A": sequence.upper().count("A"),
            "T": sequence.upper().count("T"),
            "G": sequence.upper().count("G"),
            "C": sequence.upper().count("C"),
            "N": sequence.upper().count("N"),
            "GC_content": (sequence.upper().count("G") + sequence.upper().count("C")) / len(sequence) * 100,
        }
        return stats
