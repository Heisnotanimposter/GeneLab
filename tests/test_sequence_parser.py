"""Tests for DNAParser."""

import pytest
import numpy as np
from genelab.core.sequence_parser import DNAParser


class TestDNAParser:
    """Test cases for DNAParser."""

    def test_one_hot_encoding(self):
        """Test one-hot encoding."""
        parser = DNAParser(encoding="one-hot")
        seq = "ATCG"
        encoded = parser.encode_sequence(seq)
        
        assert encoded.shape == (16,)  # 4 bases * 4 positions
        assert encoded.dtype == np.float32

    def test_integer_encoding(self):
        """Test integer encoding."""
        parser = DNAParser(encoding="integer")
        seq = "ATCG"
        encoded = parser.encode_sequence(seq)
        
        assert encoded.shape == (4,)
        assert encoded.dtype == np.int32
        assert np.array_equal(encoded, [0, 1, 3, 2])

    def test_invalid_encoding(self):
        """Test invalid encoding raises error."""
        with pytest.raises(ValueError):
            DNAParser(encoding="invalid")

    def test_validate_sequence_valid(self):
        """Test validation of valid sequence."""
        parser = DNAParser()
        assert parser.validate_sequence("ATCGATCG")

    def test_validate_sequence_invalid(self):
        """Test validation of invalid sequence."""
        parser = DNAParser()
        assert not parser.validate_sequence("ATCGXTCG")

    def test_get_sequence_stats(self):
        """Test sequence statistics."""
        parser = DNAParser()
        seq = "ATCGATCG"
        stats = parser.get_sequence_stats(seq)
        
        assert stats["length"] == 8
        assert stats["A"] == 2
        assert stats["T"] == 2
        assert stats["G"] == 2
        assert stats["C"] == 2
        assert stats["GC_content"] == 50.0

    def test_encode_multiple_sequences(self):
        """Test encoding multiple sequences."""
        parser = DNAParser(encoding="integer")
        sequences = ["ATCG", "GCTA", "TTAA"]
        encoded = parser.encode_sequences(sequences)
        
        assert encoded.shape == (3, 4)
        assert encoded.dtype == np.int32
