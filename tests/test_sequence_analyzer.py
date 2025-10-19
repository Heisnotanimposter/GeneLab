"""Tests for SequenceAnalyzer."""

import pytest
from genelab.core.sequence_analyzer import SequenceAnalyzer


class TestSequenceAnalyzer:
    """Test cases for SequenceAnalyzer."""

    def setup_method(self):
        """Set up test fixtures."""
        self.analyzer = SequenceAnalyzer()

    def test_hamming_distance_identical(self):
        """Test Hamming distance for identical sequences."""
        seq1 = "ATCGATCG"
        seq2 = "ATCGATCG"
        assert self.analyzer.hamming_distance(seq1, seq2) == 0

    def test_hamming_distance_different(self):
        """Test Hamming distance for different sequences."""
        seq1 = "ATCGATCG"
        seq2 = "GCTAGCTA"
        assert self.analyzer.hamming_distance(seq1, seq2) == 8

    def test_hamming_distance_mismatched_length(self):
        """Test Hamming distance raises error for mismatched lengths."""
        seq1 = "ATCG"
        seq2 = "ATCGA"
        with pytest.raises(ValueError):
            self.analyzer.hamming_distance(seq1, seq2)

    def test_similarity_score_identical(self):
        """Test similarity score for identical sequences."""
        seq1 = "ATCGATCG"
        seq2 = "ATCGATCG"
        assert self.analyzer.similarity_score(seq1, seq2) == 1.0

    def test_similarity_score_different(self):
        """Test similarity score for different sequences."""
        seq1 = "ATCGATCG"
        seq2 = "GCTAGCTA"
        assert self.analyzer.similarity_score(seq1, seq2) == 0.0

    def test_calculate_gc_content(self):
        """Test GC content calculation."""
        seq = "ATCGATCG"
        gc_content = self.analyzer.calculate_gc_content(seq)
        assert gc_content == 50.0

    def test_calculate_gc_content_empty(self):
        """Test GC content for empty sequence."""
        seq = ""
        gc_content = self.analyzer.calculate_gc_content(seq)
        assert gc_content == 0.0

    def test_find_mutations(self):
        """Test finding mutations between sequences."""
        seq1 = "ATCGATCG"
        seq2 = "ATCGATCG"
        mutations = self.analyzer.find_mutations(seq1, seq2)
        assert len(mutations) == 0

    def test_find_mutations_with_changes(self):
        """Test finding mutations with actual changes."""
        seq1 = "ATCGATCG"
        seq2 = "GTCGATCG"
        mutations = self.analyzer.find_mutations(seq1, seq2)
        assert len(mutations) == 1
        assert mutations[0]["position"] == 0
        assert mutations[0]["reference"] == "A"
        assert mutations[0]["mutant"] == "G"

    def test_reverse_complement(self):
        """Test reverse complement."""
        seq = "ATCG"
        rev_comp = self.analyzer.reverse_complement(seq)
        assert rev_comp == "CGAT"

    def test_analyze_sequence(self):
        """Test sequence analysis."""
        sequence = "ATCGATCGATCGATCG"
        mutation_positions = [0, 4, 8]
        expected_bases = {0: "A", 4: "A", 8: "A"}
        results = self.analyzer.analyze_sequence(sequence, mutation_positions, expected_bases)
        assert len(results) == 3
        assert "No mutation" in results[0]

    def test_analyze_sequence_with_mutations(self):
        """Test sequence analysis with mutations."""
        sequence = "GTCGATCGATCGATCG"
        mutation_positions = [0]
        expected_bases = {0: "A"}
        results = self.analyzer.analyze_sequence(sequence, mutation_positions, expected_bases)
        assert len(results) == 1
        assert "Mutation" in results[0]
