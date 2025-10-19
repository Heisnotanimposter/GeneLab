"""Tests for GeneticMutation."""

import pytest
import numpy as np
from genelab.core.genetic_mutation import GeneticMutation


class TestGeneticMutation:
    """Test cases for GeneticMutation."""

    def setup_method(self):
        """Set up test fixtures."""
        self.mutator = GeneticMutation(mutation_rate=0.01)

    def test_mutate_sequence_no_mutations(self):
        """Test mutation with zero mutation rate."""
        mutator = GeneticMutation(mutation_rate=0.0)
        seq = "ATCGATCGATCGATCG"
        mutated = mutator.mutate_sequence(seq)
        assert mutated == seq

    def test_mutate_sequence_all_mutations(self):
        """Test mutation with 100% mutation rate."""
        mutator = GeneticMutation(mutation_rate=1.0)
        seq = "ATCGATCGATCGATCG"
        mutated = mutator.mutate_sequence(seq)
        assert mutated != seq

    def test_crossover(self):
        """Test crossover operation."""
        parent1 = "ATCGATCGATCGATCG"
        parent2 = "GCTAGCTAGCTAGCTA"
        child1, child2 = self.mutator.crossover(parent1, parent2)
        
        assert len(child1) == len(parent1)
        assert len(child2) == len(parent2)

    def test_crossover_mismatched_length(self):
        """Test crossover raises error for mismatched lengths."""
        parent1 = "ATCG"
        parent2 = "ATCGA"
        with pytest.raises(ValueError):
            self.mutator.crossover(parent1, parent2)

    def test_select_elites(self):
        """Test elite selection."""
        population = ["ATCG", "GCTA", "TTAA", "CCGG"]
        scores = [0.1, 0.5, 0.3, 0.9]
        elites = self.mutator.select_elites(population, scores, elite_size=2)
        
        assert len(elites) == 2
        assert elites[0] == "CCGG"  # Highest score
        assert elites[1] == "GCTA"  # Second highest

    def test_create_population(self):
        """Test population creation."""
        template = "ATCGATCGATCGATCG"
        population = self.mutator.create_population(template, population_size=10)
        
        assert len(population) == 10
        assert all(len(seq) == len(template) for seq in population)

    def test_evolve_population(self):
        """Test population evolution."""
        def fitness_function(sequence):
            return sequence.count('A') / len(sequence)
        
        template = "ATCGATCGATCGATCG"
        population = self.mutator.create_population(template, population_size=20)
        
        final_pop, final_scores = self.mutator.evolve_population(
            population,
            fitness_function=fitness_function,
            generations=5,
            verbose=False
        )
        
        assert len(final_pop) == len(population)
        assert len(final_scores) == len(population)
        assert max(final_scores) >= 0
