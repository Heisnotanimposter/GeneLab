"""Genetic mutation operators for evolutionary algorithms."""

import logging
from typing import List, Optional, Dict
import numpy as np

logger = logging.getLogger(__name__)


class GeneticMutation:
    """Genetic mutation operators for DNA sequences."""

    def __init__(
        self,
        mutation_rate: float = 0.01,
        elitism_rate: float = 0.1,
        crossover_rate: float = 0.7,
    ):
        """
        Initialize the genetic mutation operator.

        Args:
            mutation_rate: Base probability of mutation per nucleotide
            elitism_rate: Proportion of population to preserve as elites
            crossover_rate: Probability of crossover between parents
        """
        self.base_mutation_rate = mutation_rate
        self.elitism_rate = elitism_rate
        self.crossover_rate = crossover_rate
        self.bases = ["A", "T", "C", "G"]

        logger.info(
            f"GeneticMutation initialized: mutation_rate={mutation_rate}, "
            f"elitism_rate={elitism_rate}, crossover_rate={crossover_rate}"
        )

    def mutate_sequence(
        self, sequence: str, variable_mutation_rate: Optional[Dict[int, float]] = None
    ) -> str:
        """
        Mutate a DNA sequence with optional position-specific mutation rates.

        Args:
            sequence: DNA sequence to mutate
            variable_mutation_rate: Dictionary mapping positions to additional mutation rates

        Returns:
            Mutated sequence
        """
        sequence = list(sequence)
        mutations_applied = 0

        for i in range(len(sequence)):
            current_mutation_rate = self.base_mutation_rate

            if variable_mutation_rate and i in variable_mutation_rate:
                current_mutation_rate += variable_mutation_rate[i]

            if np.random.rand() < current_mutation_rate:
                original_base = sequence[i]
                new_bases = self.bases.copy()
                new_bases.remove(original_base)
                sequence[i] = np.random.choice(new_bases)
                mutations_applied += 1

        logger.debug(f"Applied {mutations_applied} mutations to sequence of length {len(sequence)}")
        return "".join(sequence)

    def crossover(self, parent1: str, parent2: str) -> tuple[str, str]:
        """
        Perform crossover between two parent sequences.

        Args:
            parent1: First parent sequence
            parent2: Second parent sequence

        Returns:
            Tuple of two offspring sequences
        """
        if len(parent1) != len(parent2):
            raise ValueError(f"Parent sequences must have equal length. Got {len(parent1)} and {len(parent2)}")

        if np.random.rand() > self.crossover_rate:
            return parent1, parent2

        point = np.random.randint(1, len(parent1) - 1)
        child1 = parent1[:point] + parent2[point:]
        child2 = parent2[:point] + parent1[point:]

        logger.debug(f"Performed crossover at position {point}")
        return child1, child2

    def select_elites(
        self, population: List[str], scores: List[float], elite_size: Optional[int] = None
    ) -> List[str]:
        """
        Select elite individuals from a population based on scores.

        Args:
            population: List of sequences
            scores: List of fitness scores corresponding to sequences
            elite_size: Number of elites to select (default: based on elitism_rate)

        Returns:
            List of elite sequences
        """
        if elite_size is None:
            elite_size = int(len(population) * self.elitism_rate)

        if elite_size == 0:
            elite_size = 1

        elite_size = min(elite_size, len(population))

        sorted_indices = np.argsort(scores)[::-1]
        elites = [population[i] for i in sorted_indices[:elite_size]]

        logger.debug(f"Selected {len(elites)} elites from population of {len(population)}")
        return elites

    def create_population(self, template: str, population_size: int) -> List[str]:
        """
        Create a population of sequences by mutating a template.

        Args:
            template: Template sequence
            population_size: Number of individuals in population

        Returns:
            List of sequences
        """
        population = []
        for _ in range(population_size):
            mutated = self.mutate_sequence(template)
            population.append(mutated)

        logger.info(f"Created population of {population_size} from template of length {len(template)}")
        return population

    def evolve_population(
        self,
        population: List[str],
        fitness_function: callable,
        generations: int = 10,
        verbose: bool = True,
    ) -> tuple[List[str], List[float]]:
        """
        Evolve a population over multiple generations.

        Args:
            population: Initial population
            fitness_function: Function that takes a sequence and returns a fitness score
            generations: Number of generations to evolve
            verbose: Whether to log progress

        Returns:
            Tuple of (final population, fitness scores)
        """
        current_population = population.copy()
        history = []

        for generation in range(generations):
            # Evaluate fitness
            scores = [fitness_function(seq) for seq in current_population]

            # Track best fitness
            best_fitness = max(scores)
            history.append(best_fitness)

            if verbose:
                logger.info(f"Generation {generation + 1}/{generations}: Best fitness = {best_fitness:.4f}")

            # Select elites
            elites = self.select_elites(current_population, scores)

            # Create new population through crossover and mutation
            new_population = elites.copy()

            while len(new_population) < len(current_population):
                # Select parents (tournament selection)
                parent1 = self._tournament_selection(current_population, scores)
                parent2 = self._tournament_selection(current_population, scores)

                # Crossover
                child1, child2 = self.crossover(parent1, parent2)

                # Mutate
                child1 = self.mutate_sequence(child1)
                child2 = self.mutate_sequence(child2)

                new_population.extend([child1, child2])

            # Trim to original population size
            current_population = new_population[: len(population)]

        # Final evaluation
        final_scores = [fitness_function(seq) for seq in current_population]

        return current_population, final_scores

    def _tournament_selection(
        self, population: List[str], scores: List[float], tournament_size: int = 3
    ) -> str:
        """
        Select an individual using tournament selection.

        Args:
            population: List of sequences
            scores: List of fitness scores
            tournament_size: Number of individuals in tournament

        Returns:
            Selected sequence
        """
        tournament_indices = np.random.choice(len(population), tournament_size, replace=False)
        tournament_scores = [scores[i] for i in tournament_indices]
        winner_idx = tournament_indices[np.argmax(tournament_scores)]
        return population[winner_idx]
