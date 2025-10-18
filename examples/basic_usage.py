"""Basic usage examples for GeneLab."""

from genelab import (
    GeneticDataFetcher,
    SequenceAnalyzer,
    DNAParser,
    GeneticMutation,
)
from genelab.utils import setup_logging

# Set up logging
setup_logging(level="INFO")

def example_sequence_parsing():
    """Example: Parse and encode DNA sequences."""
    print("\n=== Example 1: Sequence Parsing ===")
    
    # Create parser
    parser = DNAParser(encoding="one-hot")
    
    # Sample sequences
    sequences = [
        "ATCGATCGATCGATCG",
        "GCTAGCTAGCTAGCTA",
        "TTAACCGGAATTCCGG",
    ]
    
    # Encode sequences
    encoded = parser.encode_sequences(sequences)
    
    print(f"Number of sequences: {len(sequences)}")
    print(f"Encoded shape: {encoded.shape}")
    print(f"First sequence encoded: {encoded[0][:20]}...")

def example_sequence_analysis():
    """Example: Analyze DNA sequences."""
    print("\n=== Example 2: Sequence Analysis ===")
    
    analyzer = SequenceAnalyzer()
    
    # Sample sequences
    seq1 = "ATCGATCGATCGATCG"
    seq2 = "ATCGATCGATCGATCG"  # Identical
    seq3 = "GCTAGCTAGCTAGCTA"  # Different
    
    # Calculate similarity
    similarity1 = analyzer.similarity_score(seq1, seq2)
    similarity2 = analyzer.similarity_score(seq1, seq3)
    
    print(f"Similarity between identical sequences: {similarity1:.2f}")
    print(f"Similarity between different sequences: {similarity2:.2f}")
    
    # Find mutations
    mutations = analyzer.find_mutations(seq1, seq3)
    print(f"Number of mutations: {len(mutations)}")
    
    # Calculate GC content
    gc_content = analyzer.calculate_gc_content(seq1)
    print(f"GC content: {gc_content:.2f}%")

def example_genetic_mutation():
    """Example: Genetic mutation simulation."""
    print("\n=== Example 3: Genetic Mutation ===")
    
    # Create mutation operator
    mutator = GeneticMutation(mutation_rate=0.1)  # 10% mutation rate
    
    # Original sequence
    original = "ATCGATCGATCGATCG"
    print(f"Original sequence: {original}")
    
    # Mutate sequence
    mutated = mutator.mutate_sequence(original)
    print(f"Mutated sequence:  {mutated}")
    
    # Analyze mutations
    analyzer = SequenceAnalyzer()
    mutations = analyzer.find_mutations(original, mutated)
    print(f"Number of mutations: {len(mutations)}")
    
    for mut in mutations[:3]:  # Show first 3 mutations
        print(f"  Position {mut['position']}: {mut['reference']} -> {mut['mutant']} ({mut['type']})")

def example_population_evolution():
    """Example: Evolve a population of sequences."""
    print("\n=== Example 4: Population Evolution ===")
    
    # Define fitness function (simple: prefer more A's)
    def fitness_function(sequence):
        return sequence.count('A') / len(sequence)
    
    # Create mutation operator
    mutator = GeneticMutation(mutation_rate=0.05, elitism_rate=0.2)
    
    # Create initial population
    template = "ATCGATCGATCGATCG"
    population = mutator.create_population(template, population_size=20)
    
    print(f"Initial population size: {len(population)}")
    print(f"Initial best fitness: {max(fitness_function(seq) for seq in population):.2f}")
    
    # Evolve population
    final_pop, final_scores = mutator.evolve_population(
        population,
        fitness_function=fitness_function,
        generations=10,
        verbose=False
    )
    
    print(f"Final best fitness: {max(final_scores):.2f}")
    print(f"Final best sequence: {final_pop[np.argmax(final_scores)]}")

def example_ncbi_fetch():
    """Example: Fetch sequence from NCBI (commented out to avoid API calls)."""
    print("\n=== Example 5: NCBI Sequence Fetching ===")
    print("Note: This example is commented out to avoid making API calls.")
    print("Uncomment to use with a valid email address.")
    
    # Uncomment to use:
    # fetcher = GeneticDataFetcher(email="your.email@example.com")
    # sequence = fetcher.fetch_sequence("NC_000001")
    # if sequence:
    #     print(f"Fetched sequence length: {len(sequence)}")
    #     print(f"First 100 bases: {sequence[:100]}")

if __name__ == "__main__":
    import numpy as np
    
    print("GeneLab Basic Usage Examples")
    print("=" * 50)
    
    example_sequence_parsing()
    example_sequence_analysis()
    example_genetic_mutation()
    example_population_evolution()
    example_ncbi_fetch()
    
    print("\n" + "=" * 50)
    print("Examples completed!")
