# GeneLab Quick Start Guide

Get started with GeneLab in 5 minutes!

## Installation

```bash
# Clone the repository
git clone https://github.com/Heisnotanimposter/GeneLab.git
cd GeneLab

# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
pip install -e .
```

## Quick Examples

### 1. Analyze DNA Sequences

```python
from genelab import DNAParser, SequenceAnalyzer

# Create parser
parser = DNAParser(encoding="one-hot")

# Sample sequences
sequences = ["ATCGATCGATCGATCG", "GCTAGCTAGCTAGCTA"]

# Encode sequences
encoded = parser.encode_sequences(sequences)

# Analyze sequences
analyzer = SequenceAnalyzer()
similarity = analyzer.similarity_score(sequences[0], sequences[1])
print(f"Similarity: {similarity:.2f}")
```

### 2. Simulate Genetic Mutations

```python
from genelab import GeneticMutation

# Create mutation operator
mutator = GeneticMutation(mutation_rate=0.1)

# Mutate a sequence
original = "ATCGATCGATCGATCG"
mutated = mutator.mutate_sequence(original)

print(f"Original: {original}")
print(f"Mutated:  {mutated}")
```

### 3. Train a Machine Learning Model

```python
from genelab import TreatmentPredictor
from sklearn.model_selection import train_test_split
import numpy as np

# Generate synthetic data
X = np.random.rand(1000, 50)
y = np.random.randint(0, 2, 1000)

# Split data
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Train model
predictor = TreatmentPredictor(input_shape=(50,))
predictor.train(X_train, y_train, X_test, y_test, epochs=20)

# Evaluate
metrics = predictor.evaluate(X_test, y_test)
print(f"Accuracy: {metrics['accuracy']:.2f}")
```

### 4. Run the Web Interface

```bash
streamlit run app.py
```

Then open your browser to `http://localhost:8501`

## Configuration

Create a `config.yaml` file:

```yaml
entrez:
  email: "your.email@example.com"

data:
  sequences_path: "./data/sequences"
  models_path: "./models"

training:
  epochs: 50
  batch_size: 32
```

## Run Examples

```bash
# Basic usage examples
python examples/basic_usage.py

# Machine learning examples
python examples/ml_training.py
```

## Run Tests

```bash
# Run all tests
pytest

# Run with coverage
pytest --cov=genelab --cov-report=html
```

## Next Steps

1. **Read the Documentation**: Check out [README.md](README.md) for detailed documentation
2. **Explore Examples**: Look at the `examples/` directory
3. **Try the Web App**: Run `streamlit run app.py`
4. **Read the Migration Guide**: If upgrading from v1.0, see [MIGRATION_GUIDE.md](MIGRATION_GUIDE.md)

## Common Tasks

### Fetch Sequence from NCBI

```python
from genelab import GeneticDataFetcher

fetcher = GeneticDataFetcher(email="your.email@example.com")
sequence = fetcher.fetch_sequence("NC_000001")
print(f"Sequence length: {len(sequence)}")
```

### Analyze Mutations

```python
from genelab import SequenceAnalyzer

analyzer = SequenceAnalyzer()

seq1 = "ATCGATCGATCGATCG"
seq2 = "GTCGATCGATCGATCG"

mutations = analyzer.find_mutations(seq1, seq2)
print(f"Number of mutations: {len(mutations)}")

for mut in mutations:
    print(f"Position {mut['position']}: {mut['reference']} -> {mut['mutant']}")
```

### Evolve a Population

```python
from genelab import GeneticMutation

# Define fitness function
def fitness(sequence):
    return sequence.count('A') / len(sequence)

# Create mutation operator
mutator = GeneticMutation(mutation_rate=0.05)

# Create population
template = "ATCGATCGATCGATCG"
population = mutator.create_population(template, population_size=50)

# Evolve
final_pop, scores = mutator.evolve_population(
    population,
    fitness_function=fitness,
    generations=20
)

# Get best
best_idx = np.argmax(scores)
print(f"Best sequence: {final_pop[best_idx]}")
print(f"Best fitness: {scores[best_idx]:.2f}")
```

## Troubleshooting

### Import Error

```bash
pip install -e .
```

### Configuration Not Found

```python
from genelab.config import ConfigLoader
config = ConfigLoader()  # Uses defaults
```

### GPU Not Found

```python
import tensorflow as tf
print(tf.config.list_physical_devices('GPU'))
```

## Get Help

- **Documentation**: [README.md](README.md)
- **Examples**: `examples/` directory
- **Tests**: `tests/` directory
- **Issues**: https://github.com/Heisnotanimposter/GeneLab/issues

## Resources

- **BioPython**: https://biopython.org/
- **TensorFlow**: https://www.tensorflow.org/
- **Streamlit**: https://streamlit.io/
- **NumPy**: https://numpy.org/

Happy coding! 🧬
