# GeneLab 2.0 - Comprehensive Bioinformatics Toolkit

[![Python Version](https://img.shields.io/badge/python-3.9+-blue.svg)](https://www.python.org/downloads/)
[![License](https://img.shields.io/badge/license-MIT-green.svg)](LICENSE)
[![Code Style](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)

A modern, comprehensive bioinformatics toolkit for gene mutation analysis, sequence processing, machine learning, and genomic data visualization.

## 🚀 Features

- **DNA/RNA Sequence Processing**: Parse, encode, and analyze genetic sequences
- **Genetic Mutation Simulation**: Advanced mutation operators for evolutionary algorithms
- **Machine Learning Models**: Deep learning models for sequence classification and prediction
- **Protein Structure Analysis**: U-Net for biomedical image segmentation
- **NCBI Integration**: Fetch sequences from NCBI databases with rate limiting
- **Reinforcement Learning**: RL agents for mutation optimization
- **Data Visualization**: Interactive visualizations with Plotly and Matplotlib
- **Web Interface**: Streamlit-based web application for easy access

## 📋 Table of Contents

- [Installation](#installation)
- [Quick Start](#quick-start)
- [Documentation](#documentation)
- [Examples](#examples)
- [Project Structure](#project-structure)
- [Contributing](#contributing)
- [License](#license)

## 🔧 Installation

### Prerequisites

- Python 3.9 or higher
- pip or conda

### Install from Source

```bash
# Clone the repository
git clone https://github.com/Heisnotanimposter/GeneLab.git
cd GeneLab

# Create a virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Install the package in development mode
pip install -e .
```

### Install with Optional Dependencies

```bash
# With development dependencies
pip install -e ".[dev]"

# With documentation dependencies
pip install -e ".[docs]"
```

## 🚀 Quick Start

### Basic Usage

```python
from genelab import GeneticDataFetcher, SequenceAnalyzer, DNAParser

# Fetch a sequence from NCBI
fetcher = GeneticDataFetcher(email="your.email@example.com")
sequence = fetcher.fetch_sequence("NC_000001")

# Parse and encode DNA sequences
parser = DNAParser(encoding="one-hot")
sequences = parser.parse_file("data/sequences.txt")
encoded = parser.encode_sequences(sequences)

# Analyze sequences
analyzer = SequenceAnalyzer()
similarity = analyzer.similarity_score(seq1, seq2)
mutations = analyzer.find_mutations(reference, mutated)
```

### Machine Learning

```python
from genelab import TreatmentPredictor
import numpy as np

# Create and train a model
predictor = TreatmentPredictor(input_shape=(100,))
predictor.train(X_train, y_train, X_val, y_val, epochs=50)

# Evaluate the model
metrics = predictor.evaluate(X_test, y_test)
print(f"Accuracy: {metrics['accuracy']:.2f}")

# Make predictions
predictions = predictor.predict(new_data)
```

### Genetic Mutation Simulation

```python
from genelab import GeneticMutation

# Initialize mutation operator
mutator = GeneticMutation(mutation_rate=0.01)

# Mutate a sequence
mutated = mutator.mutate_sequence(original_sequence)

# Evolve a population
population = mutator.create_population(template, population_size=100)
final_pop, scores = mutator.evolve_population(
    population,
    fitness_function=my_fitness_function,
    generations=50
)
```

## 📚 Documentation

### Core Modules

#### `genelab.core`

- **GeneticDataFetcher**: Fetch sequences from NCBI databases
- **SequenceAnalyzer**: Analyze and compare sequences
- **DNAParser**: Parse and encode DNA sequences
- **GeneticMutation**: Mutation operators for evolutionary algorithms

#### `genelab.models`

- **TreatmentPredictor**: Deep learning model for sequence classification
- **UNetModel**: U-Net for biomedical image segmentation

#### `genelab.utils`

- **Data utilities**: K-mer generation, vectorization, sequence processing
- **Logging utilities**: Centralized logging configuration

### Configuration

GeneLab uses a YAML configuration file (`config.yaml`) for settings:

```yaml
data:
  base_path: "./data"
  sequences_path: "./data/sequences"
  models_path: "./models"
  output_path: "./output"

entrez:
  email: "your.email@example.com"
  api_key: ""  # Optional
  max_retries: 3
  delay: 1.0

training:
  epochs: 50
  batch_size: 32
  validation_split: 0.2
  random_seed: 42
```

## 📖 Examples

### Example 1: Sequence Analysis

```python
from genelab import DNAParser, SequenceAnalyzer

# Parse sequences
parser = DNAParser(encoding="one-hot")
sequences = parser.parse_file("data/human.txt")

# Analyze sequences
analyzer = SequenceAnalyzer()
for seq in sequences[:5]:
    stats = parser.get_sequence_stats(seq)
    print(f"Length: {stats['length']}, GC Content: {stats['GC_content']:.2f}%")
```

### Example 2: Mutation Analysis

```python
from genelab import GeneticMutation, SequenceAnalyzer

# Create mutation operator
mutator = GeneticMutation(mutation_rate=0.01)

# Generate mutations
original = "ATCGATCGATCGATCG"
mutated = mutator.mutate_sequence(original)

# Analyze mutations
analyzer = SequenceAnalyzer()
mutations = analyzer.find_mutations(original, mutated)
similarity = analyzer.similarity_score(original, mutated)

print(f"Similarity: {similarity:.2f}")
print(f"Number of mutations: {len(mutations)}")
```

### Example 3: Training a Model

```python
from genelab import TreatmentPredictor
from genelab.utils import generate_training_data, vectorize_features
from sklearn.model_selection import train_test_split
import pandas as pd

# Load data
df = pd.read_csv("data/sequences.csv")

# Prepare features
sequences = df['sequence'].values
labels = df['label'].values

# Generate k-mers and vectorize
kmers = generate_training_data(sequences, k=6)
X, vectorizer = vectorize_features(kmers)

# Split data
X_train, X_test, y_train, y_test = train_test_split(
    X, labels, test_size=0.2, random_state=42
)

# Train model
predictor = TreatmentPredictor(input_shape=(X_train.shape[1],))
predictor.train(X_train.toarray(), y_train, X_test.toarray(), y_test, epochs=50)

# Evaluate
metrics = predictor.evaluate(X_test.toarray(), y_test)
print(metrics)
```

## 📁 Project Structure

```
GeneLab/
├── genelab/                    # Main package
│   ├── __init__.py
│   ├── core/                   # Core functionality
│   │   ├── data_fetcher.py
│   │   ├── sequence_analyzer.py
│   │   ├── sequence_parser.py
│   │   └── genetic_mutation.py
│   ├── models/                 # ML models
│   │   ├── predictor.py
│   │   └── unet.py
│   ├── utils/                  # Utilities
│   │   ├── data_utils.py
│   │   └── logging_utils.py
│   └── config/                 # Configuration
│       └── config_loader.py
├── tests/                      # Unit tests
├── examples/                   # Example scripts
├── docs/                       # Documentation
├── data/                       # Data directory
├── models/                     # Saved models
├── output/                     # Output files
├── config.yaml                 # Configuration file
├── requirements.txt            # Dependencies
├── pyproject.toml             # Package configuration
└── README.md                  # This file
```

## 🧪 Testing

```bash
# Run all tests
pytest

# Run with coverage
pytest --cov=genelab --cov-report=html

# Run specific test file
pytest tests/test_sequence_analyzer.py
```

## 🤝 Contributing

Contributions are welcome! Please follow these steps:

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add some amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

### Development Setup

```bash
# Install development dependencies
pip install -e ".[dev]"

# Run code formatter
black genelab/

# Run linter
flake8 genelab/

# Run type checker
mypy genelab/
```

## 📊 Visualization

GeneLab includes visualization capabilities for:
- Sequence similarity matrices
- Mutation patterns
- Training curves
- 3D PCA of encoded sequences
- Protein structure visualization

## 🔬 Use Cases

- **Genomic Research**: Analyze genetic variations and mutations
- **Drug Discovery**: Predict treatment success from genetic markers
- **Evolutionary Biology**: Simulate and study genetic evolution
- **Biomedical Imaging**: Segment biological structures
- **Bioinformatics Education**: Teaching and learning tools

## 📝 Citation

If you use GeneLab in your research, please cite:

```bibtex
@software{genelab2024,
  title={GeneLab: A Comprehensive Bioinformatics Toolkit},
  author={GeneLab Team},
  year={2024},
  url={https://github.com/Heisnotanimposter/GeneLab}
}
```

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgments

- BioPython for sequence handling
- TensorFlow/Keras for deep learning
- Stable-Baselines3 for reinforcement learning
- The open-source bioinformatics community

## 📧 Contact

For questions, issues, or suggestions:
- GitHub Issues: [https://github.com/Heisnotanimposter/GeneLab/issues](https://github.com/Heisnotanimposter/GeneLab/issues)
- Email: contact@genelab.dev

---

**Made with ❤️ by the GeneLab Team**