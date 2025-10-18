# Migration Guide: GeneLab v1.0 to v2.0

This guide will help you migrate from the old GeneLab structure to the new, modernized version.

## Major Changes

### 1. Project Structure

**Old Structure:**
```
GeneLab/
├── genemlnotebook.py
├── RL.py
├── UNETsample.py
└── Biopython/
```

**New Structure:**
```
GeneLab/
├── genelab/              # Main package
│   ├── core/            # Core functionality
│   ├── models/          # ML models
│   ├── utils/           # Utilities
│   └── config/          # Configuration
├── tests/               # Unit tests
├── examples/            # Example scripts
├── app.py              # Streamlit web app
├── config.yaml         # Configuration file
└── requirements.txt    # Dependencies
```

### 2. Import Changes

**Old way:**
```python
# Everything in one file
from genemlnotebook import *
```

**New way:**
```python
# Modular imports
from genelab import GeneticDataFetcher, SequenceAnalyzer, DNAParser
from genelab.models import TreatmentPredictor, UNetModel
from genelab.utils import setup_logging, generate_training_data
```

### 3. Configuration

**Old way:**
```python
# Hardcoded values
email = "your.email@example.com"
accession = "NC_000001"
mutation_rate = 0.01
```

**New way:**
```python
# Using configuration file
from genelab.config import get_config

config = get_config()
email = config.get("entrez.email")
mutation_rate = config.get("mutation.base_rate")
```

### 4. Logging

**Old way:**
```python
print("Loading data...")
print(f"Loaded {len(data)} sequences")
```

**New way:**
```python
from genelab.utils import setup_logging
import logging

setup_logging(level="INFO")
logger = logging.getLogger(__name__)

logger.info("Loading data...")
logger.info(f"Loaded {len(data)} sequences")
```

### 5. Error Handling

**Old way:**
```python
try:
    sequence = fetcher.fetch_sequence(accession)
except Exception as e:
    print(f"Error: {e}")
```

**New way:**
```python
from genelab import GeneticDataFetcher

fetcher = GeneticDataFetcher(email="your.email@example.com")
sequence = fetcher.fetch_sequence(accession)

if sequence is None:
    logger.error(f"Failed to fetch sequence {accession}")
else:
    logger.info(f"Successfully fetched sequence of length {len(sequence)}")
```

### 6. Model Training

**Old way:**
```python
# Everything in one monolithic class
app = MainApplication(email, accession, positions, bases, data_path)
app.run()
```

**New way:**
```python
from genelab import TreatmentPredictor
from genelab.utils import generate_training_data, vectorize_features
from sklearn.model_selection import train_test_split

# Prepare data
sequences = df['sequence'].values
labels = df['label'].values

# Generate features
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
```

### 7. Google Colab Specific Code

**Old way:**
```python
from google.colab import drive
drive.mount('/content/drive')
data_dir = '/content/drive/MyDrive/GeneLab/DNAsequential/'
```

**New way:**
```python
# No Colab-specific code needed
# Use local paths or configure in config.yaml
from genelab.config import get_config

config = get_config()
data_dir = config.get("data.sequences_path")
```

## Step-by-Step Migration

### Step 1: Install New Dependencies

```bash
pip install -r requirements.txt
pip install -e .
```

### Step 2: Update Configuration

Create a `config.yaml` file with your settings:

```yaml
entrez:
  email: "your.email@example.com"
  
data:
  sequences_path: "./data/sequences"
  models_path: "./models"
```

### Step 3: Refactor Your Code

1. Replace monolithic imports with modular imports
2. Add proper logging
3. Use configuration file instead of hardcoded values
4. Update error handling

### Step 4: Test Your Code

```bash
pytest tests/
```

### Step 5: Run Examples

```bash
python examples/basic_usage.py
python examples/ml_training.py
```

## Common Issues and Solutions

### Issue 1: Import Errors

**Problem:** `ModuleNotFoundError: No module named 'genelab'`

**Solution:**
```bash
pip install -e .
```

### Issue 2: Configuration Not Found

**Problem:** `FileNotFoundError: config.yaml`

**Solution:** Create `config.yaml` file or use default configuration:
```python
from genelab.config import ConfigLoader
config = ConfigLoader()  # Uses defaults
```

### Issue 3: Colab-Specific Code

**Problem:** `ModuleNotFoundError: No module named 'google.colab'`

**Solution:** Remove Colab-specific imports and use local paths or configuration.

### Issue 4: Old Function Names

**Problem:** Functions have been renamed or moved

**Solution:** Check the new API documentation and update function calls.

## Benefits of v2.0

1. **Modular Design**: Easier to maintain and extend
2. **Better Testing**: Comprehensive test suite
3. **Configuration Management**: Centralized settings
4. **Proper Logging**: Better debugging and monitoring
5. **Documentation**: Comprehensive documentation and examples
6. **Web Interface**: Streamlit app for easy access
7. **CI/CD**: Automated testing and deployment
8. **Type Hints**: Better IDE support and code quality

## Getting Help

If you encounter issues during migration:

1. Check the [README.md](README.md) for documentation
2. Look at the [examples/](examples/) directory for usage examples
3. Review the [tests/](tests/) directory for test cases
4. Open an issue on GitHub

## Backward Compatibility

Some old code may still work, but it's recommended to migrate to the new structure for:
- Better performance
- Improved maintainability
- Access to new features
- Long-term support

## Timeline

- **v1.0**: Original monolithic structure (deprecated)
- **v2.0**: Modern modular structure (current)

## Questions?

Contact us:
- GitHub Issues: https://github.com/Heisnotanimposter/GeneLab/issues
- Email: contact@genelab.dev
