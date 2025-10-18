# Contributing to GeneLab

Thank you for your interest in contributing to GeneLab! This document provides guidelines and instructions for contributing.

## Code of Conduct

- Be respectful and inclusive
- Welcome newcomers and help them learn
- Focus on what is best for the community
- Show empathy towards other community members

## How to Contribute

### Reporting Bugs

Before creating bug reports, please check the issue list as you might find out that you don't need to create one. When creating a bug report, please include:

1. **Clear title and description**
2. **Steps to reproduce**: Detailed steps to reproduce the issue
3. **Expected behavior**: What you expected to happen
4. **Actual behavior**: What actually happened
5. **Environment**: Python version, OS, package versions
6. **Minimal example**: Small code snippet that reproduces the issue

### Suggesting Enhancements

Enhancement suggestions are tracked as GitHub issues. When creating an enhancement suggestion, please include:

1. **Clear title and description**
2. **Use case**: Why this enhancement would be useful
3. **Proposed solution**: How you envision this working
4. **Alternatives**: Other solutions you've considered

### Pull Requests

1. **Fork the repository**
2. **Create a feature branch**: `git checkout -b feature/amazing-feature`
3. **Make your changes**
4. **Add tests**: Ensure new code is covered by tests
5. **Update documentation**: Update README, docstrings, etc.
6. **Run tests**: `pytest tests/`
7. **Format code**: `black genelab/`
8. **Commit changes**: `git commit -m 'Add amazing feature'`
9. **Push to branch**: `git push origin feature/amazing-feature`
10. **Open Pull Request**

## Development Setup

### Prerequisites

- Python 3.9 or higher
- Git
- pip or conda

### Setup

```bash
# Clone the repository
git clone https://github.com/Heisnotanimposter/GeneLab.git
cd GeneLab

# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install development dependencies
pip install -r requirements.txt
pip install -e ".[dev]"

# Install pre-commit hooks (optional)
pip install pre-commit
pre-commit install
```

### Running Tests

```bash
# Run all tests
pytest

# Run with coverage
pytest --cov=genelab --cov-report=html

# Run specific test file
pytest tests/test_sequence_analyzer.py

# Run with verbose output
pytest -v
```

### Code Quality

```bash
# Format code with Black
black genelab/ tests/ examples/

# Check with Flake8
flake8 genelab/ --max-line-length=100

# Type checking with mypy
mypy genelab/
```

## Coding Standards

### Python Style Guide

We follow [PEP 8](https://pep8.org/) with some modifications:

- **Line length**: 100 characters (not 79)
- **Docstrings**: Google style
- **Type hints**: Use type hints for function signatures

### Code Formatting

We use [Black](https://github.com/psf/black) for code formatting:

```bash
black genelab/
```

### Import Organization

```python
# Standard library imports
import os
import sys
from typing import List, Dict

# Third-party imports
import numpy as np
import pandas as pd

# Local imports
from genelab.core import SequenceAnalyzer
from genelab.utils import setup_logging
```

### Docstrings

Use Google-style docstrings:

```python
def my_function(param1: str, param2: int) -> bool:
    """
    Brief description of the function.
    
    More detailed description if needed.
    
    Args:
        param1: Description of param1
        param2: Description of param2
    
    Returns:
        Description of return value
    
    Raises:
        ValueError: When param2 is negative
    
    Example:
        >>> my_function("test", 5)
        True
    """
    pass
```

### Type Hints

Always use type hints for function signatures:

```python
from typing import List, Optional, Tuple

def process_sequences(
    sequences: List[str],
    k: int = 6
) -> Tuple[np.ndarray, np.ndarray]:
    """Process sequences and return encoded arrays."""
    pass
```

## Testing Guidelines

### Writing Tests

- Write tests for all new features
- Aim for 80%+ code coverage
- Use descriptive test names
- Test both success and failure cases
- Mock external dependencies

### Test Structure

```python
import pytest
from genelab.core import SequenceAnalyzer

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
```

## Documentation

### Updating Documentation

- Update README.md for major changes
- Add docstrings to all functions and classes
- Update examples if API changes
- Keep CHANGELOG.md up to date

### Documentation Style

- Use clear, concise language
- Include code examples
- Add comments for complex logic
- Keep documentation up to date with code

## Commit Messages

Follow conventional commit format:

```
type(scope): subject

body (optional)

footer (optional)
```

Types:
- `feat`: New feature
- `fix`: Bug fix
- `docs`: Documentation changes
- `style`: Code style changes (formatting, etc.)
- `refactor`: Code refactoring
- `test`: Adding or updating tests
- `chore`: Maintenance tasks

Examples:
```
feat(core): add sequence validation function

fix(models): correct prediction threshold bug

docs: update README with new examples
```

## Review Process

1. All PRs require at least one approval
2. CI/CD checks must pass
3. Code must be reviewed and approved
4. Tests must pass with 80%+ coverage
5. Documentation must be updated

## Getting Help

- GitHub Issues: https://github.com/Heisnotanimposter/GeneLab/issues
- Email: contact@genelab.dev
- Discussions: GitHub Discussions

## Recognition

Contributors will be recognized in:
- CONTRIBUTORS.md
- Release notes
- Project documentation

Thank you for contributing to GeneLab! 🧬
