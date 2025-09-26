# Contributing to MACE Model Deviation

## Development Setup

```bash
git clone https://github.com/lianxiliang/mace-model-devi.git
cd mace-model-devi
pip install -e ".[dev]"
```

## Code Style

We use:
- Black for code formatting
- isort for import sorting
- mypy for type checking

```bash
# Format code
black src/
isort src/

# Type checking
mypy src/
```

## Testing

```bash
# Test with sample data
mace-model-devi --models model1.model model2.model --traj trajectory.xyz --output test.out

# Test Python API
python -c "from mace_model_deviation import calculate_mace_model_deviation; print('Import successful')"

# Run unit tests (if available)
pytest tests/
```

## Integration with AI2Kit

The package is designed to integrate seamlessly with AI2Kit MACE workflows:

```python
# In ai2kit macelmp.py
mace_cmd = f'mace-model-devi --models "{models}" --traj traj.lammpstrj --output model_devi.out --device cuda'
```

## Release Process

1. Update version in `pyproject.toml`  
2. Update `__version__` in `src/mace_model_deviation/__init__.py`
3. Create git tag: `git tag v0.1.0`
4. Push tag: `git push origin v0.1.0`

## Guidelines

- Follow MACE's official evaluation patterns from `eval_configs.py`
- Maintain DeepMD output format compatibility
- Write comprehensive docstrings
- Test with real AI2Kit models when possible