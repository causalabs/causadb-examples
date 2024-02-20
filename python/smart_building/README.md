# CausaDB Example - Python - Smart Building Control

This example demonstrates how CausaDB can be used as part of a smart building control system. This example is written in Python and uses the CausaDB Python client.

## Files

- `simulate.py`: Simulates a smart building control system response over several years, and writes the data to a CSV file.
- `train.py`: Trains a causal and equivalent non-causal model on the simulated data using CausaDB.
- `app.py`: A Streamlit app that uses the trained models to optimise HVAC settings. 

## Commands

Run simulations to generate data:
```bash
poetry run python simulate.py
```

Train the model on the generated data:
```bash
poetry run python train.py
```

Run the Streamlit app:
```bash
poetry run streamlit run app.py
```
