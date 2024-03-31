# CausaDB Example - Python - Smart Building Heating Control

This example demonstrates how CausaDB can be used as part of a smart building control system. This example is written in Python and uses the CausaDB Python client.

## Files

- `simulate.py`: Simulates a smart building control system response over the course of a year, and writes the data to a CSV file.
- `notebook.ipynb`: Demonstrates how standard AI fails on the control problem, and trains a causal model with CausaDB for the problem. Compares the performance of the two models.
- `app.py`: A Streamlit app that uses the trained model from the notebook to make real-time recommendations for building control.

## Setup

Navigate into the root of the `smart_building` directory and run the following command to install the required dependencies:

```bash
poetry install
```

## Commands

Run simulations to generate data:
```bash
poetry run python simulate.py
```

Run the Streamlit app:
```bash
poetry run streamlit run app.py
```

## Live Demo

A live demo of the Streamlit app can be found at [causadb-examples-smart-building.streamlit.app](https://causadb-examples-smart-building.streamlit.app/).