# CausaDB Example - Python - Smart Building Control

This example demonstrates how CausaDB can be used as part of a smart building control system. This example is written in Python and uses the CausaDB Python client.

## Commands

Run simulations to generate data:
```bash
poetry run python smart_building/simulate.py
```

Train the model on the generated data:
```bash
poetry run python smart_building/train.py
```

Run the Streamlit app:
``
`poetry run streamlit run smart_building/app.py`
