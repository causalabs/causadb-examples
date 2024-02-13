# CausaDB Examples - Python

This directory contains examples of how to use the CausaDB Python client. To run the examples, you will need a token for CausaDB. If you don't already have one, you can request one at <https://causa.tech/>.

## Prerequisites

To keep things simple, all the examples are run in the same Poetry environment. If you don't already have Poetry installed, follow the instructions at <https://python-poetry.org/>. 

All the Python examples will be run from the `python` directory, so make sure you are in the `python` directory before running any of the commands. To install the Poetry dependencies, run:

```bash
poetry install
```

## Running the examples

To give access to the CausaDB server, you will need to set your CausaDB token as an environment variable. An easy way to do this is to copy the `.env.example` file to `.env` and set the `CAUSADB_TOKEN` variable to your token.

```bash
cp .env.example .env
```

Then, open the `.env` file and set the `CAUSADB_TOKEN` variable to your token.

Now you can run the examples with:

```bash
poetry run python examples/01_create_and_query.py
```

## Examples

- [01_create_and_query.py](examples/01_create_and_query.py): Create a new CausaDB database and query it.