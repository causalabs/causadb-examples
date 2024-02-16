import causadb
from dotenv import load_dotenv
import os

# Load the environment variables from .env
load_dotenv()
CAUSADB_TOKEN = os.getenv("CAUSADB_TOKEN")

if __name__ == "__main__":
    # Create a CausaDB client
    print("Setting up CausaDB client...")
    client = causadb.CausaDB()
    client.set_token("examples-token", CAUSADB_TOKEN)

    # Create a model
    print("Creating a model...")
    model = client.create_model("example-smart-building-causal-model")

    # Configure the model
    print("Configuring the model...")
    model.set_nodes(["hvac", "energy", "indoor_temp"])
    model.set_edges([
        ("hvac", "energy"),
        ("hvac", "indoor_temp"),
        ("indoor_temp", "energy")
    ])

    # Upload data
    print("Uploading data...")
    client \
        .add_data("example-smart-building-data") \
        .from_csv("data.csv")
    model.attach("example-smart-building-data")

    # Train the model
    print("Training the model...")
    model.train()

    # Print the status of the model
    print("Model status:", model.status())

    # Now create an equivalent non-causal model, where the DAG is a star graph with the outcome, "indoor_temp" as the center node.
    print("Creating a non-causal model...")
    non_causal_model = client.create_model(
        "example-smart-building-non-causal-model")
    non_causal_model.set_nodes(["hvac", "energy", "indoor_temp"])
    non_causal_model.set_edges([
        ("hvac", "indoor_temp"),
        ("energy", "indoor_temp")
    ])

    # Re-use the same data
    non_causal_model.attach("example-smart-building-data")

    # Train the model
    print("Training the model...")
    non_causal_model.train()

    # Print the status of the model
    print("Model status:", non_causal_model.status())
