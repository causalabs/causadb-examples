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
    model = client.create_model("example-smart-building-model")

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
        .from_csv("smart_building/data.csv")
    model.attach("example-smart-building-data")

    # Train the model
    print("Training the model...")
    model.train()

    # Print the status of the model
    print("Model status:", model.status())
