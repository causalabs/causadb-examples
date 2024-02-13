# Import Streamlit
import streamlit as st
from causadb import CausaDB
from dotenv import load_dotenv
import os
import matplotlib.pyplot as plt
import pandas as pd

# Load the environment variables from .env
load_dotenv()
CAUSADB_TOKEN = os.getenv("CAUSADB_TOKEN")


@st.cache_resource
def load_resources():
    # Cache the model in Streamlit to avoid unnecessary repeated API calls
    client = CausaDB()
    client.set_token("test-token-id", CAUSADB_TOKEN)
    model = client.get_model("example-smart-building-model")
    return client, model


# Load the client and model
client, model = load_resources()

# Set title of the app
st.title("CausaDB Smart Building Example App")

# Add a slider to control the HVAC setting
slider_value = st.slider("HVAC Setting", 0, 100, 50)

# When the slider value changes, simulate the action and display the results
outcome = model.simulate_action({"hvac": [0, slider_value]})
df = pd.DataFrame(outcome)

# Display the results
st.write(df)

# TODO: Show a plot of how optimal HVAC predictions differ between standard AI and Causal AI.
