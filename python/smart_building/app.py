# Import Streamlit
import time
import streamlit as st
from causadb import CausaDB
from dotenv import load_dotenv
import os
import matplotlib.pyplot as plt
import pandas as pd
from simulate import simulate_hvac

# Load the environment variables from .env
load_dotenv()
CAUSADB_TOKEN = os.getenv("CAUSADB_TOKEN")

print(os.getenv("CAUSADB_URL"))


@st.cache_resource
def load_resources():
    # Cache the model in Streamlit to avoid unnecessary repeated API calls
    client = CausaDB()
    client.set_token("test-token-id", CAUSADB_TOKEN)
    causal_model = client.get_model("example-smart-building-model")
    noncausal_model = client.get_model(
        "example-smart-building-non-causal-model")
    return client, causal_model, noncausal_model


# Load the client and model
client, causal_model, noncausal_model = load_resources()

# Set title of the app
st.title("CausaDB Smart Building Example App")

# Add a slider to control the HVAC setting
slider_value = st.slider("HVAC Setting", 0, 100, 50)

# When the slider value changes, simulate the action and display the results
# causal_outcome = causal_model.simulate_actions({"hvac": slider_value})
# df_causal = pd.DataFrame(causal_outcome)

# noncausal_outcome = noncausal_model.simulate_actions({"hvac": slider_value})
# df_noncausal = pd.DataFrame(noncausal_outcome)


start = time.time()
for i in range(10):
    outcome = causal_model.optimal_actions({"indoor_temp": 20 + i}, ["hvac"])
    print(outcome)
end = time.time()
print("Time taken:", end - start)

outcome = noncausal_model.optimal_actions({"indoor_temp": 20}, ["hvac"])
print(outcome)

# Load df from the data.csv file
df = pd.read_csv("smart_building/data.csv")
st.write(simulate_hvac(df, slider_value))

# Display the results
# st.write(df_causal)
# st.write(df_noncausal)

# TODO: Show a plot of how optimal HVAC predictions differ between standard AI and Causal AI.
