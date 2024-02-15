import os
from simulate import simulate_hvac
import pandas as pd
import matplotlib.pyplot as plt
from dotenv import load_dotenv
from causadb import CausaDB
import streamlit as st
import time
import plotly.graph_objects as go

# Load the environment variables from .
load_dotenv()
CAUSADB_TOKEN = os.getenv("CAUSADB_TOKEN")


@st.cache_resource
def load_resources():
    client = CausaDB()
    client.set_token("test-token-id", CAUSADB_TOKEN)
    causal_model = client.get_model("example-smart-building-model")
    noncausal_model = client.get_model(
        "example-smart-building-non-causal-model")
    df = pd.read_csv("smart_building/data.csv")
    return client, causal_model, noncausal_model, df


client, causal_model, noncausal_model, df = load_resources()

# Set title of the app
st.title("CausaDB Smart Building Example App")

# Add a slider to control the HVAC setting.
target_temp = st.slider(
    "Target Building Temperature (\u00b0C)", 17.0, 19.0, 18.0, 0.1)

# Let users specify the SQM size of their building
building_size = st.number_input(
    "Enter the size of your building in square meters", min_value=100, value=1000, step=100)

causal_model_hvac = causal_model.optimal_actions(
    {"indoor_temp": target_temp}, ["hvac"])["hvac"]
noncausal_model_hvac = noncausal_model.optimal_actions(
    {"indoor_temp": target_temp}, ["hvac"])["hvac"]


causal_model_temp_expected = causal_model.simulate_actions(
    {"hvac": causal_model_hvac})["do"]["indoor_temp"]
noncausal_model_temp_expected = noncausal_model.simulate_actions(
    {"hvac": noncausal_model_hvac})["do"]["indoor_temp"]

temps_causal = simulate_hvac(df, causal_model_hvac)[
    "indoor_temp"]
temps_noncausal = simulate_hvac(df, noncausal_model_hvac)[
    "indoor_temp"]

# Plot the expected and achieved temperatures over the first 365 days
colors = ['#15A07B', '#15C7B8']
background_color = '#1b1917'  # Replace with your desired background color

fig = go.Figure()

# Create line plots
fig.add_trace(go.Scatter(
    y=temps_causal[:365], mode='lines', name='Causal Model', line_color=colors[0]))
fig.add_trace(go.Scatter(
    y=temps_noncausal[:365], mode='lines', name='Standard AI', line_color=colors[1]))

# Add target temperature line
fig.add_shape(type='line',
              x0=0, y0=target_temp, x1=365, y1=target_temp,
              line=dict(color='White', width=2, dash='dash'))

# Add mean temperature line for each model
fig.add_shape(type='line',
              x0=0, y0=temps_causal.mean(), x1=365, y1=temps_causal.mean(),
              line=dict(color=colors[0], width=2, dash='dash'))

fig.add_shape(type='line',
              x0=0, y0=temps_noncausal.mean(), x1=365, y1=temps_noncausal.mean(),
              line=dict(color=colors[1], width=2, dash='dash'))

# Update axes properties
fig.update_layout(
    # width=800,
    # height=800,
    template='plotly_dark',
    paper_bgcolor=background_color,  # Background color for the whole figure
    plot_bgcolor=background_color,   # Background color for the plotting area
    xaxis_title="Days",  # X-axis label
    yaxis_title="Temperature (\u00b0C)",  # Y-axis label
    font=dict(family="DM Sans", size=22, color="white"),  # Set the font here
    # Increase X-axis label size
    xaxis=dict(showgrid=False),
    # Increase Y-axis label size
    yaxis=dict(showgrid=True),
)

# Show plot
st.plotly_chart(fig)


# Display the optimal HVAC setting in a table
st.subheader("Optimal HVAC Setting")
st.table({
    "Model": ["Causal Model", "Standard AI"],
    "HVAC Setting": [round(causal_model_hvac), round(noncausal_model_hvac)]
})


# Display the expected and achieved temperatures in a table
st.subheader("Expected and Achieved Temperatures")
st.table({
    "Model": ["Causal Model", "Standard AI"],
    "Expected Avg. Temperature": [causal_model_temp_expected, noncausal_model_temp_expected],
    "Achieved Avg. Temperature": [temps_causal.mean(), temps_noncausal.mean()]
})

# Run a quick cost model to see how much is wasted per year and what that will cost.
# Assume cost per square meter per degree difference from target temperature
cost_per_sqm_per_degree = 0.4  # This is a placeholder, replace with actual cost

total_deviation_causal = abs(temps_causal.mean() - target_temp)
total_deviation_noncausal = abs(temps_noncausal.mean() - target_temp)

causal_model_cost = (total_deviation_causal *
                     building_size * cost_per_sqm_per_degree) * 365
noncausal_model_cost = (total_deviation_noncausal *
                        building_size * cost_per_sqm_per_degree) * 365

# Display the annual wastage in a table
st.subheader("Annual Wastage Cost (£)")
st.table({
    "Model": ["Causal Model", "Standard AI"],
    "Annual Cost": [round(causal_model_cost, 2), round(noncausal_model_cost, 2)]
})

# Show the difference in cost between the two models.
cost_difference = noncausal_model_cost - causal_model_cost
st.subheader("Annual Savings with Causal Model")
st.write(f"£{round(cost_difference, 2)}")
