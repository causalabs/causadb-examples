import os
from simulate import simulate_hvac
import pandas as pd
import matplotlib.pyplot as plt
from dotenv import load_dotenv, find_dotenv
from causadb import CausaDB
import streamlit as st
import time
import plotly.graph_objects as go

st.set_page_config(
    page_title="CausaDB Smart Building Optimisation",
    page_icon=":thermometer:"
)

# Load the environment variables from .env
load_dotenv(find_dotenv())
CAUSADB_TOKEN = os.getenv("CAUSADB_TOKEN")


@st.cache_resource
def load_resources():
    client = CausaDB()
    client.set_token("test-token-id", CAUSADB_TOKEN)
    causal_model = client.get_model("example-smart-building-causal-model")
    non_causal_model = client.get_model(
        "example-smart-building-non-causal-model")
    filepath = os.path.join(os.path.dirname(__file__), "data.csv")
    df = pd.read_csv(filepath)
    return client, causal_model, non_causal_model, df


client, causal_model, non_causal_model, df = load_resources()

# Set title of the app
st.title("Smart Building Optimisation with CausaDB")

st.markdown("""
This example demonstrates the use of causal AI in optimising the HVAC (Heating, Ventilation, and Air Conditioning) settings in a smart building. In large spaces, pre-heating and pre-cooling can be used to maintain a comfortable indoor temperature while minimising energy consumption. This is challenging because effective control of HVAC settings requires accurate estimates of the future indoor temperature.

Standard AI methods for controlling HVAC settings rely on correlations in historical data. These methods fail to make optimal decisions when the underlying data distribution changes, such as when the HVAC settings are changed. On the other hand, causal AI algorithms capture the cause-effect relationships between variables, which allows it to make accurate predictions and optimal decisions even under changing conditions. 
""")

# Expander with an image of the DAG for the causal model
with st.expander("Building a Causal Model with CausaDB"):
    st.markdown("""
    CausaDB is designed to make it easy to build and train causal models. An important part of this process is to define a **causal graph** - a graph that captures the cause-effect relationships between variables in the system. Causal graphs and associated code can be easily built in the CausaDB model builder, shown below.
                
    Check out the full training code in the [Github repo](https://github.com/causalabs/causadb-examples/blob/main/python/smart_building/train.py).
    """)
    st.image(os.path.join(os.path.dirname(__file__), "images/model_builder.png"))

st.markdown("""
In this example, we compare the performance of a causal AI model built with CausaDB against a standard AI model in controlling the HVAC settings to maintain a target indoor temperature.
            
Use the slider below to set a target indoor temperature, and see how the two models perform in achieving the target temperature for simulations over a year.
""")

# Add a slider to control the HVAC setting.
target_temp = st.slider(
    "Target Building Temperature (\u00b0C)", 16.0, 20.0, 18.0, 0.1)

causal_model_hvac = causal_model.optimal_actions(
    {"indoor_temp": target_temp}, ["hvac"])["hvac"]
non_causal_model_hvac = non_causal_model.optimal_actions(
    {"indoor_temp": target_temp}, ["hvac"])["hvac"]


causal_model_temp_expected = causal_model.simulate_actions(
    {"hvac": causal_model_hvac})["do"]["indoor_temp"]
non_causal_model_temp_expected = non_causal_model.simulate_actions(
    {"hvac": non_causal_model_hvac})["do"]["indoor_temp"]

temps_causal = simulate_hvac(df, causal_model_hvac)[
    "indoor_temp"]
temps_non_causal = simulate_hvac(df, non_causal_model_hvac)[
    "indoor_temp"]

# '#15A07B',

# Plot the expected and achieved temperatures over the first 365 days
colors = ['#15C7B8',  '#B13CA0']
background_color = '#1b1917'  # Replace with your desired background color

fig = go.Figure()

# Create line plots
fig.add_trace(go.Scatter(
    y=temps_causal[:365], mode='lines', name='Causal Model', line_color=colors[0]))
fig.add_trace(go.Scatter(
    y=temps_non_causal[:365], mode='lines', name='Standard AI', line_color=colors[1]))

# Add target temperature line
fig.add_shape(type='line',
              x0=0, y0=target_temp, x1=365, y1=target_temp,
              line=dict(color='White', width=2, dash='dash'))

# Add mean temperature line for each model
fig.add_shape(type='line',
              x0=0, y0=temps_causal.mean(), x1=365, y1=temps_causal.mean(),
              line=dict(color=colors[0], width=2, dash='dash'))

fig.add_shape(type='line',
              x0=0, y0=temps_non_causal.mean(), x1=365, y1=temps_non_causal.mean(),
              line=dict(color=colors[1], width=2, dash='dash'))

# Update axes properties
fig.update_layout(
    # width=800,
    # height=800,
    template='plotly_dark',
    paper_bgcolor=background_color,  # Background color for the whole figure
    plot_bgcolor=background_color,   # Background color for the plotting area
    xaxis_title="Days",  # X-axis label
    yaxis_title="Achieved Temperature (\u00b0C)",  # Y-axis label
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

st.markdown("""
These are the optimal HVAC settings the two models believe will achieve the target indoor temperature. These are the actionable outputs that the algorithms provide to control the HVAC settings.
""")

st.table({
    "Model": ["Causal Model", "Standard AI"],
    "HVAC Setting": [round(causal_model_hvac), round(non_causal_model_hvac)]
})


# Display the expected and achieved temperatures in a table
st.subheader("Expected and Achieved Temperatures")

st.markdown("""
These are the average expected and achieved indoor temperatures over a year in the simulated dataset. The expected temperature is the average indoor temperature that the models predict will be achieved by using the optimal HVAC settings. The achieved temperature is the average indoor temperature that is actually achieved by using the optimal HVAC settings.
""")

st.table({
    "Model": ["Causal Model", "Standard AI"],
    "Expected Avg. Temperature": [causal_model_temp_expected, non_causal_model_temp_expected],
    "Achieved Avg. Temperature": [temps_causal.mean(), temps_non_causal.mean()],
    "Deviation": [abs(temps_causal.mean() - target_temp), abs(temps_non_causal.mean() - target_temp)]
})

# Display the annual wastage in a table
st.subheader("Estimated Annual Wastage (£)")

st.markdown("""
Using the size of the building in SQM, you can compare how much energy is wasted from the two algorithms over the course of a year.
""")

# Let users specify the SQM size of their building
building_size = st.number_input(
    "Enter the size of your building in square meters", min_value=100, value=1000, step=100)

# Run a quick cost model to see how much is wasted per year and what that will cost.
# Assume cost per square meter per degree difference from target temperature
cost_per_sqm_per_degree = 0.4  # This is a placeholder, replace with actual cost

total_deviation_causal = abs(temps_causal.mean() - target_temp)
total_deviation_non_causal = abs(temps_non_causal.mean() - target_temp)

causal_model_cost = (total_deviation_causal *
                     building_size * cost_per_sqm_per_degree) * 365
non_causal_model_cost = (total_deviation_non_causal *
                         building_size * cost_per_sqm_per_degree) * 365

# Create bar chart
fig2 = go.Figure()

fig2.add_trace(go.Bar(
    x=["Causal AI", "Standard AI"],
    y=[causal_model_cost, non_causal_model_cost],
    marker_color=colors
))

# Update axes properties
fig2.update_layout(
    template='plotly_dark',
    paper_bgcolor=background_color,  # Background color for the whole figure
    plot_bgcolor=background_color,   # Background color for the plotting area
    xaxis_title="Model",  # X-axis label
    yaxis_title="Annual Wastage (£)",  # Y-axis label
    font=dict(family="DM Sans", size=22, color="white"),  # Set the font here
    xaxis=dict(showgrid=False),  # Increase X-axis label size
    yaxis=dict(showgrid=True),   # Increase Y-axis label size
)

# Show plot
st.plotly_chart(fig2)

st.table({
    "Model": ["Causal AI", "Standard AI", "CausaDB Savings"],
    "Annual Wastage": [f'£{causal_model_cost:,.2f}', f'£{non_causal_model_cost:,.2f}', f'£{(non_causal_model_cost - causal_model_cost):,.2f}']
})

st.subheader("Conclusion")

st.markdown("""
In this example we've demonstrated how a causal model built with CausaDB vastly outperforms an equivalent standard AI model for controlling building temperature. Using causal AI is the only way to avoid costly mistakes with standard AI, and to build truly trustworthy and effective AI models. If you'd like to learn more about CausaDB, visit [causa.tech](https://causa.tech).

Check out the [Github repo](https://github.com/causalabs/causadb-examples/blob/main/python/smart_building/README.md) to start building your own causal AI models with CausaDB.            
""")
