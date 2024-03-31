import os
from simulate import set_heating, calculate_wasted_heating_cost
import pandas as pd
import matplotlib.pyplot as plt
from dotenv import load_dotenv, find_dotenv
from causadb import CausaDB
import streamlit as st
import time
import plotly.graph_objects as go
import numpy as np
import xgboost as xgb

# Plot the expected and achieved temperatures over the first 365 days
colors = ['#15C7B8',  '#B13CA0', '#D3D3D3']
background_color = '#1b1917'  # Replace with your desired background color

st.set_page_config(
    page_title="CausaDB Smart Building Optimisation",
    page_icon=":thermometer:"
)

# Load the environment variables from .env
load_dotenv(find_dotenv())
CAUSADB_TOKEN = os.getenv("CAUSADB_TOKEN")

st.image(os.path.join(os.path.dirname(
    __file__), "images/logo.svg"), width=200)


@st.cache_resource
def load_resources():
    client = CausaDB(CAUSADB_TOKEN)
    causal_model = client.get_model("example-heating-model")
    filepath = os.path.join(os.path.dirname(__file__),
                            "example_heating_data.csv")
    df = pd.read_csv(filepath)

    X = df[['outdoor_temp', 'indoor_temp']].values
    y = df['heating'].values

    xgb_model = xgb.XGBRegressor()
    xgb_model.fit(X, y)

    return client, causal_model, xgb_model, df


client, causal_model, xgb_model, df = load_resources()

# Set title of the app
st.title("Smart Building Energy Optimisation with CausaDB")

st.markdown("""
This app accompanies the notebook [Smart Building Energy Optimisation with CausaDB](https://github.com/causalabs/causadb-examples/blob/main/python/smart_building/notebook.ipynb) and demonstrates how a deployed causal model can be used in production.
            
This app is an example of an end-user application that makes recommendations to technicians on how to set the heating in a building to maintain a desired target temperature on a given day. The app also estimates the cost of wasted heating energy for different strategies.
""")

st.subheader("Heating Control")

# Add a slider to control the HVAC setting.
target_temp = st.slider(
    "Target Building Temperature (\u00b0C)", 16.0, 20.0, 18.0, 0.1)

# st.markdown(f"Target temperature: {target_temp} \u00b0C")

outdoor_temp = st.slider(
    "Outdoor Temperature (\u00b0C)", 10.0, 20.0, 15.0, 0.1)

# st.markdown(f"Outdoor temperature: {outdoor_temp} \u00b0C")

best_actions_causal = causal_model.find_best_actions(
    {"indoor_temp": target_temp},
    actionable=["heating"],
    fixed={"outdoor_temp": outdoor_temp}
)["heating"][0]

best_actions_xgb = xgb_model.predict(
    np.array([[outdoor_temp, target_temp]]))[0]

achieved_temp_causal = set_heating(
    best_actions_causal, np.repeat(outdoor_temp, 1e4), noise=True)[0]

achieved_temp_xgb = set_heating(
    best_actions_xgb, np.repeat(outdoor_temp, 1e4), noise=True)[0]

st.markdown(f"<h2 style='text-align: center; color: white;'>Target: {target_temp} \u00b0C</h2>",
            unsafe_allow_html=True)

# Show target temperature, HVAC setting, and achieved temperature in a table
st.subheader("Outcomes")

# Percentage deviation from target temperature
deviation_causal = abs(achieved_temp_causal.mean() -
                       target_temp)
deviation_xgb = abs(achieved_temp_xgb.mean() - target_temp)

st.table({
    "Model": ["Standard AI", "CausaDB"],
    "HVAC Setting": [round(best_actions_xgb), round(best_actions_causal)],
    "Achieved Avg. Temperature": [f'{achieved_temp_xgb.mean():.2f}°C', f'{achieved_temp_causal.mean():.2f}°C'],
    "Deviation": [f'{deviation_xgb:.2f}°C', f'{deviation_causal:.2f}°C']
})

st.markdown(f"<h2 style='text-align: center; color: white;'>Standard: {achieved_temp_xgb.mean():.1f} \u00b0C | CausaDB: {achieved_temp_causal.mean():.1f} \u00b0C</h2>",
            unsafe_allow_html=True)

fig = go.Figure()

# Create histogram plots
fig.add_trace(go.Histogram(
    x=achieved_temp_xgb, name='Standard AI', marker_color=colors[1]))
fig.add_trace(go.Histogram(
    x=achieved_temp_causal, name='CausaDB', marker_color=colors[0]))

fig.add_shape(type='line',
              x0=target_temp, y0=0, x1=target_temp, y1=500,
              line=dict(color='White', width=2, dash='dash'))

# Update layout for the histogram
fig.update_layout(
    barmode='overlay',  # Overlay both histograms
    template='plotly_dark',
    paper_bgcolor=background_color,  # Background color for the whole figure
    plot_bgcolor=background_color,   # Background color for the plotting area
    xaxis_title="Achieved Temperature (\u00b0C)",  # X-axis label
    yaxis_title="Frequency",  # Y-axis label
    title="Simulated Achieved Temperatures Over 1000 Repeats",  # Set the title here
    font=dict(family="Helvetica", size=22, color="white"),  # Set the font here
    xaxis=dict(showgrid=False),  # Increase X-axis label size
    yaxis=dict(showgrid=True),   # Increase Y-axis label size
)

# Reduce opacity to see both histograms
fig.update_traces(opacity=0.75)

# Show plot
st.plotly_chart(fig)


st.subheader("Energy Savings")


cost_original, cost_original_daily, power_original_daily = calculate_wasted_heating_cost(
    df["indoor_temp"], target_temp)
cost_xgb, cost_xgb_daily, power_xgb_daily = calculate_wasted_heating_cost(
    achieved_temp_xgb, target_temp)
cost_causadb, cost_causadb_daily, power_causadb_daily = calculate_wasted_heating_cost(
    achieved_temp_causal, target_temp)

# Create a bar plot to visualize the costs
fig = go.Figure(data=[
    go.Bar(name='Human', x=['Original', 'XGB', 'CausaDB'], y=[
           cost_original, cost_xgb, cost_causadb], marker_color=[colors[2], colors[0], colors[1]]),
])

# Change the bar mode
fig.update_layout(barmode='group')
fig.update_layout(title_text='Wasted Heating Costs')

# Show plot
st.plotly_chart(fig)

st.subheader("Conclusion")

st.markdown("""
In this example we've demonstrated how a causal model built with CausaDB vastly outperforms an equivalent standard AI model for controlling building temperature. Using causal AI is the only way to avoid costly mistakes with standard AI, and to build truly trustworthy and effective AI models. If you'd like to learn more about CausaDB, visit [causa.tech](https://causa.tech).

Check out the [Github repo for this example](https://github.com/causalabs/causadb-examples/blob/main/python/smart_building/README.md) to start building your own causal AI models with CausaDB.
""")


# Add copyright footer
st.markdown("""
© 2024 Causa Ltd. All rights reserved.
""")
