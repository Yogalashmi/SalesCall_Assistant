from collections import Counter
import plotly.graph_objects as go
import matplotlib.pyplot as plt
import streamlit as st
import pandas as pd
import plotly.express as px

def create_line_chart(data_df):
    fig_line = go.Figure()
    fig_line.add_trace(
        go.Scatter(
            x=data_df["Timestamp"],
            y=data_df["Score"],
            mode="lines+markers",
            marker=dict(
                size=10,
                color=data_df["Score"],
                colorscale="Viridis",  # Cividis
                showscale=True,
                colorbar=dict(title="Sentiment Score")
            ),
            line=dict(color="lightgray"),
            text=data_df["Sentiment"],
            hovertemplate="Sentiment: %{text}<br>Score: %{y}<br>Time: %{x}"
        )
    )
    fig_line.update_layout(
        title="Sentiment Score Over Time",
        xaxis_title="Time",
        yaxis_title="Sentiment Score",
        template="plotly_dark"
    )
    return fig_line

# Function to create the sentiment distribution pie chart
def create_pie_chart(data_df):
    sentiment_counts = data_df["Sentiment"].value_counts()
    fig_pie = px.pie(
        names=sentiment_counts.index,
        values=sentiment_counts,
        title="Sentiment Distribution",
        color_discrete_sequence=["#66c2a5", "#fc8d62", "#8da0cb"]
    )
    return fig_pie

def create_bar_chart(categories):
    category_counts = dict(Counter(categories))
    fig_bar = go.Figure(data=[go.Bar(x=list(category_counts.keys()), y=list(category_counts.values()))])
    fig_bar.update_layout(
        title="Objections Raised",
        xaxis_title="Objection Type",
        yaxis_title="Count"
    )
    st.plotly_chart(fig_bar)
