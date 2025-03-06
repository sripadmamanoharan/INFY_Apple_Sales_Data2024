%%writefile app.py
import streamlit as st
import pandas as pd
import plotly.express as px
import google.generativeai as genai
import os

# Set up Google Gemini API Key
os.environ["GOOGLE_API_KEY"] = "your-gemini-api-key"  # Replace with your actual key
genai.configure(api_key=os.environ["GOOGLE_API_KEY"])

# AI-Powered Insight Agent
class InsightAgent:
    def __init__(self):
        self.model = genai.GenerativeModel("gemini-pro")

    def generate_insights(self, kpis):
        prompt = f"""
        Based on the following sales KPIs: {kpis}.
        - Identify key trends.
        - Highlight strong and weak performing products and regions.
        - Suggest strategies to optimize performance.
        """
        response = self.model.generate_content(prompt)
        return response.text

# Initialize AI agent
insight_agent = InsightAgent()

# Streamlit App
st.title("🚀 AI-Powered Sales Performance Dashboard")

# File Upload for Dynamic Data
uploaded_file = st.file_uploader("📂 Upload Sales Data (CSV or Excel)", type=["csv", "xlsx"])
if uploaded_file:
    if uploaded_file.name.endswith(".csv"):
        df = pd.read_csv(uploaded_file)
    else:
        df = pd.read_excel(uploaded_file)

    # Clean column names
    df.columns = df.columns.str.strip()

    # Display the dataset
    st.write("📊 Preview of Sales Data")
    st.dataframe(df.head())

    # KPI Calculations
    kpis = {
        "Total iPhone Sales": df["iPhone Sales (in million units)"].sum(),
        "Total iPad Sales": df["iPad Sales (in million units)"].sum(),
        "Total Mac Sales": df["Mac Sales (in million units)"].sum(),
        "Total Wearables Sales": df["Wearables (in million units)"].sum(),
        "Total Services Revenue": df["Services Revenue (in billion $)"].sum(),
    }

    st.write("### 📊 Key Performance Indicators")
    st.json(kpis)

    # Product-wise Sales Visualization
    fig_product = px.bar(
        pd.DataFrame(kpis.items(), columns=["Product", "Total Sales"]),
        x="Product", y="Total Sales", color="Product", title="📈 Sales Performance by Product"
    )
    st.plotly_chart(fig_product)

    # Country-wise Sales Analysis
    if "Region" in df.columns and "iPhone Sales (in million units)" in df.columns:
        country_sales = df.groupby("Region")["iPhone Sales (in million units)"].sum().reset_index()
        fig_region = px.bar(
            country_sales, x="Region", y="iPhone Sales (in million units)", color="Region",
            title="📈 iPhone Sales by Country"
        )
        st.plotly_chart(fig_region)

    # Generate AI Insights
    insights = insight_agent.generate_insights(kpis)
    st.write("### 🤖 AI-Generated Insights")
    st.write(insights)
