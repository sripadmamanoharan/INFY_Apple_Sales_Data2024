import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import google.generativeai as genai
import os
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.messages import HumanMessage
from sklearn.linear_model import LinearRegression
import numpy as np

# ✅ Set Google Gemini API Key (Replace with your actual API key)
os.environ["GOOGLE_API_KEY"] = "AIzaSyCskTdzvoDekYyxWAAQJfWcFAUAakIPtKo"
genai.configure(api_key=os.environ["GOOGLE_API_KEY"])

# ✅ Load Sales Data
@st.cache_data
def load_data():
    df = pd.read_csv("sales_data.csv")  # Ensure sales_data.csv is in the same folder
    df.columns = df.columns.str.strip().str.lower().str.replace(' ', '_')  # Clean column names
    return df

df = load_data()

# 🎯 Streamlit UI
st.title("📊 AI-Powered Sales KPI Dashboard")
st.sidebar.header("Filter Options")

# 📌 Select Role (CXO, Division Head, Line Manager)
user_role = st.sidebar.selectbox("Choose Your Role", ["CXO", "Division Head", "Line Manager"])

# ✅ KPI Metrics Based on Role
st.subheader(f"📈 KPI Metrics for {user_role}")
if user_role == "CXO":
    col1, col2, col3 = st.columns(3)
    col1.metric("Total Revenue", f"${df['actual_sales'].sum():,.2f}")
    col2.metric("Revenue Growth", f"{df['sales_vs_target'].mean():.2f}%")
    col3.metric("Profit Margin", "18.5%")  # Placeholder

elif user_role == "Division Head":
    region = st.sidebar.selectbox("Select Region", df["region"].unique())
    df_region = df[df["region"] == region]
    col1, col2 = st.columns(2)
    col1.metric(f"{region} Sales", f"${df_region['actual_sales'].sum():,.2f}")
    col2.metric(f"{region} Sales Growth", f"{df_region['sales_vs_target'].mean():.2f}%")

elif user_role == "Line Manager":
    salesperson = st.sidebar.selectbox("Select Salesperson", df["salesperson"].unique())
    df_salesperson = df[df["salesperson"] == salesperson]
    col1, col2 = st.columns(2)
    col1.metric(f"{salesperson} Sales", f"${df_salesperson['actual_sales'].sum():,.2f}")
    col2.metric(f"{salesperson} Target Achievement", f"{df_salesperson['sales_vs_target'].mean():.2f}%")

# ✅ AI-Powered Sales Insights (Google Gemini)
st.subheader("🔍 AI-Generated Sales Insights")
llm = ChatGoogleGenerativeAI(model="gemini-pro", google_api_key=os.getenv("GOOGLE_API_KEY"))

def generate_ai_insights(role):
    prompt = f"Analyze the sales data for the role: {role}.\n{df.to_string(index=False)}"
    response = llm.invoke([HumanMessage(content=prompt)])
    return response.content

if st.button("🔍 Generate AI Insights"):
    ai_insights = generate_ai_insights(user_role)
    st.write(ai_insights)
