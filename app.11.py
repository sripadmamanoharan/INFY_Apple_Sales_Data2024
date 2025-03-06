import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import google.generativeai as genai
import os
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.messages import HumanMessage
import numpy as np
from sqlalchemy import create_engine
import sqlite3

# ✅ Set Page Configuration FIRST
st.set_page_config(page_title="AI Sales Dashboard", layout="wide")

# ✅ Securely Load API Key
os.environ["GOOGLE_API_KEY"] = st.secrets["GOOGLE_API_KEY"]
genai.configure(api_key=os.environ["GOOGLE_API_KEY"])

# 🎯 Streamlit UI
st.title("📊 AI-Powered Sales KPI Dashboard")
st.sidebar.header("📂 Upload or Select Data Source")

# ✅ File Upload Section
uploaded_file = st.sidebar.file_uploader("Upload Sales Data", type=["csv", "xlsx"])

# ✅ Load Data Function
@st.cache_data
def load_data(uploaded_file):  
    if uploaded_file is not None:
        file_extension = uploaded_file.name.split(".")[-1]
        try:
            if file_extension == "csv":
                df = pd.read_csv(uploaded_file, nrows=5000)  # Load 5000 rows for optimization
            elif file_extension == "xlsx":
                df = pd.read_excel(uploaded_file, engine="openpyxl")
            else:
                st.error("⚠️ Unsupported file format. Upload CSV or Excel.")
                return None

            # ✅ Ensure Column Names are Cleaned for Consistency
            df.columns = df.columns.str.strip().str.lower().str.replace(r'[^\w]', '', regex=True)
            return df  # Return cleaned DataFrame

        except Exception as e:
            st.error(f"⚠️ Error loading file: {e}")  # Show error message in Streamlit
            return None
    else:
        return None  # Return None if no file is uploaded

# ✅ Load Data Safely (Fixing the TypeError)
df = load_data(uploaded_file) if uploaded_file else None  # ✅ Pass uploaded_file correctly

# ✅ Display Data if Loaded
if df is not None:
    st.write("✅ File uploaded successfully! Preview:")
    st.dataframe(df.head())  # Show first few rows
else:
    st.warning("⚠️ No file uploaded.")

# ✅ Compute Sales Metrics Only If Data is Loaded
if df is not None and not df.empty:
    # ✅ Ensure Key Sales Metrics Exist Safely
    df['actual_sales'] = (
        df.get('iphonesalesinmillionunits', 0) +
        df.get('ipadsalesinmillionunits', 0) +
        df.get('macsalesinmillionunits', 0) +
        df.get('wearablesinmillionunits', 0)
    )
    df['actual_sales'] += df.get('servicesrevenueinbillion', 0) * 1000
    df['sales_target'] = df['actual_sales'] * 0.9
    df['sales_vs_target'] = df['actual_sales'] - df['sales_target']

    st.write("✅ Sales Metrics Computed Successfully!")
else:
    st.warning("⚠️ No valid data found. Please upload a correct CSV or Excel file.")

    # 📌 Select Role
    user_role = st.sidebar.selectbox("Choose Your Role", ["CXO", "Division Head", "Line Manager"])
    st.subheader(f"📈 KPI Metrics for {user_role}")

    if user_role == "CXO":
        col1, col2, col3 = st.columns(3)
        col1.metric("Total Revenue", f"${df['actual_sales'].sum():,.2f}")
        col2.metric("Revenue Growth", f"{df['sales_vs_target'].mean():.2f}%")
        col3.metric("Profit Margin", "18.5%")

    elif user_role == "Division Head" and "region" in df.columns:
        region = st.sidebar.selectbox("Select Region", df["region"].unique())
        df_region = df[df["region"] == region]
        col1, col2 = st.columns(2)
        col1.metric(f"{region} Sales", f"${df_region['actual_sales'].sum():,.2f}")
        col2.metric(f"{region} Sales Growth", f"{df_region['sales_vs_target'].mean():,.2f}%")

    # ✅ AI-Powered Insights
    st.subheader("🔍 AI-Generated Sales Insights")
    llm = ChatGoogleGenerativeAI(model="gemini-1.5-pro", google_api_key=os.getenv("GOOGLE_API_KEY"))

    def generate_ai_insights(role):
        try:
            small_df = df.head(20)  # Only send 20 rows to AI
            prompt = f"Analyze this sales data for {role}: {small_df.to_string(index=False)}"
        
            response = llm.invoke([HumanMessage(content=prompt)])
            return response.content
    
        except Exception as e:
            return f"⚠️ AI Model Error: {e}"  # Handles errors gracefully



    if st.button("🔍 Generate AI Insights"):
        with st.spinner("⏳ Generating AI insights..."):
            ai_insights = generate_ai_insights(user_role)
            st.write(ai_insights)

