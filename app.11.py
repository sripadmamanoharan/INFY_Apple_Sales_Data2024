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

# ✅ Securely Load API Key
os.environ["GOOGLE_API_KEY"] = st.secrets["GOOGLE_API_KEY"]
genai.configure(api_key=os.environ["GOOGLE_API_KEY"])

# 🎯 Streamlit UI
st.title("📊 AI-Powered Sales KPI Dashboard")
st.sidebar.header("📂 Upload or Select Data Source")

# ✅ File Upload Section
uploaded_file = st.sidebar.file_uploader("Upload Sales Data", type=["csv", "xlsx"])

# ✅ Database Connection (SQLite Example)
DATABASE_URL = "sqlite:///sales.db" 

def initialize_database():
    conn = sqlite3.connect("sales.db")
    cursor = conn.cursor()
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS sales_data (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            region TEXT,
            actual_sales REAL,
            sales_target REAL,
            sales_vs_target REAL
        )
    """)
    conn.commit()
    conn.close()

# ✅ Load Data Function (CSV, Excel, or Database)
@st.cache_data
def load_data(uploaded_file):
    if uploaded_file is not None:
        file_extension = uploaded_file.name.split(".")[-1]
        try:
            if file_extension == "csv":
                df = pd.read_csv(uploaded_file, nrows=5000)  # Load first 5000 rows for performance
            elif file_extension == "xlsx":
                df = pd.read_excel(uploaded_file, engine="openpyxl", nrows=5000)
            else:
                st.error("⚠️ Unsupported file format. Upload CSV or Excel.")
                return None
        except Exception as e:
            st.error(f"⚠️ Error loading file: {e}")
            return None
    else:
        st.warning("⚠️ No file uploaded.")
        return None

    # ✅ Clean Column Names for Consistency
    df.columns = df.columns.str.strip().str.lower().str.replace(r'[^\w]', '', regex=True)
    return df

# ✅ Load Data
df = load_data(uploaded_file)

if df is not None and not df.empty:
    # ✅ Compute Key Sales Metrics
    df['actual_sales'] = (
        df.get('iphonesalesinmillionunits', 0) +
        df.get('ipadsalesinmillionunits', 0) +
        df.get('macsalesinmillionunits', 0) +
        df.get('wearablesinmillionunits', 0)
    )
    df['actual_sales'] += df.get('servicesrevenueinbillion', 0) * 1000  # Convert billions to millions
    df['sales_target'] = df['actual_sales'] * 0.9
    df['sales_vs_target'] = df['actual_sales'] - df['sales_target']

    # 📌 Select Role (CXO, Division Head, Line Manager)
    user_role = st.sidebar.selectbox("Choose Your Role", ["CXO", "Division Head", "Line Manager"])

    # ✅ KPI Metrics Based on Role
    st.subheader(f"📈 KPI Metrics for {user_role}")
    if user_role == "CXO":
        col1, col2, col3 = st.columns(3)
        col1.metric("Total Revenue", f"${df['actual_sales'].sum():,.2f}")
        col2.metric("Revenue Growth", f"{df['sales_vs_target'].mean():.2f}%")
        col3.metric("Profit Margin", "18.5%")  # Placeholder

    elif user_role == "Division Head" and "region" in df.columns:
        region = st.sidebar.selectbox("Select Region", df["region"].unique())
        df_region = df[df["region"] == region]
        col1, col2 = st.columns(2)
        col1.metric(f"{region} Sales", f"${df_region['actual_sales'].sum():,.2f}")
        col2.metric(f"{region} Sales Growth", f"{df_region['sales_vs_target'].mean():,.2f}%")

    elif user_role == "Line Manager" and "salesperson" in df.columns:
        salesperson = st.sidebar.selectbox("Select Salesperson", df["salesperson"].unique())
        df_salesperson = df[df["salesperson"] == salesperson]
        col1, col2 = st.columns(2)
        col1.metric(f"{salesperson} Sales", f"${df_salesperson['actual_sales'].sum():,.2f}")
        col2.metric(f"{salesperson} Target Achievement", f"{df_salesperson['sales_vs_target'].mean():,.2f}%")

    # ✅ AI-Powered Sales Insights (Google Gemini)
    st.subheader("🔍 AI-Generated Sales Insights")
    llm = ChatGoogleGenerativeAI(model="gemini-1.5-pro", google_api_key=os.environ["GOOGLE_API_KEY"])

    def generate_ai_insights(role):
        selected_columns = ["region", "actual_sales", "sales_target", "sales_vs_target"]
        filtered_df = df[selected_columns] if all(col in df.columns for col in selected_columns) else df

        prompt = f"""
        You are an AI sales analyst. Analyze the following sales data for the role: {role}.
        {filtered_df.to_string(index=False)}

        🔍 **Key Insights:**
        - **Top-Performing Region:**  
        - **Fastest-Growing Segment:**  
        - **Slowest-Growing Segment:**  
        - **Unexpected Trends:**  

        **Strategies to Optimize Sales Performance**
        """

        response = llm.invoke([HumanMessage(content=prompt)])
        return response.content

    if st.button("🔍 Generate AI Insights"):
        with st.spinner("⏳ Generating AI insights..."):
            ai_insights = generate_ai_insights(user_role)
        st.write(ai_insights)

else:
    st.warning("⚠️ No valid data found. Please upload a correct CSV or Excel file.")

