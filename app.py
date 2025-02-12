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
import urllib.request

# ✅ Securely Load API Key
os.environ["GOOGLE_API_KEY"] = st.secrets["GOOGLE_API_KEY"]
genai.configure(api_key=os.environ["GOOGLE_API_KEY"])

# 🎯 Streamlit UI
st.title("📊 AI-Powered Sales KPI Dashboard")
st.sidebar.header("📂 Upload or Select Data Source")

# ✅ Database File Handling
DATABASE_URL = "sqlite:///sales.db"
db_url = "https://raw.githubusercontent.com/sripadmamanoharan/INFY_Apple_Sales_Data2024/main/sales.db"

# ✅ Function to Initialize Database
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

# ✅ Download `sales.db` if missing
if not os.path.exists("sales.db"):
    st.warning("⚠️ 'sales.db' not found. Downloading from GitHub...")
    try:
        urllib.request.urlretrieve(db_url, "sales.db")
        st.success("✅ Database downloaded successfully!")
    except Exception as e:
        st.error(f"❌ Failed to download database: {e}")
        st.info("Please manually upload `sales.db` below.")

# ✅ Manual Upload Option for `sales.db`
uploaded_file = st.sidebar.file_uploader("Upload Sales Database (`sales.db`)", type=["db"])
if uploaded_file is not None:
    with open("sales.db", "wb") as f:
        f.write(uploaded_file.getbuffer())
    st.success("✅ Database uploaded successfully!")

initialize_database()  # Ensure database table exists

# ✅ Load Data from Database
def load_from_database():
    engine = create_engine(DATABASE_URL)
    try:
        df = pd.read_sql("SELECT * FROM sales_data", con=engine)
        if df.empty:
            st.warning("⚠️ Database is empty! Upload a valid `sales.db` file.")
        return df
    except Exception as e:
        st.error(f"❌ Database error: {e}")
        return pd.DataFrame()

# ✅ Load Data Function (CSV, Excel, or Database)
@st.cache_data
def load_data():
    uploaded_csv = st.sidebar.file_uploader("Upload Sales Data (CSV/Excel)", type=["csv", "xlsx"])
    
    if uploaded_csv is not None:
        file_extension = uploaded_csv.name.split(".")[-1]
        if file_extension == "csv":
            df = pd.read_csv(uploaded_csv)
        elif file_extension == "xlsx":
            df = pd.read_excel(uploaded_csv, engine="openpyxl")
        else:
            st.error("⚠️ Unsupported file format. Upload CSV or Excel.")
            return None
    else:
        df = load_from_database()  # Load from database if no file is uploaded

    if df is None or df.empty:
        st.error("⚠️ No data loaded. Check database or uploaded file.")
        return None

    # ✅ Standardize column names
    df.columns = df.columns.str.strip().str.lower().str.replace(r'[^\w]', '', regex=True)

    return df

df = load_data()

# ✅ Debugging: Show Column Names
if df is not None:
    st.write("🛠 **Debugging: Column Names in Dataset**")
    st.write(df.columns.tolist())

if df is not None and not df.empty:
    # ✅ Ensure Key Sales Metrics Exist
    if all(col in df.columns for col in ["iphonesalesinmillionunits", "ipadsalesinmillionunits", 
                                         "macsalesinmillionunits", "wearablesinmillionunits", 
                                         "servicesrevenueinbillion"]):
        df['actual_sales'] = (
            df['iphonesalesinmillionunits'] +
            df['ipadsalesinmillionunits'] +
            df['macsalesinmillionunits'] +
            df['wearablesinmillionunits']
        )
        df['actual_sales'] += df['servicesrevenueinbillion'] * 1000
        df['sales_target'] = df['actual_sales'] * 0.9
        df['sales_vs_target'] = df['actual_sales'] - df['sales_target']
    else:
        st.error("⚠️ Missing expected columns in dataset.")

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
    llm = ChatGoogleGenerativeAI(model="gemini-pro", google_api_key=os.getenv("GOOGLE_API_KEY"))

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

        🚀 **Strategies to Optimize Sales Performance**
        """

        response = llm.invoke([HumanMessage(content=prompt)])
        return response.content

    if st.button("🔍 Generate AI Insights"):
        ai_insights = generate_ai_insights(user_role)
        st.write(ai_insights)
else:
    st.warning("⚠️ No data available for analysis. Please check your database or upload a valid file.")
