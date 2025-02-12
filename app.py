import streamlit as st
import pandas as pd
import google.generativeai as genai
import os
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.messages import HumanMessage
import sqlite3
import urllib.request
from sqlalchemy import create_engine
import time
from google.api_core.exceptions import ResourceExhausted

# ✅ Load API Key
os.environ["GOOGLE_API_KEY"] = st.secrets["GOOGLE_API_KEY"]
genai.configure(api_key=os.environ["GOOGLE_API_KEY"])

# ✅ Download Database if Not Exists
DATABASE_URL = "sqlite:///sales.db"
db_url = "https://raw.githubusercontent.com/sripadmamanoharan/INFY_Apple_Sales_Data2024/main/sales.db"

if not os.path.exists("sales.db"):
    st.warning("⚠️ Database file 'sales.db' not found. Downloading from GitHub...")
    urllib.request.urlretrieve(db_url, "sales.db")
    st.success("✅ Database downloaded successfully!")

# ✅ Load Data Function
@st.cache_data
def load_data():
    try:
        engine = create_engine(DATABASE_URL)
        df = pd.read_sql("SELECT * FROM sales_data", con=engine)
        return df
    except Exception as e:
        st.error(f"⚠️ Database error: {e}. Creating a new database...")
        return pd.DataFrame()  

df = load_data()

# ✅ Debugging Column Names
if df is not None and not df.empty:
    st.write("🛠 **Debugging: Column Names in Dataset**")
    st.write(df.columns.tolist())  
else:
    st.error("⚠️ No data loaded. Check database or uploaded file.")

# ✅ Ensure Key Sales Metrics Exist
if df is not None and not df.empty:
    df.columns = df.columns.str.strip().str.lower().str.replace(r'[^\w]', '', regex=True)
    
    df['actual_sales'] = (
        df['iphonesalesinmillionunits'] + 
        df['ipadsalesinmillionunits'] + 
        df['macsalesinmillionunits'] + 
        df['wearablesinmillionunits']
    )
    df['actual_sales'] += df['servicesrevenueinbillion'] * 1000
    df['sales_target'] = df['actual_sales'] * 0.9
    df['sales_vs_target'] = df['actual_sales'] - df['sales_target']

# ✅ Select Role
user_role = st.sidebar.selectbox("Choose Your Role", ["CXO", "Division Head", "Line Manager"])

# ✅ KPI Metrics Based on Role
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
    required_columns = ["region", "actual_sales", "sales_target", "sales_vs_target"]
    
    if all(col in df.columns for col in required_columns):
        filtered_df = df[required_columns]
    else:
        filtered_df = df  

    prompt = f"""
    You are an AI sales analyst. Analyze the following sales data for the role: {role}.
    {filtered_df.to_string(index=False)}
    """

    try:
        response = llm.invoke([HumanMessage(content=prompt)])
        return response.content
    except ResourceExhausted:
        time.sleep(10)
        st.warning("⚠️ API quota exceeded. Try again later.")
        return "API quota exceeded."

if st.button("🔍 Generate AI Insights"):
    ai_insights = generate_ai_insights(user_role)
    st.write(ai_insights)
