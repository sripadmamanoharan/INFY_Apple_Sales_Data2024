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
GOOGLE_API_KEY = st.secrets.get("GOOGLE_API_KEY", None)

if not GOOGLE_API_KEY:
    st.error("⚠️ GOOGLE_API_KEY not found in Streamlit secrets! Add it to secrets.toml")
else:
    os.environ["GOOGLE_API_KEY"] = GOOGLE_API_KEY
    genai.configure(api_key=GOOGLE_API_KEY)

# ✅ Select a Working Gemini Model (Fix for Deprecated Models)
try:
    models = genai.list_models()
    available_models = [model.name for model in models]
    print("🧠 Available Models:", available_models)  # Debugging line

    if "gemini-1.5-flash" in available_models:
        MODEL_NAME = "gemini-1.5-flash"  # Fastest option
    elif "gemini-1.5-pro" in available_models:
        MODEL_NAME = "gemini-1.5-pro"  # More powerful
    else:
        st.error("⚠️ No valid Gemini models found! Check your API key permissions.")
        MODEL_NAME = None
except Exception as e:
    st.error(f"❌ Error fetching available Gemini models: {e}")
    MODEL_NAME = None

# 🎯 Streamlit UI
st.title("📊 AI-Powered Sales KPI Dashboard")
st.sidebar.header("📂 Upload or Select Data Source")

# ✅ File Upload Section
uploaded_file = st.sidebar.file_uploader("Upload Sales Data", type=["csv", "xlsx"])

if uploaded_file is not None:
    st.success(f"✅ File Uploaded: {uploaded_file.name}")
    print("📂 File uploaded:", uploaded_file.name)  # Debugging line
else:
    st.warning("⚠️ No file uploaded!")
    print("❌ No file uploaded.")  # Debugging line

# ✅ Database Connection (SQLite Example)
DATABASE_URL = "sqlite:///sales.db"

def initialize_database():
    """Creates the sales_data table if it does not exist."""
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

# ✅ Download Database If Missing
db_url = "https://github.com/sripadmamanoharan/INFY_Apple_Sales_Data2024/raw/main/sales.db"

if not os.path.exists("sales.db"):
    st.warning("⚠️ Database file 'sales.db' not found. Downloading from GitHub...")
    try:
        urllib.request.urlretrieve(db_url, "sales.db")
        st.success("✅ Database downloaded successfully!")
        initialize_database()
    except Exception as e:
        st.error(f"❌ Database download failed: {e}")

def load_from_database():
    """Loads data from the SQLite database."""
    engine = create_engine(DATABASE_URL)
    try:
        df = pd.read_sql("SELECT * FROM sales_data", con=engine)
        return df
    except Exception as e:
        st.error(f"⚠️ Database error: {e}. Creating a new database...")
        initialize_database()
        return pd.DataFrame()

# ✅ Load Data Function (CSV, Excel, or Database)
@st.cache_data
def load_data():
    """Loads data from uploaded file or database."""
    if uploaded_file is not None:
        st.write(f"✅ Processing file: {uploaded_file.name}")
        file_extension = uploaded_file.name.split(".")[-1]
        
        try:
            if file_extension == "csv":
                df = pd.read_csv(uploaded_file, encoding="utf-8", on_bad_lines="skip")
            elif file_extension == "xlsx":
                df = pd.read_excel(uploaded_file, engine="openpyxl")
            else:
                st.error("⚠️ Unsupported file format. Upload CSV or Excel.")
                return None

            if df is not None and not df.empty:
                print("✅ Data Loaded Successfully!", df.shape)  # Debugging line
                st.write("✅ Data Loaded Successfully!")
                return df
            else:
                print("❌ No data found in the file.")  # Debugging line
                st.error("❌ File read error. No data found in the file.")
                return None
        except Exception as e:
            print(f"❌ Error reading file: {e}")  # Debugging line
            st.error(f"❌ Error reading file: {e}")
            return None
    else:
        st.warning("⚠️ No file uploaded, loading from database instead...")
        return load_from_database()

df = load_data()

# ✅ Check if Data is Loaded
if df is not None and not df.empty:
    st.write("📌 **Dataset Preview**")
    st.dataframe(df)  # ✅ Ensure data is displayed in Streamlit UI
    print("📊 Columns after cleaning:", df.columns.tolist())  # Debugging line

    # ✅ Ensure Key Sales Metrics Exist
    required_columns = [
        "iphonesalesinmillionunits",
        "ipadsalesinmillionunits",
        "macsalesinmillionunits",
        "wearablesinmillionunits",
        "servicesrevenueinbillion",
    ]

    if all(col in df.columns for col in required_columns):
        df["actual_sales"] = (
            df["iphonesalesinmillionunits"]
            + df["ipadsalesinmillionunits"]
            + df["macsalesinmillionunits"]
            + df["wearablesinmillionunits"]
        )
        df["actual_sales"] += df["servicesrevenueinbillion"] * 1000
        df["sales_target"] = df["actual_sales"] * 0.9
        df["sales_vs_target"] = df["actual_sales"] - df["sales_target"]
    else:
        st.error(f"⚠️ Missing required columns! Found: {df.columns.tolist()}")

    # 📌 Select Role (CXO, Division Head, Line Manager)
    user_role = st.sidebar.selectbox("Choose Your Role", ["CXO", "Division Head", "Line Manager"])

    # ✅ AI-Powered Sales Insights
    st.subheader("🔍 AI-Generated Sales Insights")

    if MODEL_NAME:
        llm = ChatGoogleGenerativeAI(model=MODEL_NAME, google_api_key=GOOGLE_API_KEY)

        def generate_ai_insights(role):
            selected_columns = ["region", "actual_sales", "sales_target", "sales_vs_target"]
            filtered_df = df[selected_columns] if all(col in df.columns for col in selected_columns) else df

            prompt = f"""
            Analyze the following sales data for the role: {role}.
            {filtered_df.to_string(index=False)}

            🔍 **Key Insights:**
            - **Top-Performing Region:**  
            - **Fastest-Growing Segment:**  
            - **Unexpected Trends:**  

            🚀 **Strategies to Optimize Sales Performance**
            """

            response = llm.invoke([HumanMessage(content=prompt)])
            return response.content

        if st.button("🔍 Generate AI Insights"):
            ai_insights = generate_ai_insights(user_role)
            st.write(ai_insights)
    else:
        st.error("⚠️ No valid Gemini model available. Please check your API key permissions.")

else:
    st.error("⚠️ No data loaded. Check database or uploaded file.")
