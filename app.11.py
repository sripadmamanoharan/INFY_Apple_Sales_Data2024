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
import io
from datetime import datetime

# ✅ Securely Load API Key
os.environ["GOOGLE_API_KEY"] = st.secrets["GOOGLE_API_KEY"]
genai.configure(api_key=os.environ["GOOGLE_API_KEY"])

# 🎯 Streamlit UI
st.set_page_config(page_title="AI Sales Dashboard", layout="wide")
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
                df = pd.read_csv(uploaded_file, nrows=5000)
            elif file_extension == "xlsx":
                df = pd.read_excel(uploaded_file, engine="openpyxl", nrows=5000)
            else:
                st.error("⚠️ Unsupported file format. Upload CSV or Excel.")
                return None
        except Exception as e:
            st.error(f"⚠️ Error loading file: {e}")
            return None
        df.columns = df.columns.str.strip().str.lower().str.replace(r'[^\w]', '', regex=True)
        return df
    else:
        st.warning("⚠️ No file uploaded.")
        return None

# ✅ Load Data
df = load_data(uploaded_file)

# ✅ Initialize AI Model
llm = ChatGoogleGenerativeAI(model="gemini-1.5-pro", google_api_key=os.environ["GOOGLE_API_KEY"])

if df is not None and not df.empty:
    # ✅ Compute Sales Metrics
    df['actual_sales'] = (
        df.get('iphonesalesinmillionunits', 0) +
        df.get('ipadsalesinmillionunits', 0) +
        df.get('macsalesinmillionunits', 0) +
        df.get('wearablesinmillionunits', 0)
    )
    df['actual_sales'] += df.get('servicesrevenueinbillion', 0) * 1000
    df['sales_target'] = df['actual_sales'] * 0.9
    df['sales_vs_target'] = df['actual_sales'] - df['sales_target']

    # ✅ Role Selection
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

    elif user_role == "Line Manager" and "salesperson" in df.columns:
        salesperson = st.sidebar.selectbox("Select Salesperson", df["salesperson"].unique())
        df_salesperson = df[df["salesperson"] == salesperson]
        col1, col2 = st.columns(2)
        col1.metric(f"{salesperson} Sales", f"${df_salesperson['actual_sales'].sum():,.2f}")
        col2.metric(f"{salesperson} Target Achievement", f"{df_salesperson['sales_vs_target'].mean():,.2f}%")

    # ✅ AI Insights
    st.subheader("🔍 AI-Generated Sales Insights")
    def generate_ai_insights(role):
        selected_columns = ["region", "actual_sales", "sales_target", "sales_vs_target"]
        filtered_df = df[selected_columns] if all(col in df.columns for col in selected_columns) else df
        prompt = f"""
        You are an AI sales analyst. Analyze the following sales data for the role: {role}.
        {filtered_df.to_string(index=False)}

        🔍 Key Insights:
        - Top-Performing Region:
        - Fastest-Growing Segment:
        - Slowest-Growing Segment:
        - Unexpected Trends:

        Strategies to Optimize Sales Performance:
        """
        response = llm.invoke([HumanMessage(content=prompt)])
        return response.content

    if st.button("🔍 Generate AI Insights"):
        with st.spinner("⏳ Generating AI insights..."):
            ai_insights = generate_ai_insights(user_role)
        st.write(ai_insights)

        # ✅ Download Insights
        buffer = io.StringIO()
        buffer.write("Sales KPI AI Insights\n\n")
        buffer.write(ai_insights)
        st.download_button(
            label="📥 Download Insights Report",
            data=buffer.getvalue(),
            file_name=f"sales_insights_{datetime.now().strftime('%Y%m%d_%H%M')}.txt",
            mime="text/plain"
        )

    # ✅ AI Chart Generator
   
    def generate_ai_chart(role):
        prompt = f"""
        You are a Python data assistant. Generate a Python code snippet using matplotlib or seaborn 
        to visualize a key sales KPI from the following data for the role {role}:
        {df.head(15).to_string(index=False)}

        Use readable labels and titles. Just return code, no explanation.
        """
        response = llm.invoke([HumanMessage(content=prompt)])
        return response.content

    # ✅ Generate and display AI-generated visualization
    if st.button("📊 Generate AI Visualization"):
        with st.spinner("⏳ Generating Chart with AI..."):
        ai_code = generate_ai_chart(user_role)
        st.code(ai_code, language="python")

        # ✅ Clean the AI code (in case it includes markdown formatting)
        if ai_code.startswith("```"):
            ai_code = ai_code.strip("`").replace("python", "").strip()
        ai_code = ai_code.replace("plt.show()", "")  # Remove plt.show() if included

        try:
            local_env = {}
            exec(ai_code, {"pd": pd, "plt": plt, "sns": sns}, local_env)
            st.pyplot(plt.gcf())  # ✅ Display the current plot
        except Exception as e:
            st.error(f"⚠️ Failed to run AI-generated code: {e}")

    # ✅ Natural Language Q&A
    st.subheader("💬 Ask a Question About Your Sales Data")
    user_query = st.text_input("Example: 'Which region has the highest revenue?'")

    if user_query:
        with st.spinner("🤖 Thinking..."):
            q_prompt = f"Answer this question based on the below sales data:\n{df.head(20)}\nQuestion: {user_query}"
            answer = llm.invoke([HumanMessage(content=q_prompt)])
            st.markdown(f"**AI Answer:** {answer.content}")

else:
    st.warning("⚠️ No valid data found. Please upload a correct CSV or Excel file.")
