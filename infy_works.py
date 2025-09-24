import pandas as pd
import streamlit as st
import plotly.express as px

# ---------- Title ----------
st.title("🚀 AI-Powered Sales Dashboard (Gemini)")

# ---------- File upload ----------
uploaded_file = st.file_uploader("Upload apple_sales_2024.csv", type=["csv"])

df = None
if uploaded_file is not None:
    try:
        df = pd.read_csv(uploaded_file)
        df.dropna(inplace=True)
        df.rename(columns=lambda x: x.strip(), inplace=True)
        st.success("✅ File uploaded and cleaned.")
        st.dataframe(df.head())
    except Exception as e:
        st.error(f"Could not read the CSV: {e}")

# ---------- KPIs & Chart ----------
if df is not None:
    # Compute KPIs
    try:
        kpis = {
            "Total iPhone Sales": float(df["iPhone Sales (in million units)"].sum()),
            "Total iPad Sales": float(df["iPad Sales (in million units)"].sum()),
            "Total Mac Sales": float(df["Mac Sales (in million units)"].sum()),
            "Total Wearables Sales": float(df["Wearables (in million units)"].sum()),
            "Total Services Revenue": float(df["Services Revenue (in billion $)"].sum()),
        }
    except KeyError as e:
        st.error(f"Missing expected column in CSV: {e}")
        st.stop()

    st.subheader("📊 Key Performance Indicators")
    st.json(kpis)

    st.subheader("📈 Sales by Region (stacked)")
    try:
        value_cols = [
            "iPhone Sales (in million units)",
            "iPad Sales (in million units)",
            "Mac Sales (in million units)",
            "Wearables (in million units)",
        ]
        region_sales = df.groupby("Region", as_index=False).sum(numeric_only=True)
        melted = region_sales.melt(
            id_vars="Region", value_vars=value_cols, var_name="Product", value_name="Units"
        )
        fig = px.bar(melted, x="Region", y="Units", color="Product", barmode="stack")
        st.plotly_chart(fig, use_container_width=True)
    except Exception as e:
        st.info(f"Could not plot by Region: {e}")

    # ---------- AI Insights (Google AI Studio / Gemini) ----------
    GOOGLE_API_KEY = st.secrets.get("GOOGLE_API_KEY")
    if not GOOGLE_API_KEY:
        st.info("💡 Add GOOGLE_API_KEY in Settings → Secrets to enable AI insights.")
    else:
        try:
            import google.generativeai as genai
            genai.configure(api_key=GOOGLE_API_KEY)

            # Use a lightweight model for quick insights; switch to gemini-1.5-pro if you prefer
            model = genai.GenerativeModel("gemini-1.5-flash")

            prompt = f"""
You are a sales analyst. Given these KPIs:
{kpis}

Return:
1) 3–5 short insights about trends or patterns.
2) 3 practical recommendations to improve sales.
Keep it concise and bullet-pointed.
"""
            with st.spinner("Generating AI insights..."):
                resp = model.generate_content(prompt)
            st.subheader("🤖 AI-Generated Insights")
            st.write(resp.text if hasattr(resp, "text") else resp.candidates[0].content.parts[0].text)
        except Exception as e:
            st.info(f"AI insights unavailable: {e}")
