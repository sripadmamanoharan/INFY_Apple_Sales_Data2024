# 📊 AI-Powered Sales KPI Dashboard

🚀 **AI-powered dashboard for sales insights and forecasting using Python, Streamlit, and Google Gemini AI.**

## 🔥 Features
- 📂 Upload sales data (CSV/XLSX) or load from a database
- 📊 Visualize **Sales KPIs** for CXOs, Division Heads, and Line Managers
- 🤖 AI-Powered Insights using **Google Gemini AI**
- 📈 Forecast future sales trends with **Machine Learning**
- 🎨 Interactive **Streamlit Dashboard**
- 📡 Supports **SQLite database** integration

---

## 🛠️ **Technologies Used**
- **Python** 🐍 (Pandas, NumPy, Matplotlib, Seaborn)
- **Streamlit** 📊 (for interactive dashboard)
- **Google Gemini AI** 🤖 (for AI-powered sales insights)
- **LangChain** ⚡ (for AI interactions)
- **SQLAlchemy & SQLite** 🛢️ (for database support)
- **OpenAI API / Google API** 🔑 (for AI-powered recommendations)

---

## 📌 **Installation & Setup**
### 1️⃣ Clone the Repository
```bash
git clone https://github.com/yourusername/infy_apple_sales_data2024.git
cd infy_apple_sales_data2024
```

### 2️⃣ Install Dependencies
```bash
pip install -r requirements.txt
```

### 3️⃣ Set Up API Keys (Google AI)
- Add your **Google API Key** in a `.env` file or as a **Streamlit secret**:
```bash
GOOGLE_API_KEY=your_google_api_key
```
or inside **Streamlit secrets**:
```python
st.secrets["GOOGLE_API_KEY"] = "your_google_api_key"
```

### 4️⃣ Run the Dashboard
```bash
streamlit run app.py
```

---

## 📂 **Dataset**
Use a sample dataset **(CSV/XLSX format)** with columns like:
```csv
Date, Region, iPhone Sales (in million units), iPad Sales (in million units), Mac Sales (in million units), Wearables (in million units), Services Revenue (in billion $)
```

---

## 📸 **Screenshots**
📌 **Dashboard Interface:**
![Sales Dashboard](https://yourimageurl.com/dashboard.png)

📌 **AI Insights:**
![AI Insights](https://yourimageurl.com/ai_insights.png)

---

## 🎯 **Future Enhancements**
- ✅ Add **more AI-powered recommendations**
- ✅ **Integrate Power BI** for advanced analytics
- ✅ Deploy on **AWS/GCP**


