# 📈 Advanced Stock Market Analytics AI

An interactive stock market analytics dashboard built with Python and Streamlit to explore stock prices, technical indicators, market risk, experimental machine learning forecasts, and financial news sentiment.

## 🚀 Project Overview

Advanced Stock Market Analytics AI brings multiple stock analysis tools together in one dashboard. It helps users explore historical market data, understand price movements, analyze risk, and evaluate experimental machine learning predictions.

## ✨ Features

* **Interactive Dashboard:** View stock market information and recent price movements.
* **Stock Analysis:** Explore historical stock prices, moving averages, RSI, and other technical indicators.
* **Risk Analysis:** Analyze volatility, maximum drawdown, and historical recovery patterns.
* **Machine Learning Predictions:** Generate experimental next-session price estimates using a Ridge Regression model.
* **News Sentiment Analysis:** Examine financial headlines using sentiment analysis techniques.
* **Automated Reports:** Generate and download analytical reports and CSV results.
* **Interactive Visualizations:** Explore financial data through charts and tables.

## 🛠️ Technologies Used

* Python
* Streamlit
* Pandas and NumPy
* Matplotlib
* Scikit-learn
* XGBoost
* yfinance
* FinBERT and sentiment analysis tools
* Jupyter Notebook

## 📂 Project Structure

```text
Advanced-Stock-Market-Analytics-AI/
├── app.py
├── Stock_Vission_AI.ipynb
├── requirements.txt
├── .gitignore
├── data/
├── models/
└── project_outputs/
```

## ⚙️ Installation and Setup

### 1. Clone the repository

```bash
git clone https://github.com/gy296725-gif/Advanced-Stock-Market-Analytics-AI.git
```

### 2. Open the project folder

```bash
cd Advanced-Stock-Market-Analytics-AI
```

### 3. Create a virtual environment

```bash
python -m venv .venv
```

### 4. Activate the environment

**Windows:**

```bash
.venv\Scripts\activate
```

### 5. Install dependencies

```bash
python -m pip install -r requirements.txt
```

### 6. Run the dashboard

```bash
streamlit run app.py
```

## 📊 Machine Learning Approach

The project includes a Ridge Regression model that uses historical market information and engineered features to estimate next-session price movements.

Predictions are experimental and may be inaccurate. They should not be treated as guaranteed forecasts or financial advice.

## ⚠️ Disclaimer

This project is intended for educational and analytical purposes only. Stock market data may be delayed or incomplete, and machine learning predictions are uncertain. This dashboard does not provide financial advice or guarantee investment returns.

## 👨‍💻 Author

**Gautam Yadav**

GitHub: [gy296725-gif](https://github.com/gy296725-gif)
