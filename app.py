# ==============================================================
# 📈 Stock Forecasting App using Prophet (Matplotlib + MultiIndex Safe)
# ==============================================================

import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from datetime import date, timedelta
from prophet import Prophet

# --------------------------------------------------------------
# 🎨 Streamlit Page Configuration
# --------------------------------------------------------------
st.set_page_config(page_title="📈 Stock Forecasting App", layout="wide")
st.title("📊 Stock Forecasting App")

st.markdown("""
This interactive app allows you to:
- View **historical stock performance**
- Generate **future price forecasts** using Meta’s Prophet model
- Visualize results using **Matplotlib**
---
""")

# --------------------------------------------------------------
# 🧭 Sidebar Controls
# --------------------------------------------------------------
st.sidebar.header("🔧 Configuration")

ticker = st.sidebar.text_input("Enter Stock Symbol (e.g. TCS.NS, INFY.NS, AAPL)", "TCS.NS")

today = date.today()
default_start = today - timedelta(days=365)

start_date = st.sidebar.date_input("Start Date", default_start)
end_date = st.sidebar.date_input("End Date", today)
forecast_period = st.sidebar.slider("Forecast Days", 30, 180, 60)

# --------------------------------------------------------------
# 📥 Download Stock Data
# --------------------------------------------------------------
st.write(f"### 📈 Historical Prices for **{ticker.upper()}**")

try:
    data = yf.download(ticker, start=start_date, end=end_date)

    if data.empty:
        st.warning("⚠️ No data found. Please check the stock symbol or date range.")
    else:
        # --------------------------------------------------------------
        # 🧩 Handle MultiIndex Columns (e.g., ('Close', 'TCS.NS'))
        # --------------------------------------------------------------
        data.columns = [
            '_'.join(col).strip() if isinstance(col, tuple) else col
            for col in data.columns
        ]

        # Identify the correct "Close" column dynamically
        close_cols = [col for col in data.columns if 'Close' in col]
        if not close_cols:
            st.error("❌ Could not find any 'Close' column in the dataset.")
        else:
            close_col = close_cols[0]  # Use the first matching one
            st.success(f"✅ Using column: **{close_col}** for Prophet model")

            st.dataframe(data.tail())

            # --------------------------------------------------------------
            # 🪄 Plot Historical Closing Price
            # --------------------------------------------------------------
            st.subheader("Closing Price Trend")
            fig, ax = plt.subplots(figsize=(10, 5))
            ax.plot(data[close_col], label='Closing Price', color='blue')
            ax.set_xlabel("Date")
            ax.set_ylabel("Price (INR)")
            ax.set_title(f"{ticker.upper()} - Closing Price History")
            ax.legend()
            st.pyplot(fig)

            # --------------------------------------------------------------
            # 📊 Summary Statistics
            # --------------------------------------------------------------
            st.subheader("Summary Statistics")

            avg_price = float(data[close_col].mean())
            max_price = float(data[close_col].max())
            min_price = float(data[close_col].min())

            col1, col2, col3 = st.columns(3)
            col1.metric("Average Price", f"₹{avg_price:,.2f}")
            col2.metric("Highest Price", f"₹{max_price:,.2f}")
            col3.metric("Lowest Price", f"₹{min_price:,.2f}")

            # --------------------------------------------------------------
            # 🔮 Prophet Forecasting (Matplotlib-based)
            # --------------------------------------------------------------
            st.subheader("🔮 Forecast using Prophet Model")

            try:
                # Prepare data for Prophet
                df = data[[close_col]].reset_index()
                df = df.rename(columns={'Date': 'ds', close_col: 'y'})

                # Ensure correct datatypes
                df['ds'] = pd.to_datetime(df['ds'], errors='coerce')
                df['y'] = pd.to_numeric(df['y'], errors='coerce')
                df = df.replace([np.inf, -np.inf], np.nan).dropna(subset=['ds', 'y'])

                if len(df) < 2:
                    st.warning("⚠️ Not enough data points for Prophet model.")
                else:
                    model = Prophet()
                    model.fit(df)

                    # Create future dataframe
                    future = model.make_future_dataframe(periods=forecast_period)
                    forecast = model.predict(future)

                    # ------------------------
                    # Forecast Plot (Matplotlib)
                    # ------------------------
                    st.subheader("📊 Forecast Plot")
                    fig1 = model.plot(forecast, xlabel="Date", ylabel="Price (INR)")
                    plt.title(f"{ticker.upper()} - Prophet Forecast ({forecast_period} Days Ahead)")
                    plt.grid(True)
                    st.pyplot(fig1)

                    # ------------------------
                    # Forecast Components Plot
                    # ------------------------
                    st.subheader("📈 Forecast Components")
                    fig2 = model.plot_components(forecast)
                    st.pyplot(fig2)

                    # ------------------------
                    # Forecast Data
                    # ------------------------
                    st.subheader("📄 Forecast Data Preview")
                    st.dataframe(forecast[['ds', 'yhat', 'yhat_lower', 'yhat_upper']].tail())

            except Exception as e:
                st.error(f"❌ Prophet model failed: {e}")

except Exception as e:
    st.error(f"❌ Error fetching data: {e}")

# --------------------------------------------------------------
# 📘 Footer
# --------------------------------------------------------------
st.markdown("""
---
✅ Developed by **Your Name**  
📅 Powered by Yahoo Finance API & Meta Prophet  
""")
