import streamlit as st
import yfinance as yf
import google.generativeai as genai
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px

# ===========================
# 🔑 CONFIG
# ===========================
st.set_page_config(page_title="📈 Smart Stock Analyzer", layout="wide")
st.markdown(
    """
    <style>
    .big-font { font-size:20px !important; font-weight: bold; }
    .stMetric { background-color: #f8f9fa; border-radius: 12px; padding: 15px; box-shadow: 0 2px 6px rgba(0,0,0,0.1); }
    </style>
    """,
    unsafe_allow_html=True
)

# Replace with st.secrets in production
genai.configure(api_key="YOUR_API_KEY")
model = genai.GenerativeModel("gemini-2.5-pro")


# ===========================
# 📌 HELPERS
# ===========================
@st.cache_data
def get_stock_data(ticker: str) -> dict:
    """Fetch stock data with fallback priority: NSE → BSE → Intl"""
    ticker = ticker.upper()
    suffixes = [".NS", ".BO", ""]
    exchanges = ["NSE", "BSE", "INTL"]

    for suffix, exch in zip(suffixes, exchanges):
        ticker_symbol = ticker + suffix
        try:
            stock = yf.Ticker(ticker_symbol)
            info = stock.info

            if not info or "currentPrice" not in info:
                continue

            # Convert to serializable types
            return {
                "symbol": str(ticker_symbol),
                "exchange": exch,
                "current_price": float(info.get("currentPrice", 0) or 0),
                "high_52week": float(info.get("fiftyTwoWeekHigh", 0) or 0),
                "low_52week": float(info.get("fiftyTwoWeekLow", 0) or 0),
                "pe_ratio": float(info.get("trailingPE", 0) or 0),
                "pb_ratio": float(info.get("priceToBook", 0) or 0),
                "roe": float(info.get("returnOnEquity", 0) or 0),
                "de_ratio": float(info.get("debtToEquity", 0) or 0),
                "div_yield": float(info.get("dividendYield", 0) or 0),
                "book_value": float(info.get("bookValue", 0) or 0),
                "face_value": str(info.get("lastSplitFactor", "N/A")),
                "eps_ttm": float(info.get("trailingEps", 0) or 0),
                "market_cap": int(info.get("marketCap", 0) or 0),
                "volume": int(info.get("volume", 0) or 0),
                "currency": str(info.get("currency", "N/A")),
            }
        except Exception:
            continue
    return {"error": f"⚠️ Could not fetch data for {ticker}"}


def analyze_stock_with_gemini(ticker: str, data: dict) -> str:
    """Ask Gemini to analyze stock fundamentals & give recommendation"""
    prompt = f"""
    You are a financial advisor. Analyze {ticker} with this data:

    Current Price: {data.get('current_price')}
    52 Week High/Low: {data.get('high_52week')} / {data.get('low_52week')}
    P/E Ratio: {data.get('pe_ratio')}, P/B: {data.get('pb_ratio')}
    ROE: {data.get('roe')}, D/E: {data.get('de_ratio')}
    Dividend Yield: {data.get('div_yield')}
    EPS (TTM): {data.get('eps_ttm')}
    Market Cap: {data.get('market_cap')}, Volume: {data.get('volume')}

    Provide:
    1. Beginner-friendly overview 📘
    2. Strengths ✅
    3. Weaknesses ⚠️
    4. Recommendation 🎯 (Buy / Hold / Sell) with reasons
    5. Short vs Long-term outlook
    """
    try:
        response = model.generate_content(prompt)
        return response.text
    except Exception as e:
        return f"❌ Gemini analysis failed: {e}"


def plot_stock_chart(stock, period="6mo"):
    """Interactive candlestick chart with MAs & volume"""
    hist = stock.history(period=period)
    if hist.empty:
        return None

    hist["MA20"] = hist["Close"].rolling(20).mean()
    hist["MA50"] = hist["Close"].rolling(50).mean()
    hist["MA200"] = hist["Close"].rolling(200).mean()

    fig = go.Figure()

    # Price Candlestick
    fig.add_trace(go.Candlestick(
        x=hist.index, open=hist["Open"], high=hist["High"],
        low=hist["Low"], close=hist["Close"], name="Price",
        increasing_line_color="green", decreasing_line_color="red"
    ))

    # Moving Averages
    for ma, color in [("MA20", "blue"), ("MA50", "orange"), ("MA200", "purple")]:
        fig.add_trace(go.Scatter(x=hist.index, y=hist[ma], mode="lines", name=ma, line=dict(color=color)))

    # Volume
    fig.add_trace(go.Bar(x=hist.index, y=hist["Volume"], name="Volume", yaxis="y2", opacity=0.3))

    fig.update_layout(
        title="📉 Stock Trend with Volume",
        xaxis=dict(title="Date"),
        yaxis=dict(title="Price"),
        yaxis2=dict(title="Volume", overlaying="y", side="right", showgrid=False),
        template="plotly_white", height=600
    )
    return fig


def plot_financials(stock):
    """Revenue, Profit, Equity if available"""
    try:
        fin = stock.income_stmt
        bs = stock.balance_sheet
        if fin.empty or bs.empty:
            return None

        df = pd.DataFrame()
        if "Total Revenue" in fin.index:
            df["Revenue"] = fin.loc["Total Revenue"].T
        if "Net Income" in fin.index:
            df["Profit"] = fin.loc["Net Income"].T
        if "Stockholders Equity" in bs.index:
            df["Equity"] = bs.loc["Stockholders Equity"].T

        if df.empty:
            return None

        df.index = pd.to_datetime(df.index).year
        fig = px.line(df, x=df.index, y=df.columns, markers=True,
                      title="📊 Financial Performance (Yearly)")
        return fig
    except Exception as e:
        print("Financials error:", e)
        return None


# ===========================
# 🎨 UI
# ===========================
st.title("📈 Smart Stock Analyzer")
ticker = st.text_input("🔍 Enter Stock Ticker", placeholder="e.g., AAPL, TSLA, INFY").upper()

if st.button("🚀 Analyze"):
    if not ticker:
        st.warning("⚠️ Please enter a stock ticker.")
    else:
        data = get_stock_data(ticker)
        if "error" in data:
            st.error(data["error"])
        else:
            st.success(f"✅ Found {ticker} on {data['exchange']}")
            stock_obj = yf.Ticker(data["symbol"])

            tab1, tab2, tab3, tab4 = st.tabs(
                ["📊 Key Metrics", "📉 Price Chart", "📑 Financials", "🤖 AI Insights"]
            )

            # Key Metrics
            with tab1:
                st.markdown("### Company Snapshot")
                cols = st.columns(4)
                cols[0].metric("💵 Price", f"{data['current_price']} {data['currency']}")
                cols[1].metric("📊 52W High", f"{data['high_52week']}")
                cols[2].metric("📉 52W Low", f"{data['low_52week']}")
                cols[3].metric("📈 P/E", data['pe_ratio'])

                cols2 = st.columns(4)
                cols2[0].metric("🏦 Market Cap", f"{data['market_cap']:,}")
                cols2[1].metric("📊 Volume", f"{data['volume']:,}")
                cols2[2].metric("📘 P/B", data.get("pb_ratio", "N/A"))
                cols2[3].metric("📈 ROE", data.get("roe", "N/A"))

                cols3 = st.columns(4)
                cols3[0].metric("⚖️ D/E", data.get("de_ratio", "N/A"))
                cols3[1].metric("💸 Div Yield", data.get("div_yield", "N/A"))
                cols3[2].metric("📖 Book Value", data.get("book_value", "N/A"))
                cols3[3].metric("💰 EPS (TTM)", data.get("eps_ttm", "N/A"))

            # Charts
            with tab2:
                st.markdown("### Price & Volume Trend")
                fig = plot_stock_chart(stock_obj)
                if fig:
                    st.plotly_chart(fig, use_container_width=True)
                    st.markdown("""
                            **Explanation:**  
                            - The candlestick chart shows daily stock prices: green candles indicate price increases, red candles indicate decreases.  
                            - Moving Averages (MA20, MA50, MA200) are shown as lines to indicate short, medium, and long-term trends.  
                            - Volume bars at the bottom show trading activity; higher bars indicate more trades.
                            """)
                else:
                    st.warning("⚠️ No historical data.")

            # Financials
            with tab3:
                st.markdown("### Annual Performance")
                fin_fig = plot_financials(stock_obj)
                if fin_fig:
                    st.plotly_chart(fin_fig, use_container_width=True)
                    st.markdown("""
                            **Explanation:**  
                            - **Revenue**: Total income generated by the company each year.  
                            - **Profit (Net Income)**: Money left after all expenses are paid.  
                            - **Equity (Net Worth)**: Value of the company owned by shareholders.  
                            - Trends help investors identify growth, profitability, and financial health over time.
                            """)
                else:
                    st.warning("⚠️ Financial data unavailable.")

            # AI Insights
            with tab4:
                st.markdown("### AI-Powered Investment Insights")
                analysis = analyze_stock_with_gemini(ticker, data)
                st.write(analysis)
