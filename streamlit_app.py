import streamlit as st
import yfinance as yf
import google.generativeai as genai
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go

# 🔑 Configure Gemini API (replace with st.secrets in production)
genai.configure(api_key="AIzaSyCuQ8cH78R1VUKfdHqAZBrxgeBYKXgURlY")
model = genai.GenerativeModel("gemini-2.5-pro")


# 📌 Cache stock data for faster performance
@st.cache_data
def get_stock_data(ticker):
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

            return {
                "symbol": ticker_symbol,
                "exchange": exch,
                "current_price": info.get("currentPrice"),
                "high_52week": info.get("fiftyTwoWeekHigh"),
                "low_52week": info.get("fiftyTwoWeekLow"),
                "pe_ratio": info.get("trailingPE"),
                "pb_ratio": info.get("priceToBook"),
                "roe": info.get("returnOnEquity"),
                "de_ratio": info.get("debtToEquity"),
                "div_yield": info.get("dividendYield"),
                "book_value": info.get("bookValue"),
                "face_value": info.get("lastSplitFactor"),  # sometimes used as proxy
                "eps_ttm": info.get("trailingEps"),
                "market_cap": info.get("marketCap"),
                "volume": info.get("volume"),
                "currency": info.get("currency"),
            }
        except Exception:
            continue
    return {"error": f"⚠️ Could not fetch data for {ticker}"}


def analyze_stock_with_gemini(ticker, data):
    """Ask Gemini to analyze stock fundamentals & give recommendation"""
    prompt = f"""
    You are a professional stock analyst. Analyze {ticker} using the following data:

    - Current Price: {data.get('current_price')}
    - 52 Week High: {data.get('high_52week')}
    - 52 Week Low: {data.get('low_52week')}
    - P/E Ratio: {data.get('pe_ratio')}
    - P/B Ratio: {data.get('pb_ratio')}
    - ROE: {data.get('roe')}
    - Debt/Equity: {data.get('de_ratio')}
    - Dividend Yield: {data.get('div_yield')}
    - EPS (TTM): {data.get('eps_ttm')}
    - Market Cap: {data.get('market_cap')}
    - Volume: {data.get('volume')}
    - Currency: {data.get('currency')}

    Provide:
    1. A short, beginner-friendly summary of this stock.
    2. Key opportunities and risks.
    3. A clear recommendation (Buy, Hold, or Sell) with reasoning.
    4. Long-term vs. short-term outlook.
    """
    try:
        response = model.generate_content(prompt)
        return response.text
    except Exception as e:
        return f"❌ Gemini analysis failed: {e}"

def plot_stock_chart(stock, period="6mo"):
    """Interactive chart with candlesticks & moving averages"""
    hist = stock.history(period=period)
    if hist.empty:
        return None

    hist["MA50"] = hist["Close"].rolling(50).mean()
    hist["MA200"] = hist["Close"].rolling(200).mean()

    fig = go.Figure()

    fig.add_trace(go.Candlestick(
        x=hist.index,
        open=hist["Open"],
        high=hist["High"],
        low=hist["Low"],
        close=hist["Close"],
        name="Price"
    ))

    fig.add_trace(go.Scatter(x=hist.index, y=hist["MA50"], mode="lines", name="50D MA"))
    fig.add_trace(go.Scatter(x=hist.index, y=hist["MA200"], mode="lines", name="200D MA"))

    fig.update_layout(
        title="📉 Stock Price with Moving Averages",
        yaxis=dict(title="Price"),
        xaxis=dict(title="Date"),
        template="plotly_white"
    )
    return fig


def plot_financials(stock):
    """Try to plot revenue, profit, net worth if available"""
    print("plot_financials :",stock)
    try:
        # Safely get financials
        fin = getattr(stock, "income_stmt", None)
        bs = getattr(stock, "balance_sheet", None)

        if fin is None or bs is None:
            return None

        if fin.empty or bs.empty:
            return None

        fin = fin.T
        bs = bs.T

        df = pd.DataFrame()

        if "Total Revenue" in fin.columns:
            df["Revenue"] = fin["Total Revenue"]
        if "Net Income" in fin.columns:
            df["Profit"] = fin["Net Income"]
        if "Total Stockholder Equity" in bs.columns:
            df["Net Worth"] = bs["Total Stockholder Equity"]

        if df.empty:
            return None

        df.index = pd.to_datetime(df.index).year
        fig = px.line(df, x=df.index, y=df.columns,
                      title="📊 Financial Performance (Yearly)",
                      markers=True)
        return fig

    except Exception as e:
        print("Financials error:", e)
        return None

# 🎨 Streamlit UI
st.set_page_config(page_title="Stock Buy/Sell Analyzer", layout="wide")
st.title("📈 Stock Buy/Sell Analyzer")

ticker = st.text_input("🔍 Enter Stock Ticker (e.g., AAPL, TSLA, INFY)", help="Supports NSE, BSE, and International tickers.")

if ticker:
    data = get_stock_data(ticker)

    if "error" in data:
        st.error(data["error"])
    else:
        st.success(f"✅ Found {ticker} on {data['exchange']}")
        stock_obj = yf.Ticker(data["symbol"])

        # Tabs
        tab1, tab2, tab3, tab4 = st.tabs(["📊 Key Metrics", "📉 Charts", "📑 Financials", "🤖 AI Insights"])

        with tab1:
            st.subheader("Key Metrics")
            cols = st.columns(4)

            cols[0].metric("💵 Current Price", f"{data['current_price']} {data['currency']}")
            cols[1].metric("📊 52W High", f"{data['high_52week']} {data['currency']}")
            cols[2].metric("📉 52W Low", f"{data['low_52week']} {data['currency']}")
            cols[3].metric("📈 P/E Ratio", data['pe_ratio'])

            cols2 = st.columns(4)
            cols2[0].metric("🏦 Market Cap", f"{data['market_cap']:,}")
            cols2[1].metric("📊 Volume", f"{data['volume']:,}")
            cols2[2].metric("📘 P/B Ratio", data.get("pb_ratio", "N/A"))
            cols2[3].metric("📈 ROE", data.get("roe", "N/A"))

            cols3 = st.columns(4)
            cols3[0].metric("⚖️ Debt/Equity", data.get("de_ratio", "N/A"))
            cols3[1].metric("💸 Dividend Yield", data.get("div_yield", "N/A"))
            cols3[2].metric("📖 Book Value", data.get("book_value", "N/A"))
            cols3[3].metric("💰 EPS (TTM)", data.get("eps_ttm", "N/A"))

        with tab2:
            st.subheader("Stock Price Chart (6 Months)")
            fig = plot_stock_chart(stock_obj)
            if fig:
                st.plotly_chart(fig, use_container_width=True)
            else:
                st.warning("⚠️ No historical data available.")

        with tab3:
            st.subheader("📑 Financial Performance")
            fin_fig = plot_financials(stock_obj)
            print("tab3: ",fin_fig)
            if fin_fig:
                st.plotly_chart(fin_fig, use_container_width=True)
            else:
                st.warning("⚠️ Financial data not available for this stock.")

        with tab4:
            st.subheader("AI-Powered Investment Analysis")
            analysis = analyze_stock_with_gemini(ticker, data)
            st.write(analysis)
