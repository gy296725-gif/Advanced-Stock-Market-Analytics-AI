# STEP 291: CREATE STOCK VISSION AI WEB DASHBOARD

import streamlit as st
from pathlib import Path

import io
import re
import requests
import numpy as np
import pandas as pd
import yfinance as yf
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_absolute_error
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

# ---------------- PAGE CONFIGURATION ----------------

st.set_page_config(
    page_title="Stock Vission AI",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ---------------- CUSTOM CSS ----------------

st.markdown("""
<style>
    .stApp {
        background-color: #0b1220;
        color: #f1f5f9;
    }

    h1, h2, h3 {
        color: #e2e8f0;
    }

    .hero-box {
        padding: 28px;
        border-radius: 18px;
        background: linear-gradient(120deg, #172554, #0f766e);
        margin-bottom: 22px;
    }

    .hero-box p {
        color: #dbeafe;
        font-size: 17px;
    }

    .metric-box {
        background-color: #162033;
        padding: 18px;
        border-radius: 14px;
        border: 1px solid #263449;
    }

    .metric-label {
        color: #94a3b8;
        font-size: 14px;
    }

    .metric-value {
        color: #f8fafc;
        font-size: 24px;
        font-weight: bold;
    }

    section[data-testid="stSidebar"] {
        background-color: #111827;
    }
</style>
""", unsafe_allow_html=True)


# ---------------- SHARED MARKET DATA AND STOCK UNIVERSE ----------------

# A fallback snapshot is included so the page can still show a list if
# the official CSV download is temporarily unavailable. The official CSV
# is tried first and should be preferred whenever it can be downloaded.
fallback_constituents = [
    ("Adani Enterprises Ltd.", "ADANIENT"),
    ("Adani Ports & SEZ Ltd.", "ADANIPORTS"),
    ("Apollo Hospitals Enterprise Ltd.", "APOLLOHOSP"),
    ("Asian Paints Ltd.", "ASIANPAINT"),
    ("Axis Bank Ltd.", "AXISBANK"),
    ("Bajaj Auto Ltd.", "BAJAJ-AUTO"),
    ("Bajaj Finserv Ltd.", "BAJAJFINSV"),
    ("Bajaj Finance Ltd.", "BAJFINANCE"),
    ("Bharat Electronics Ltd.", "BEL"),
    ("Bharti Airtel Ltd.", "BHARTIARTL"),
    ("Cipla Ltd.", "CIPLA"),
    ("Coal India Ltd.", "COALINDIA"),
    ("Dr. Reddy's Laboratories Ltd.", "DRREDDY"),
    ("Eicher Motors Ltd.", "EICHERMOT"),
    ("Eternal Ltd.", "ETERNAL"),
    ("Grasim Industries Ltd.", "GRASIM"),
    ("HCL Technologies Ltd.", "HCLTECH"),
    ("HDFC Bank Ltd.", "HDFCBANK"),
    ("HDFC Life Insurance Company Ltd.", "HDFCLIFE"),
    ("Hindalco Industries Ltd.", "HINDALCO"),
    ("Hindustan Unilever Ltd.", "HINDUNILVR"),
    ("ICICI Bank Ltd.", "ICICIBANK"),
    ("InterGlobe Aviation Ltd.", "INDIGO"),
    ("Infosys Ltd.", "INFY"),
    ("ITC Ltd.", "ITC"),
    ("Jio Financial Services Ltd.", "JIOFIN"),
    ("JSW Steel Ltd.", "JSWSTEEL"),
    ("Kotak Mahindra Bank Ltd.", "KOTAKBANK"),
    ("Larsen & Toubro Ltd.", "LT"),
    ("Mahindra & Mahindra Ltd.", "M&M"),
    ("Maruti Suzuki India Ltd.", "MARUTI"),
    ("Max Healthcare Institute Ltd.", "MAXHEALTH"),
    ("Nestle India Ltd.", "NESTLEIND"),
    ("NTPC Ltd.", "NTPC"),
    ("Oil & Natural Gas Corporation Ltd.", "ONGC"),
    ("Power Grid Corporation of India Ltd.", "POWERGRID"),
    ("Reliance Industries Ltd.", "RELIANCE"),
    ("SBI Life Insurance Company Ltd.", "SBILIFE"),
    ("State Bank of India", "SBIN"),
    ("Shriram Finance Ltd.", "SHRIRAMFIN"),
    ("Sun Pharmaceutical Industries Ltd.", "SUNPHARMA"),
    ("Tata Consumer Products Ltd.", "TATACONSUM"),
    ("Tata Motors Passenger Vehicles Ltd.", "TMPV"),
    ("Tata Steel Ltd.", "TATASTEEL"),
    ("Tata Consultancy Services Ltd.", "TCS"),
    ("Tech Mahindra Ltd.", "TECHM"),
    ("Titan Company Ltd.", "TITAN"),
    ("Trent Ltd.", "TRENT"),
    ("UltraTech Cement Ltd.", "ULTRACEMCO"),
    ("BSE Ltd.", "BSE"),
]

@st.cache_data(ttl=86400, show_spinner=False)
def load_nifty50_constituents():
    """Load NIFTY 50 members from official CSV sources, with a fallback."""
    csv_sources = [
        (
            "https://nsearchives.nseindia.com/content/indices/ind_nifty50list.csv",
            "https://www.nseindia.com/static/products-services/indices-nifty50-index",
        ),
        (
            "https://www.niftyindices.com/IndexConstituent/ind_nifty50list.csv",
            "https://www.niftyindices.com/indices/equity/broad-based-indices/nifty--50",
        ),
    ]
    base_headers = {
        "User-Agent": (
            "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
            "AppleWebKit/537.36 (KHTML, like Gecko) "
            "Chrome/126.0.0.0 Safari/537.36"
        ),
    }

    for url, referer in csv_sources:
        try:
            headers = {**base_headers, "Referer": referer}
            response = requests.get(url, headers=headers, timeout=10)
            response.raise_for_status()
            raw = pd.read_csv(io.BytesIO(response.content))
            raw.columns = [str(column).strip() for column in raw.columns]

            normalized_columns = {
                re.sub(r"[^a-z0-9]", "", column.lower()): column
                for column in raw.columns
            }
            company_col = next(
                (
                    normalized_columns[key]
                    for key in ("companyname", "securityname", "company", "name")
                    if key in normalized_columns
                ),
                None,
            )
            symbol_col = next(
                (
                    normalized_columns[key]
                    for key in ("symbol", "tradingsymbol", "nsecodesymbol")
                    if key in normalized_columns
                ),
                None,
            )
            industry_col = next(
                (
                    normalized_columns[key]
                    for key in ("industry", "sector")
                    if key in normalized_columns
                ),
                None,
            )

            if company_col is None or symbol_col is None:
                raise ValueError("Expected company-name and symbol columns were not found.")

            constituents = pd.DataFrame(
                {
                    "Company Name": raw[company_col].astype(str).str.strip(),
                    "Symbol": raw[symbol_col].astype(str).str.strip().str.upper(),
                }
            )
            if industry_col is not None:
                constituents["Industry"] = raw[industry_col].fillna("—").astype(str).str.strip()
            else:
                constituents["Industry"] = "Not provided by source"

            constituents = constituents[
                constituents["Company Name"].ne("")
                & constituents["Symbol"].ne("")
                & constituents["Company Name"].str.lower().ne("nan")
                & constituents["Symbol"].str.lower().ne("nan")
            ].drop_duplicates(subset=["Symbol"]).reset_index(drop=True)

            if len(constituents) < 45:
                raise ValueError(
                    f"CSV contained only {len(constituents)} valid constituents."
                )

            # Apply the already-effective 30 Sep 2026 index change if the
            # downloadable CSV is still showing the older April snapshot.
            symbols_now = set(constituents["Symbol"])
            replacement_applied = False
            if "WIPRO" in symbols_now and "BSE" not in symbols_now:
                constituents = constituents[constituents["Symbol"] != "WIPRO"].copy()
                bse_row = pd.DataFrame(
                    [{
                        "Company Name": "BSE Ltd.",
                        "Symbol": "BSE",
                        "Industry": "Financial Services",
                    }]
                )
                constituents = pd.concat(
                    [constituents, bse_row], ignore_index=True
                )
                replacement_applied = True

            constituents["Yahoo Finance Ticker"] = constituents["Symbol"].apply(
                lambda symbol: symbol if symbol.endswith(".NS") else f"{symbol}.NS"
            )
            constituents = constituents[
                ["Company Name", "Industry", "Symbol", "Yahoo Finance Ticker"]
            ].sort_values("Company Name").reset_index(drop=True)
            source_note = "Official constituent CSV loaded"
            if replacement_applied:
                source_note += "; applied the 30 Sep 2026 WIPRO-to-BSE change"
            return constituents, f"{source_note} ({len(constituents)} companies)."

        except Exception:
            # Try the next official CSV URL before using the fallback.
            continue

    fallback = pd.DataFrame(
        fallback_constituents,
        columns=["Company Name", "Symbol"],
    )
    fallback["Industry"] = "Not included in fallback snapshot"
    fallback["Yahoo Finance Ticker"] = fallback["Symbol"].apply(
        lambda symbol: f"{symbol}.NS"
    )
    fallback = fallback[
        ["Company Name", "Industry", "Symbol", "Yahoo Finance Ticker"]
    ]
    return (
        fallback,
        "Official CSV could not be reached. Showing a built-in fallback snapshot "
        "effective 30 Sep 2026; it may not include later index changes.",
    )

nifty50_constituents, constituents_source = load_nifty50_constituents()

ticker_options = {
    "NIFTY 50 Index": "^NSEI",
    "SENSEX Index": "^BSESN",
}

for _, constituent in nifty50_constituents.iterrows():
    company_name = str(constituent["Company Name"]).strip()
    company_ticker = str(constituent["Yahoo Finance Ticker"]).strip()
    if (
        company_name
        and company_ticker
        and company_name.lower() != "nan"
        and company_ticker.lower() != "nan"
        and company_ticker not in ticker_options.values()
    ):
        ticker_options[company_name] = company_ticker

# Keep familiar short labels from the original dashboard available too.
for alias, symbol in {
    "TCS": "TCS.NS",
    "Reliance Industries": "RELIANCE.NS",
    "Infosys": "INFY.NS",
    "HDFC Bank": "HDFCBANK.NS",
    "ICICI Bank": "ICICIBANK.NS",
}.items():
    ticker_options.setdefault(alias, symbol)


@st.cache_data(ttl=3600, show_spinner=False)
def load_stock_history(symbol, period="5y"):
    """Download adjusted daily OHLCV history for any supported Yahoo ticker."""
    history = yf.download(
        symbol,
        period=period,
        interval="1d",
        auto_adjust=True,
        progress=False,
        threads=False,
    )
    if history is None or history.empty:
        return pd.DataFrame()

    if isinstance(history.columns, pd.MultiIndex):
        history.columns = history.columns.get_level_values(0)

    history.columns = [str(column).title() for column in history.columns]
    history = history.loc[:, ~history.columns.duplicated()].copy()
    history = history.sort_index()

    for column in ["Open", "High", "Low", "Close", "Volume"]:
        if column not in history.columns:
            history[column] = np.nan
        history[column] = pd.to_numeric(history[column], errors="coerce")

    return history


def market_price_text(value, is_index=False):
    """Use index points for indices and INR for individual equities."""
    if pd.isna(value):
        return "Not available"
    return f"{value:,.2f} points" if is_index else f"₹{value:,.2f}"


def technical_indicators(history):
    """Return a copy of history with commonly used technical indicators."""
    data = history.copy().sort_index()
    close = data["Close"]
    data["Daily_Return_Pct"] = close.pct_change() * 100
    data["MA20"] = close.rolling(20).mean()
    data["MA50"] = close.rolling(50).mean()

    delta = close.diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)
    avg_gain = gain.ewm(alpha=1 / 14, min_periods=14, adjust=False).mean()
    avg_loss = loss.ewm(alpha=1 / 14, min_periods=14, adjust=False).mean()
    relative_strength = avg_gain / avg_loss.replace(0, np.nan)
    rsi = 100 - (100 / (1 + relative_strength))
    rsi = rsi.mask(avg_loss.eq(0) & avg_gain.gt(0), 100)
    rsi = rsi.mask(avg_loss.eq(0) & avg_gain.eq(0), 50)
    data["RSI_14"] = rsi
    return data


FORECAST_FEATURE_COLUMNS = [
    "Open_Rel_Pct", "High_Rel_Pct", "Low_Rel_Pct",
    "MA20_Rel_Pct", "MA50_Rel_Pct", "MA200_Rel_Pct",
    "EMA12_Rel_Pct", "EMA26_Rel_Pct",
    "BB_Middle_Rel_Pct", "BB_Upper_Rel_Pct", "BB_Lower_Rel_Pct",
    "MACD_Pct", "MACD_Signal_Pct", "Daily_Return", "RSI_14",
    "Volume_Ratio", "High_Low_Range_Pct", "Open_Close_Change_Pct",
]

FORECAST_TARGET_COLUMNS = [
    "Target_Next_Open_Pct", "Target_Next_High_Pct",
    "Target_Next_Low_Pct", "Target_Next_Close_Pct",
]


def build_normalized_features(history):
    """Build current-day features as percentages/ratios to support each ticker."""
    data = history.copy().sort_index()
    required = ["Open", "High", "Low", "Close"]
    missing = [column for column in required if column not in data.columns]
    if missing:
        raise ValueError(f"Missing required OHLC columns: {missing}")

    data = data.dropna(subset=required).copy()
    close = data["Close"].replace(0, np.nan)
    volume = pd.to_numeric(data.get("Volume", pd.Series(np.nan, index=data.index)), errors="coerce")

    data["Daily_Return"] = data["Close"].pct_change() * 100
    for window in (20, 50, 200):
        data[f"MA{window}"] = data["Close"].rolling(window).mean()

    delta = data["Close"].diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)
    avg_gain = gain.ewm(alpha=1 / 14, min_periods=14, adjust=False).mean()
    avg_loss = loss.ewm(alpha=1 / 14, min_periods=14, adjust=False).mean()
    rs = avg_gain / avg_loss.replace(0, np.nan)
    rsi = 100 - (100 / (1 + rs))
    rsi = rsi.mask(avg_loss.eq(0) & avg_gain.gt(0), 100)
    rsi = rsi.mask(avg_loss.eq(0) & avg_gain.eq(0), 50)
    data["RSI_14"] = rsi

    data["EMA12"] = data["Close"].ewm(span=12, adjust=False).mean()
    data["EMA26"] = data["Close"].ewm(span=26, adjust=False).mean()
    data["MACD"] = data["EMA12"] - data["EMA26"]
    data["MACD_Signal"] = data["MACD"].ewm(span=9, adjust=False).mean()
    data["BB_Middle"] = data["Close"].rolling(20).mean()
    rolling_std = data["Close"].rolling(20).std()
    data["BB_Upper"] = data["BB_Middle"] + 2 * rolling_std
    data["BB_Lower"] = data["BB_Middle"] - 2 * rolling_std

    if volume.notna().any() and volume.fillna(0).gt(0).any():
        volume_ma20 = volume.rolling(20).mean()
        volume_ratio = volume / volume_ma20.replace(0, np.nan)
        # Some index feeds do not publish useful volume; use a neutral ratio then.
        data["Volume_Ratio"] = volume_ratio.replace([np.inf, -np.inf], np.nan).fillna(1.0)
    else:
        data["Volume_Ratio"] = 1.0

    features = pd.DataFrame(index=data.index)
    relative_columns = [
        "Open", "High", "Low", "MA20", "MA50", "MA200",
        "EMA12", "EMA26", "BB_Middle", "BB_Upper", "BB_Lower",
    ]
    for column in relative_columns:
        features[f"{column}_Rel_Pct"] = (data[column] / close - 1) * 100

    features["MACD_Pct"] = data["MACD"] / close * 100
    features["MACD_Signal_Pct"] = data["MACD_Signal"] / close * 100
    features["Daily_Return"] = data["Daily_Return"]
    features["RSI_14"] = data["RSI_14"]
    features["Volume_Ratio"] = data["Volume_Ratio"]
    features["High_Low_Range_Pct"] = (data["High"] - data["Low"]) / close * 100
    features["Open_Close_Change_Pct"] = (data["Close"] / data["Open"].replace(0, np.nan) - 1) * 100
    return features[FORECAST_FEATURE_COLUMNS].replace([np.inf, -np.inf], np.nan)


@st.cache_data(ttl=900, show_spinner=False)
def load_selected_ticker_news(symbol, company_name=None):
    """Fetch recent headlines from Yahoo Finance, then Google News RSS as a fallback."""
    columns = ["Published", "Headline", "Publisher", "Link"]
    rows = []

    # Source 1: Yahoo Finance. Coverage can be empty for some Indian tickers.
    try:
        items = yf.Ticker(symbol).news or []
    except Exception:
        items = []

    for item in items:
        if not isinstance(item, dict):
            continue
        content = item.get("content", item)
        if not isinstance(content, dict):
            continue
        title = content.get("title") or content.get("headline") or item.get("title")
        if not title:
            continue
        url_data = content.get("canonicalUrl") or content.get("clickThroughUrl") or {}
        if isinstance(url_data, dict):
            link = url_data.get("url", "")
        else:
            link = str(url_data or "")
        link = link or item.get("link", "")
        provider = content.get("provider") or item.get("publisher") or {}
        publisher = provider.get("displayName", "") if isinstance(provider, dict) else str(provider)
        published = content.get("pubDate") or item.get("providerPublishTime") or ""
        if isinstance(published, (int, float)):
            published = pd.to_datetime(published, unit="s", errors="coerce", utc=True)
            published = published.strftime("%d %b %Y %H:%M UTC") if pd.notna(published) else ""
        rows.append({"Published": str(published), "Headline": str(title), "Publisher": str(publisher), "Link": str(link)})

    if rows:
        return pd.DataFrame(rows, columns=columns).drop_duplicates(subset=["Headline"]).reset_index(drop=True)

    # Source 2: Google News RSS fallback. Use the display name for better
    # relevance and include the ticker symbol to help distinguish companies.
    name = str(company_name or symbol).strip()
    short_symbol = str(symbol).replace(".NS", "").replace(".BO", "").replace("^", "").strip()
    if name == "NIFTY 50 Index":
        query = 'NIFTY 50 India market index'
    elif name == "SENSEX Index":
        query = 'BSE SENSEX India stock market index'
    else:
        query = f'"{name}" {short_symbol} India stock shares'

    try:
        response = requests.get(
            "https://news.google.com/rss/search",
            params={"q": query, "hl": "en-IN", "gl": "IN", "ceid": "IN:en"},
            headers={"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 Chrome/126.0.0.0 Safari/537.36"},
            timeout=12,
        )
        response.raise_for_status()
        import xml.etree.ElementTree as ET
        root = ET.fromstring(response.content)
        for item in root.findall(".//item")[:25]:
            title = (item.findtext("title") or "").strip()
            if not title:
                continue
            link = (item.findtext("link") or "").strip()
            pub_date = (item.findtext("pubDate") or "").strip()
            source_element = item.find("source")
            publisher = (source_element.text or "").strip() if source_element is not None else "Google News RSS"
            parsed_date = pd.to_datetime(pub_date, errors="coerce", utc=True)
            published = parsed_date.strftime("%d %b %Y %H:%M UTC") if pd.notna(parsed_date) else pub_date
            rows.append({"Published": published, "Headline": title, "Publisher": publisher, "Link": link})
    except Exception:
        # Return an empty result cleanly if neither news source is reachable.
        pass

    if not rows:
        return pd.DataFrame(columns=columns)
    return pd.DataFrame(rows, columns=columns).drop_duplicates(subset=["Headline"]).reset_index(drop=True)


@st.cache_resource(show_spinner=False)
def load_local_finbert_classifier():
    """Use a locally cached FinBERT model when available; never force a large download."""
    try:
        from transformers import AutoModelForSequenceClassification, AutoTokenizer, pipeline

        tokenizer = AutoTokenizer.from_pretrained(
            "ProsusAI/finbert", local_files_only=True
        )
        model = AutoModelForSequenceClassification.from_pretrained(
            "ProsusAI/finbert", local_files_only=True
        )
        classifier = pipeline(
            "text-classification",
            model=model,
            tokenizer=tokenizer,
            return_all_scores=True,
            device=-1,
        )
        return classifier, None
    except Exception as error:
        # The keyword fallback below keeps this page useful without model weights.
        return None, str(error)


def keyword_headline_sentiment(text):
    """Transparent lightweight fallback; it is a heuristic, not a trained model."""
    text = str(text).lower()
    positive_terms = {
        "beat", "beats", "growth", "grows", "gain", "gains", "rise", "rises",
        "rose", "surge", "surges", "rally", "record high", "profit", "profits",
        "strong", "upgrade", "upgraded", "buyback", "dividend", "expansion",
        "outperform", "bullish", "rebound", "recovery", "wins", "positive",
        "improves", "improvement", "jumps", "jumps", "higher", "inflow",
    }
    negative_terms = {
        "miss", "misses", "missed", "decline", "declines", "fall", "falls",
        "fell", "drop", "drops", "dropped", "slump", "crash", "weak",
        "downgrade", "downgraded", "cut", "cuts", "loss", "losses",
        "lawsuit", "probe", "penalty", "fraud", "warning", "bearish",
        "selloff", "outflow", "lower", "debt concern", "layoff", "layoffs",
        "negative", "below estimates", "disappoints", "disappointing",
    }
    positive_count = sum(1 for term in positive_terms if term in text)
    negative_count = sum(1 for term in negative_terms if term in text)
    net_score = positive_count - negative_count
    if net_score > 0:
        sentiment = "Positive"
    elif net_score < 0:
        sentiment = "Negative"
    else:
        sentiment = "Neutral"
    confidence = 0.50 if net_score == 0 else min(0.88, 0.55 + 0.08 * abs(net_score))
    return sentiment, confidence, "Keyword fallback"


def score_headline_sentiment(news_frame):
    """Score live headlines using locally cached FinBERT, or a disclosed fallback."""
    if news_frame is None or news_frame.empty:
        empty = pd.DataFrame(columns=list(news_frame.columns) + [
            "Sentiment", "Sentiment_Confidence", "Sentiment_Method"
        ]) if isinstance(news_frame, pd.DataFrame) else pd.DataFrame()
        return empty, "No headlines available"

    result = news_frame.copy()
    headlines = result["Headline"].fillna("").astype(str).tolist()
    classifier, model_error = load_local_finbert_classifier()

    if classifier is not None:
        try:
            raw_predictions = classifier(headlines, truncation=True, max_length=128)
            if len(raw_predictions) != len(headlines):
                raise ValueError("FinBERT returned a different number of results than headlines.")

            sentiments = []
            confidences = []
            for item in raw_predictions:
                candidates = item if isinstance(item, list) else [item]
                candidates = [candidate for candidate in candidates if isinstance(candidate, dict)]
                if not candidates:
                    raise ValueError("FinBERT returned an unrecognised result format.")
                best = max(candidates, key=lambda candidate: float(candidate.get("score", 0)))
                label = str(best.get("label", "neutral")).lower()
                mapped = {"positive": "Positive", "negative": "Negative", "neutral": "Neutral"}.get(label)
                if mapped is None:
                    raise ValueError(f"Unrecognised FinBERT label: {label}")
                sentiments.append(mapped)
                confidences.append(float(best.get("score", 0.5)))

            result["Sentiment"] = sentiments
            result["Sentiment_Confidence"] = confidences
            result["Sentiment_Method"] = "FinBERT (local model)"
            return result, "FinBERT (local model)"
        except Exception:
            # Fall back instead of making the user's News Sentiment page fail.
            pass

    scores = [keyword_headline_sentiment(headline) for headline in headlines]
    result["Sentiment"] = [score[0] for score in scores]
    result["Sentiment_Confidence"] = [score[1] for score in scores]
    result["Sentiment_Method"] = [score[2] for score in scores]
    return result, "Keyword fallback (heuristic; not FinBERT)"


def generate_forecast_bundle(history):
    """Train/evaluate the selected instrument's normalized Ridge forecast."""
    forecast_history = history.dropna(subset=["Open", "High", "Low", "Close"]).copy()
    if forecast_history.empty:
        raise ValueError("No valid OHLC rows are available for forecasting.")

    features_all = build_normalized_features(forecast_history)
    close = forecast_history["Close"].astype(float)
    targets_all = pd.DataFrame(index=forecast_history.index)
    targets_all["Target_Next_Open_Pct"] = (forecast_history["Open"].shift(-1) / close - 1) * 100
    targets_all["Target_Next_High_Pct"] = (forecast_history["High"].shift(-1) / close - 1) * 100
    targets_all["Target_Next_Low_Pct"] = (forecast_history["Low"].shift(-1) / close - 1) * 100
    targets_all["Target_Next_Close_Pct"] = (forecast_history["Close"].shift(-1) / close - 1) * 100

    labelled = features_all.join(targets_all).replace([np.inf, -np.inf], np.nan).dropna()
    latest_features = features_all.replace([np.inf, -np.inf], np.nan).dropna()
    if len(labelled) < 150 or latest_features.empty:
        raise ValueError("At least 150 usable historical rows are needed for a forecast and chronological evaluation.")

    split_at = int(len(labelled) * 0.80)
    if split_at < 100 or len(labelled) - split_at < 20:
        raise ValueError("Not enough rows remain for chronological holdout validation.")

    X_train = labelled[FORECAST_FEATURE_COLUMNS].iloc[:split_at]
    y_train = labelled[FORECAST_TARGET_COLUMNS].iloc[:split_at]
    X_test = labelled[FORECAST_FEATURE_COLUMNS].iloc[split_at:]
    y_test = labelled[FORECAST_TARGET_COLUMNS].iloc[split_at:]

    evaluation_model = make_pipeline(StandardScaler(), Ridge(alpha=10.0))
    evaluation_model.fit(X_train, y_train)
    test_predictions = evaluation_model.predict(X_test)
    baseline_predictions = np.zeros_like(y_test.to_numpy(dtype=float))
    model_mae_by_target = mean_absolute_error(y_test, test_predictions, multioutput="raw_values")
    baseline_mae_by_target = mean_absolute_error(y_test, baseline_predictions, multioutput="raw_values")
    model_mae = float(np.mean(model_mae_by_target))
    baseline_mae = float(np.mean(baseline_mae_by_target))
    improvement_pct = ((baseline_mae - model_mae) / baseline_mae * 100) if baseline_mae else np.nan

    final_model = make_pipeline(StandardScaler(), Ridge(alpha=10.0))
    final_model.fit(labelled[FORECAST_FEATURE_COLUMNS], labelled[FORECAST_TARGET_COLUMNS])
    current_close = float(forecast_history["Close"].iloc[-1])
    latest_date = forecast_history.index[-1]
    predicted_returns = np.asarray(final_model.predict(latest_features.iloc[[-1]][FORECAST_FEATURE_COLUMNS])).reshape(-1)
    estimated_prices = current_close * (1 + predicted_returns / 100)
    output_names = ["Open", "High", "Low", "Close"]

    forecast_df = pd.DataFrame({
        "OHLC Target": output_names,
        "Estimated Change (%)": predicted_returns,
        "Estimated Price / Level": estimated_prices,
    })
    evaluation_df = pd.DataFrame({
        "Target": output_names,
        "Ridge MAE (percentage points)": model_mae_by_target,
        "Zero-Return Baseline MAE (percentage points)": baseline_mae_by_target,
    })
    evaluation_df["Improvement vs Baseline (%)"] = np.where(
        evaluation_df["Zero-Return Baseline MAE (percentage points)"].ne(0),
        (1 - evaluation_df["Ridge MAE (percentage points)"] / evaluation_df["Zero-Return Baseline MAE (percentage points)"]) * 100,
        np.nan,
    )
    details = {
        "latest_date": latest_date,
        "current_close": current_close,
        "predicted_returns": predicted_returns,
        "estimated_prices": estimated_prices,
        "model_mae": model_mae,
        "baseline_mae": baseline_mae,
        "improvement_pct": improvement_pct,
        "train_rows": len(X_train),
        "test_rows": len(X_test),
        "training_end": labelled.index[split_at - 1],
        "test_start": labelled.index[split_at],
        "test_end": labelled.index[-1],
    }
    return forecast_df, evaluation_df, details


# ---------------- SIDEBAR ----------------

st.sidebar.title("📈 Stock Vission AI")
st.sidebar.caption("Intelligent Market Analytics")

def refresh_data_and_retrain_ai():
    """Clear cached downloads and reopen AI Predictions for a fresh refit."""
    st.cache_data.clear()
    st.session_state["page_nav"] = "AI Predictions"
    st.session_state["refresh_notice"] = True


page = st.sidebar.radio(
    "NAVIGATION",
    [
        "Dashboard",
        "Stock Analysis",
        "Risk Analysis",
        "AI Predictions",
        "News Sentiment",
        "Reports"
    ],
    key="page_nav",
)

st.sidebar.divider()
st.sidebar.subheader("🔎 Analyse Any Stock")
selected_stock = st.sidebar.selectbox(
    "Select company or index",
    list(ticker_options.keys()),
    key="global_selected_stock",
)
ticker = ticker_options[selected_stock]
is_index = selected_stock in {"NIFTY 50 Index", "SENSEX Index"}
price_unit = "Index Points" if is_index else "Price (₹)"
st.sidebar.caption(f"Selected: {selected_stock} · {ticker}")

st.sidebar.button(
    "🔄 Refresh Data + Retrain AI",
    key="refresh_data_retrain_ai",
    help=(
        "Clears cached market/news downloads and opens AI Predictions. "
        "The selected instrument's Ridge model is then refitted using the "
        "latest available history."
    ),
    use_container_width=True,
    on_click=refresh_data_and_retrain_ai,
)
st.sidebar.caption(
    "Refresh clears cached market/headline data and refits the selected stock's "
    "Ridge forecast. Saved TCS-only FinBERT labels and older notebook reports "
    "are not regenerated by this button."
)
st.sidebar.caption("Market data and model estimates are for research purposes.")

# ---------------- HEADER ----------------

st.markdown("""
<div class="hero-box">
    <h1>📈 Stock Vission AI</h1>
    <p>
        Explore stock market data, technical indicators,
        model predictions, news sentiment and historical risk.
    </p>
</div>
""", unsafe_allow_html=True)

if st.session_state.pop("refresh_notice", False):
    st.info(
        f"Refresh requested for {selected_stock}. Cached market/news data was cleared. "
        "The selected instrument's AI forecast is now being recalculated from the "
        "latest available history. Other pages will recalculate their own metrics "
        "when you open them."
    )

# ---------------- DASHBOARD PAGE ----------------

if page == "Dashboard":

    st.subheader("Market Intelligence Dashboard")
    st.caption("The sidebar selection is shared across Stock Analysis, Risk Analysis and AI Predictions.")

    try:
        dashboard_history = load_stock_history(ticker, period="1y")
        if dashboard_history.empty:
            raise ValueError("No recent market data was returned.")

        dashboard_history = dashboard_history.dropna(subset=["Close"])
        latest_close = float(dashboard_history["Close"].iloc[-1])
        previous_close = (
            float(dashboard_history["Close"].iloc[-2])
            if len(dashboard_history) > 1 else None
        )
        daily_change = (
            (latest_close / previous_close - 1) * 100
            if previous_close not in (None, 0) else None
        )
        latest_date = dashboard_history.index[-1]

        col1, col2, col3 = st.columns(3)
        col1.metric("Selected Company / Index", selected_stock)
        col2.metric("Latest Price / Level", market_price_text(latest_close, is_index))
        col3.metric(
            "Latest Daily Change",
            f"{daily_change:+.2f}%" if daily_change is not None else "Not available",
        )
        st.caption(f"Latest available market-data date: {latest_date:%d %b %Y}")
        st.line_chart(dashboard_history[["Close"]], y_label=price_unit)
    except Exception as error:
        st.warning(f"Dashboard market summary could not load: {error}")

    st.info(
        "Choose any NIFTY 50 constituent or an index in the sidebar. "
        "All live market-analysis pages will use that selection."
    )

    st.subheader("Project Modules")
    col1, col2 = st.columns(2)
    with col1:
        st.markdown("### 📊 Stock Analysis")
        st.write("Historical prices, daily returns and technical indicators for the selected instrument.")
        st.markdown("### ⚠️ Risk Analysis")
        st.write("Volatility, drawdown, peak/trough and recovery calculations for the selected instrument.")
        st.markdown("### 🧠 AI Predictions")
        st.write("An experimental next-session OHLC estimate trained on the selected instrument's own history.")
    with col2:
        st.markdown("### 📰 News Sentiment")
        st.write("Saved FinBERT sentiment is available for TCS; other instruments show their available live headlines.")
        st.markdown("### 📂 Reports")
        st.write("Download saved research outputs. Older files with a tcs_ prefix are TCS-specific.")

    st.warning(
        "AI forecasts are experimental estimates, not guaranteed market outcomes or financial advice."
    )

# ---------------- STOCK ANALYSIS PAGE ----------------

# STEP 293: CONNECT STOCK ANALYSIS TO MARKET DATA

elif page == "Stock Analysis":

    st.subheader(f"📊 {selected_stock} — Stock Analysis")
    st.caption("Historical prices, technical indicators and downloadable market data")

    if selected_stock == "NIFTY 50 Index":
        st.subheader("🏢 NIFTY 50 Companies")
        st.caption(constituents_source)
        if "fallback snapshot" in constituents_source.lower():
            st.warning(
                "The fallback membership list is a snapshot. The official CSV is used automatically when available."
            )
        st.dataframe(nifty50_constituents, width="stretch", hide_index=True)
        st.divider()

    try:
        with st.spinner(f"Loading market data for {selected_stock}..."):
            stock_df = load_stock_history(ticker, period="2y")

        if stock_df.empty:
            st.error("Market data could not be loaded. Check the internet connection or try again later.")
        else:
            required_columns = {"Close", "High", "Low"}
            if not required_columns.issubset(stock_df.columns):
                missing = sorted(required_columns - set(stock_df.columns))
                st.error(f"Expected market data columns are missing: {missing}")
            else:
                stock_df = stock_df.dropna(subset=["Close"]).copy()
                stock_df = technical_indicators(stock_df)
                latest = stock_df.iloc[-1]
                previous_close = float(stock_df["Close"].iloc[-2]) if len(stock_df) > 1 else None
                latest_close = float(latest["Close"])
                daily_change = (
                    (latest_close / previous_close - 1) * 100
                    if previous_close not in (None, 0) else None
                )
                ma20_display = market_price_text(latest["MA20"], is_index)

                metric1, metric2, metric3 = st.columns(3)
                metric1.metric(
                    "Latest Index Level" if is_index else "Latest Adjusted Close",
                    market_price_text(latest_close, is_index),
                    delta=f"{daily_change:+.2f}%" if daily_change is not None else None,
                )
                metric2.metric(
                    "RSI (14)",
                    f"{latest['RSI_14']:.2f}" if pd.notna(latest["RSI_14"]) else "Not available",
                )
                metric3.metric("20-Day Moving Average", ma20_display)
                st.caption(f"Latest market-data date: {stock_df.index[-1]:%d %b %Y}")

                st.divider()
                st.subheader(f"{selected_stock} — Historical Price")
                chart_choice = st.radio(
                    "Choose chart",
                    ["Close Price", "Moving Averages"],
                    horizontal=True,
                    key="stock_analysis_chart_choice",
                )
                if chart_choice == "Close Price":
                    st.line_chart(stock_df[["Close"]], y_label=price_unit)
                else:
                    st.line_chart(stock_df[["Close", "MA20", "MA50"]], y_label=price_unit)

                st.subheader("📉 Daily Returns")
                st.line_chart(stock_df[["Daily_Return_Pct"]], y_label="Daily Return (%)")

                st.subheader("📋 Latest Market Data")
                display_columns = [
                    column for column in [
                        "Open", "High", "Low", "Close", "Volume",
                        "Daily_Return_Pct", "MA20", "MA50", "RSI_14",
                    ] if column in stock_df.columns
                ]
                if is_index or not stock_df["Volume"].notna().any():
                    display_columns = [column for column in display_columns if column != "Volume"]
                st.dataframe(stock_df[display_columns].tail(20).round(2), width="stretch")

                st.download_button(
                    "📥 Download Selected Instrument Data CSV",
                    data=stock_df.to_csv().encode("utf-8"),
                    file_name=f"{ticker.replace('^', '').replace('.', '_')}_history.csv",
                    mime="text/csv",
                )
    except Exception as error:
        st.error("Stock analysis could not be loaded. Check the internet connection and installed packages.")
        st.caption(f"Technical details: {error}")

    st.caption(
        "Prices use Yahoo Finance auto-adjusted historical data. Indicators describe past data and do not guarantee future movements."
    )

# STEP 292: CONNECT RISK ANALYSIS TO REAL PROJECT DATA

elif page == "Risk Analysis":

    st.subheader(f"⚠️ {selected_stock} Risk Analysis")
    st.caption("Risk metrics are calculated from the selected instrument's own market history.")

    try:
        with st.spinner(f"Calculating risk measures for {selected_stock}..."):
            risk_history = load_stock_history(ticker, period="5y")
        risk_history = risk_history.dropna(subset=["Close"]).copy()
        if len(risk_history) < 30:
            raise ValueError("Not enough valid price history to calculate risk measures.")

        close = risk_history["Close"].astype(float)
        returns_pct = close.pct_change() * 100
        running_peak = close.cummax()
        drawdown_pct = (close / running_peak - 1) * 100
        max_drawdown = float(drawdown_pct.min())
        trough_date = drawdown_pct.idxmin()
        drawdown_peak_price = float(running_peak.loc[trough_date])
        drawdown_peak_date = close.loc[:trough_date].idxmax()
        trough_price = float(close.loc[trough_date])
        latest_close = float(close.iloc[-1])
        historical_peak_price = float(close.max())
        historical_peak_date = close.idxmax()
        daily_return = float(returns_pct.iloc[-1]) if pd.notna(returns_pct.iloc[-1]) else np.nan
        rolling_volatility = returns_pct.rolling(20).std() * np.sqrt(252)
        latest_volatility = float(rolling_volatility.iloc[-1]) if pd.notna(rolling_volatility.iloc[-1]) else np.nan
        rebound_pct = (latest_close / trough_price - 1) * 100 if trough_price else np.nan
        gain_needed = (historical_peak_price / latest_close - 1) * 100 if latest_close else np.nan
        recovered = latest_close >= historical_peak_price

        c1, c2, c3 = st.columns(3)
        c1.metric("Latest Price / Level", market_price_text(latest_close, is_index))
        c2.metric("Latest Daily Return", f"{daily_return:+.2f}%" if pd.notna(daily_return) else "Not available")
        c3.metric("Maximum Drawdown", f"{max_drawdown:.2f}%")

        c4, c5, c6 = st.columns(3)
        c4.metric("20-Day Annualized Volatility", f"{latest_volatility:.2f}%" if pd.notna(latest_volatility) else "Not available")
        c5.metric("Rebound from Trough", f"{rebound_pct:.2f}%" if pd.notna(rebound_pct) else "Not available")
        c6.metric("Gain Needed to Reach Historical Peak", f"{gain_needed:.2f}%" if pd.notna(gain_needed) else "Not available")

        st.subheader("📋 Peak, Trough & Recovery")
        st.write(f"**Peak before the worst drawdown:** {drawdown_peak_date:%d %b %Y} ({market_price_text(drawdown_peak_price, is_index)})")
        st.write(f"**Worst drawdown trough:** {trough_date:%d %b %Y} ({market_price_text(trough_price, is_index)})")
        st.write(f"**5-year historical closing peak:** {historical_peak_date:%d %b %Y} ({market_price_text(historical_peak_price, is_index)})")
        st.write(f"**Current status vs 5-year historical peak:** {'At or above the historical peak' if recovered else 'Below the historical peak'}")

        st.divider()
        st.subheader("📈 20-Day Rolling Annualized Volatility")
        volatility_frame = rolling_volatility.rename("Rolling_Volatility_Pct").dropna().to_frame()
        st.line_chart(volatility_frame, y_label="Annualized Volatility (%)")

        st.subheader("📉 Drawdown from Prior High")
        drawdown_frame = drawdown_pct.rename("Drawdown_Pct").to_frame()
        st.line_chart(drawdown_frame, y_label="Drawdown (%)")

        risk_export = risk_history.copy()
        risk_export["Daily_Return_Pct"] = returns_pct
        risk_export["Rolling_Volatility_20D_Annualized_Pct"] = rolling_volatility
        risk_export["Drawdown_Pct"] = drawdown_pct
        st.download_button(
            "📥 Download Selected Instrument Risk Data",
            data=risk_export.to_csv().encode("utf-8"),
            file_name=f"{ticker.replace('^', '').replace('.', '_')}_risk_history.csv",
            mime="text/csv",
        )
    except Exception as error:
        st.error("Risk analysis could not be calculated for the selected instrument.")
        st.caption(f"Technical details: {error}")

    st.caption("Risk metrics describe historical behaviour. They do not predict or guarantee future returns.")

# STEP 304: DISPLAY NEXT-SESSION OHLC PREDICTIONS

elif page == "AI Predictions":

    st.subheader(f"🧠 {selected_stock} — Next-Session OHLC Forecast")
    st.caption("A separate experimental model is fitted to the selected instrument's own historical data.")

    try:
        with st.spinner(f"Downloading history and training a time-ordered model for {selected_stock}..."):
            forecast_history = load_stock_history(ticker, period="5y")

        if forecast_history.empty:
            raise ValueError("No market history was returned by Yahoo Finance.")
        forecast_history = forecast_history.dropna(subset=["Open", "High", "Low", "Close"]).copy()
        features_all = build_normalized_features(forecast_history)
        close = forecast_history["Close"].astype(float)
        targets_all = pd.DataFrame(index=forecast_history.index)
        targets_all["Target_Next_Open_Pct"] = (forecast_history["Open"].shift(-1) / close - 1) * 100
        targets_all["Target_Next_High_Pct"] = (forecast_history["High"].shift(-1) / close - 1) * 100
        targets_all["Target_Next_Low_Pct"] = (forecast_history["Low"].shift(-1) / close - 1) * 100
        targets_all["Target_Next_Close_Pct"] = (forecast_history["Close"].shift(-1) / close - 1) * 100

        labelled = features_all.join(targets_all).replace([np.inf, -np.inf], np.nan).dropna()
        latest_features = features_all.replace([np.inf, -np.inf], np.nan).dropna()
        if len(labelled) < 150 or latest_features.empty:
            raise ValueError("At least 150 usable historical rows are needed for a time-ordered evaluation.")

        split_at = int(len(labelled) * 0.80)
        if split_at < 100 or len(labelled) - split_at < 20:
            raise ValueError("Not enough rows remained for the training and chronological test split.")

        X_train = labelled[FORECAST_FEATURE_COLUMNS].iloc[:split_at]
        y_train = labelled[FORECAST_TARGET_COLUMNS].iloc[:split_at]
        X_test = labelled[FORECAST_FEATURE_COLUMNS].iloc[split_at:]
        y_test = labelled[FORECAST_TARGET_COLUMNS].iloc[split_at:]

        evaluation_model = make_pipeline(StandardScaler(), Ridge(alpha=10.0))
        evaluation_model.fit(X_train, y_train)
        test_predictions = evaluation_model.predict(X_test)
        baseline_predictions = np.zeros_like(y_test.to_numpy(dtype=float))

        model_mae_by_target = mean_absolute_error(
            y_test, test_predictions, multioutput="raw_values"
        )
        baseline_mae_by_target = mean_absolute_error(
            y_test, baseline_predictions, multioutput="raw_values"
        )
        model_mae = float(np.mean(model_mae_by_target))
        baseline_mae = float(np.mean(baseline_mae_by_target))
        improvement_pct = ((baseline_mae - model_mae) / baseline_mae * 100) if baseline_mae else np.nan

        # Fit on all labeled rows only after the chronological holdout has been evaluated.
        final_model = make_pipeline(StandardScaler(), Ridge(alpha=10.0))
        final_model.fit(labelled[FORECAST_FEATURE_COLUMNS], labelled[FORECAST_TARGET_COLUMNS])

        latest_row = latest_features.iloc[[-1]][FORECAST_FEATURE_COLUMNS]
        current_close = float(forecast_history["Close"].iloc[-1])
        latest_date = forecast_history.index[-1]
        predicted_returns = np.asarray(final_model.predict(latest_row)).reshape(-1)
        estimated_prices = current_close * (1 + predicted_returns / 100)

        st.write(f"**Selected instrument:** {selected_stock} (`{ticker}`)")
        st.write(f"**Latest available market-data date:** {latest_date:%d %b %Y}")
        st.write(f"**Latest close / level:** {market_price_text(current_close, is_index)}")
        st.info("Forecast refers to the next available trading session after the latest market-data date. It is an estimate, not a promised price.")

        output_names = ["Open", "High", "Low", "Close"]
        forecast_table = pd.DataFrame({
            "OHLC Target": output_names,
            "Estimated Change": [f"{value:+.3f}%" for value in predicted_returns],
            "Estimated Price / Level": [market_price_text(value, is_index) for value in estimated_prices],
        })
        st.subheader("🎯 Experimental Next-Session Estimate")
        st.dataframe(forecast_table, width="stretch", hide_index=True)

        close_change = float(predicted_returns[3])
        col1, col2 = st.columns(2)
        col1.metric("Estimated Close / Level", market_price_text(float(estimated_prices[3]), is_index))
        col2.metric("Estimated Change vs Latest Close", f"{close_change:+.3f}%")

        if estimated_prices[1] < max(estimated_prices[0], estimated_prices[3]) or estimated_prices[2] > min(estimated_prices[0], estimated_prices[3]):
            st.warning("The separate OHLC estimates do not satisfy normal High/Low price relationships. Treat the result cautiously.")

        target_labels = ["Open", "High", "Low", "Close"]
        evaluation_table = pd.DataFrame({
            "Target": target_labels,
            "Ridge MAE (percentage points)": model_mae_by_target,
            "Zero-Return Baseline MAE (percentage points)": baseline_mae_by_target,
        })
        evaluation_table["Improvement vs Baseline (%)"] = np.where(
            evaluation_table["Zero-Return Baseline MAE (percentage points)"] != 0,
            (1 - evaluation_table["Ridge MAE (percentage points)"] / evaluation_table["Zero-Return Baseline MAE (percentage points)"]) * 100,
            np.nan,
        )
        st.divider()
        st.subheader("📉 Historical Chronological Holdout Check")
        st.caption(f"Train rows: {len(X_train):,} · Test rows: {len(X_test):,} · Training dates end: {labelled.index[split_at - 1]:%d %b %Y} · Test dates: {labelled.index[split_at]:%d %b %Y} to {labelled.index[-1]:%d %b %Y}")
        m1, m2, m3 = st.columns(3)
        m1.metric("Ridge Average MAE", f"{model_mae:.3f} percentage points")
        m2.metric("Zero-Return Baseline MAE", f"{baseline_mae:.3f} percentage points")
        m3.metric("MAE Improvement vs Baseline", f"{improvement_pct:+.2f}%" if pd.notna(improvement_pct) else "Not available")
        st.dataframe(evaluation_table.round(3), width="stretch", hide_index=True)
        st.caption("The holdout check uses older dates to test later dates. Results differ by instrument and market regime. A lower MAE is better; this small experiment is not proof of future accuracy.")

        forecast_download = pd.DataFrame({
            "Target": output_names,
            "Estimated_Change_Pct": predicted_returns,
            "Estimated_Price_or_Level": estimated_prices,
        })
        st.download_button(
            "📥 Download Selected Instrument Forecast CSV",
            data=forecast_download.to_csv(index=False).encode("utf-8"),
            file_name=f"{ticker.replace('^', '').replace('.', '_')}_next_session_forecast.csv",
            mime="text/csv",
        )
        st.download_button(
            "📥 Download Selected Instrument Validation CSV",
            data=evaluation_table.to_csv(index=False).encode("utf-8"),
            file_name=f"{ticker.replace('^', '').replace('.', '_')}_forecast_validation.csv",
            mime="text/csv",
        )
    except Exception as error:
        st.error("A forecast could not be generated for this selection.")
        st.caption(f"Technical details: {error}")

    st.warning("Every instrument is forecast from its own historical data using a freshly fitted Ridge model. Forecasts are experimental—not guaranteed, not investment advice, and not proof of future performance.")

# STEP 296: CONNECT FINBERT NEWS SENTIMENT TO STREAMLIT

elif page == "News Sentiment":

    st.subheader(f"📰 {selected_stock} — News Sentiment")
    st.caption(
        "The page tries Yahoo Finance headlines first and falls back to Google News RSS. "
        "If a locally cached FinBERT model is available, it is used; otherwise a clearly "
        "labelled keyword fallback is used. Historical pre-scored FinBERT records are still available only for TCS."
    )

    try:
        with st.spinner(f"Loading latest headlines for {selected_stock}..."):
            live_news = load_selected_ticker_news(ticker, selected_stock)

        if live_news.empty:
            st.info(
                "No headlines were returned by Yahoo Finance or Google News RSS for this selection. "
                "Try Refresh Data + Retrain AI, or try another instrument."
            )
        else:
            with st.spinner("Classifying the available headlines..."):
                scored_news, sentiment_method = score_headline_sentiment(live_news)

            st.info(f"Sentiment method: {sentiment_method}")
            counts = scored_news["Sentiment"].value_counts().reindex(
                ["Positive", "Neutral", "Negative"], fill_value=0
            )
            c1, c2, c3, c4 = st.columns(4)
            c1.metric("Latest Headlines", f"{len(scored_news):,}")
            c2.metric("Positive", f"{int(counts.get('Positive', 0)):,}")
            c3.metric("Neutral", f"{int(counts.get('Neutral', 0)):,}")
            c4.metric("Negative", f"{int(counts.get('Negative', 0)):,}")

            st.subheader("📊 Live Headline Sentiment Distribution")
            st.bar_chart(counts, y_label="Number of Headlines")

            if sentiment_method.startswith("FinBERT"):
                confidence = pd.to_numeric(scored_news["Sentiment_Confidence"], errors="coerce")
                if confidence.notna().any():
                    st.metric("Average FinBERT Confidence", f"{confidence.mean() * 100:.2f}%")
            else:
                st.warning(
                    "FinBERT weights were not available locally, so the displayed labels "
                    "are a keyword-based heuristic, not model-generated sentiment. The "
                    "heuristic may misread context or sarcasm."
                )

            sentiment_filter = st.selectbox(
                "Filter headlines by sentiment",
                ["All", "Positive", "Neutral", "Negative"],
                key="live_news_sentiment_filter",
            )
            filtered_news = scored_news.copy()
            if sentiment_filter != "All":
                filtered_news = filtered_news[
                    filtered_news["Sentiment"].str.lower() == sentiment_filter.lower()
                ]

            display_columns = [
                column for column in [
                    "Published", "Headline", "Publisher", "Sentiment",
                    "Sentiment_Confidence", "Sentiment_Method", "Link"
                ] if column in filtered_news.columns
            ]
            display_news = filtered_news[display_columns].copy()
            if "Sentiment_Confidence" in display_news.columns:
                display_news["Sentiment_Confidence"] = display_news["Sentiment_Confidence"].map(
                    lambda value: f"{value * 100:.1f}%" if pd.notna(value) else ""
                )
            st.dataframe(display_news, width="stretch", hide_index=True)
            safe_ticker = re.sub(r"[^A-Za-z0-9_-]+", "_", ticker).strip("_") or "instrument"
            st.download_button(
                "📥 Download Selected Instrument News & Sentiment CSV",
                data=filtered_news.to_csv(index=False).encode("utf-8"),
                file_name=f"{safe_ticker}_news_sentiment.csv",
                mime="text/csv",
            )
    except Exception as error:
        st.warning(f"Could not retrieve or classify headlines for {selected_stock}: {error}")

    if ticker == "TCS.NS":
        with st.expander("📚 Historical TCS FinBERT sentiment archive"):
            news_path = Path(__file__).parent / "project_outputs" / "tcs_news_sentiment.csv"
            if not news_path.exists():
                st.info("The saved historical TCS FinBERT dataset was not found.")
            else:
                try:
                    saved_news = pd.read_csv(news_path)
                    if saved_news.empty:
                        st.info("The saved TCS sentiment dataset is empty.")
                    else:
                        saved_counts = saved_news["FinBERT_Sentiment"].fillna("Unknown").value_counts().reindex(
                            ["Positive", "Neutral", "Negative"], fill_value=0
                        )
                        st.write("This is an older, pre-scored TCS-only dataset, separate from live headlines above.")
                        st.bar_chart(saved_counts, y_label="Number of Saved Headlines")
                        saved_filter = st.selectbox(
                            "Filter historical TCS headlines",
                            ["All", "Positive", "Neutral", "Negative"],
                            key="saved_tcs_sentiment_filter",
                        )
                        saved_display = saved_news.copy()
                        saved_display["FinBERT_Sentiment"] = saved_display["FinBERT_Sentiment"].fillna("Unknown").astype(str)
                        if saved_filter != "All":
                            saved_display = saved_display[
                                saved_display["FinBERT_Sentiment"].str.lower() == saved_filter.lower()
                            ]
                        cols = [column for column in [
                            "Published", "Headline", "FinBERT_Sentiment", "FinBERT_Confidence", "Link"
                        ] if column in saved_display.columns]
                        st.dataframe(saved_display[cols], width="stretch", hide_index=True)
                        st.download_button(
                            "📥 Download Historical TCS FinBERT CSV",
                            data=saved_display[cols].to_csv(index=False).encode("utf-8"),
                            file_name="tcs_news_sentiment_historical.csv",
                            mime="text/csv",
                        )
                except Exception as error:
                    st.warning(f"Could not read the historical TCS sentiment archive: {error}")

    st.caption(
        "Headline sentiment is about language in the news, not a prediction that a stock "
        "will rise or fall. Live coverage and labels may be incomplete."
    )


elif page == "Reports":
    st.subheader(f"📂 {selected_stock} — Reports")
    st.caption(
        "This page now builds reports for the instrument selected in the sidebar. "
        "Older saved files are shown separately as a historical archive; files prefixed "
        "tcs_ are TCS-specific and should not be interpreted as reports for another stock."
    )

    output_dir = Path(__file__).parent / "project_outputs"
    safe_ticker = re.sub(r"[^A-Za-z0-9_-]+", "_", ticker).strip("_") or "instrument"

    try:
        with st.spinner(f"Building current reports for {selected_stock}..."):
            report_history = load_stock_history(ticker, period="5y")
        if report_history.empty:
            raise ValueError("Yahoo Finance returned no historical market data for this selection.")

        report_history = report_history.dropna(subset=["Close"]).copy()
        report_history = technical_indicators(report_history)
        close = pd.to_numeric(report_history["Close"], errors="coerce").dropna()
        if len(close) < 30:
            raise ValueError("At least 30 valid closing-price observations are needed for the report.")

        latest_date = close.index[-1]
        latest_close = float(close.iloc[-1])
        previous_close = float(close.iloc[-2]) if len(close) > 1 else np.nan
        latest_return = (latest_close / previous_close - 1) * 100 if pd.notna(previous_close) and previous_close != 0 else np.nan
        return_series = close.pct_change() * 100
        running_peak = close.cummax()
        drawdown_pct = (close / running_peak - 1) * 100
        rolling_volatility = return_series.rolling(20).std() * np.sqrt(252)
        max_drawdown = float(drawdown_pct.min())
        max_drawdown_date = drawdown_pct.idxmin()
        historical_peak_date = close.idxmax()
        historical_peak = float(close.max())
        latest_row = report_history.iloc[-1]

        report_history["Daily_Return_Pct"] = report_history["Close"].pct_change() * 100
        report_history["Drawdown_Pct"] = drawdown_pct.reindex(report_history.index)
        report_history["Rolling_Volatility_20D_Annualized_Pct"] = rolling_volatility.reindex(report_history.index)

        summary_rows = [
            ("Instrument", selected_stock),
            ("Yahoo Finance ticker", ticker),
            ("Latest market-data date", latest_date.strftime("%d %b %Y")),
            ("Latest Close / Index Level", market_price_text(latest_close, is_index)),
            ("Latest daily return", f"{latest_return:+.2f}%" if pd.notna(latest_return) else "Not available"),
            ("RSI (14)", f"{latest_row['RSI_14']:.2f}" if pd.notna(latest_row.get("RSI_14", np.nan)) else "Not available"),
            ("20-day moving average", market_price_text(latest_row.get("MA20", np.nan), is_index)),
            ("50-day moving average", market_price_text(latest_row.get("MA50", np.nan), is_index)),
            ("20-day annualized volatility", f"{rolling_volatility.iloc[-1]:.2f}%" if pd.notna(rolling_volatility.iloc[-1]) else "Not available"),
            ("Maximum drawdown over available history", f"{max_drawdown:.2f}%"),
            ("Worst drawdown date", max_drawdown_date.strftime("%d %b %Y")),
            ("Historical closing peak", market_price_text(historical_peak, is_index)),
            ("Historical peak date", historical_peak_date.strftime("%d %b %Y")),
            ("Status vs historical peak", "At/above historical peak" if latest_close >= historical_peak else "Below historical peak"),
        ]
        summary_df = pd.DataFrame(summary_rows, columns=["Metric", "Value"])

        st.subheader("📌 Current Instrument Summary")
        st.dataframe(summary_df, width="stretch", hide_index=True)
        st.download_button(
            "📥 Download Current Summary CSV",
            data=summary_df.to_csv(index=False).encode("utf-8"),
            file_name=f"{safe_ticker}_current_summary.csv",
            mime="text/csv",
        )

        st.subheader("📈 Historical Data & Technical Indicators")
        st.caption(f"{len(report_history):,} daily rows · latest date {latest_date:%d %b %Y}")
        st.dataframe(report_history.tail(20).round(3), width="stretch")
        st.download_button(
            "📥 Download Full History + Technical Indicators CSV",
            data=report_history.to_csv(index_label="Date").encode("utf-8"),
            file_name=f"{safe_ticker}_history_technical.csv",
            mime="text/csv",
        )
        st.download_button(
            "📥 Download Risk History CSV",
            data=report_history[[
                column for column in ["Open", "High", "Low", "Close", "Volume", "Daily_Return_Pct", "Drawdown_Pct", "Rolling_Volatility_20D_Annualized_Pct"]
                if column in report_history.columns
            ]].to_csv(index_label="Date").encode("utf-8"),
            file_name=f"{safe_ticker}_risk_history.csv",
            mime="text/csv",
        )

        st.divider()
        st.subheader("🧠 Current Instrument Forecast Report")
        with st.spinner("Training the selected instrument's experimental Ridge model and checking a chronological holdout..."):
            forecast_df, evaluation_df, forecast_details = generate_forecast_bundle(report_history)
        forecast_display = forecast_df.copy()
        forecast_display["Estimated Change (%)"] = forecast_display["Estimated Change (%)"].map(lambda value: f"{value:+.3f}%")
        forecast_display["Estimated Price / Level"] = forecast_display["Estimated Price / Level"].map(lambda value: market_price_text(value, is_index))
        st.write(f"**Latest data date:** {forecast_details['latest_date']:%d %b %Y} · **Latest close / level:** {market_price_text(forecast_details['current_close'], is_index)}")
        st.dataframe(forecast_display, width="stretch", hide_index=True)
        val1, val2, val3 = st.columns(3)
        val1.metric("Ridge Average MAE", f"{forecast_details['model_mae']:.3f} percentage points")
        val2.metric("Zero-Return Baseline MAE", f"{forecast_details['baseline_mae']:.3f} percentage points")
        val3.metric("Improvement vs Baseline", f"{forecast_details['improvement_pct']:+.2f}%" if pd.notna(forecast_details['improvement_pct']) else "Not available")
        st.dataframe(evaluation_df.round(3), width="stretch", hide_index=True)
        st.caption("This forecast is experimental. Its chronological holdout metrics describe past test data only and do not guarantee future accuracy.")
        st.download_button(
            "📥 Download Current Forecast CSV",
            data=forecast_df.to_csv(index=False).encode("utf-8"),
            file_name=f"{safe_ticker}_next_session_forecast.csv",
            mime="text/csv",
        )
        st.download_button(
            "📥 Download Forecast Validation CSV",
            data=evaluation_df.to_csv(index=False).encode("utf-8"),
            file_name=f"{safe_ticker}_forecast_validation.csv",
            mime="text/csv",
        )

        st.divider()
        st.subheader("📰 Latest Headlines & Sentiment Report")
        news_report = load_selected_ticker_news(ticker, selected_stock)
        if news_report.empty:
            st.info("No live headlines were available, so no current news report was generated.")
            scored_news = news_report
        else:
            scored_news, sentiment_method = score_headline_sentiment(news_report)
            news_counts = scored_news["Sentiment"].value_counts().reindex(["Positive", "Neutral", "Negative"], fill_value=0)
            st.caption(f"Sentiment method: {sentiment_method}")
            st.bar_chart(news_counts, y_label="Number of Headlines")
            st.dataframe(scored_news, width="stretch", hide_index=True)
            st.download_button(
                "📥 Download Current News + Sentiment CSV",
                data=scored_news.to_csv(index=False).encode("utf-8"),
                file_name=f"{safe_ticker}_news_sentiment.csv",
                mime="text/csv",
            )

        if st.button("💾 Save Current Instrument Report Pack", type="primary"):
            output_dir.mkdir(parents=True, exist_ok=True)
            saved_paths = []
            summary_path = output_dir / f"{safe_ticker}_current_summary.csv"
            history_path = output_dir / f"{safe_ticker}_history_technical.csv"
            risk_path = output_dir / f"{safe_ticker}_risk_history.csv"
            forecast_path = output_dir / f"{safe_ticker}_next_session_forecast.csv"
            validation_path = output_dir / f"{safe_ticker}_forecast_validation.csv"
            summary_df.to_csv(summary_path, index=False)
            report_history.to_csv(history_path, index_label="Date")
            report_history[[
                column for column in ["Open", "High", "Low", "Close", "Volume", "Daily_Return_Pct", "Drawdown_Pct", "Rolling_Volatility_20D_Annualized_Pct"]
                if column in report_history.columns
            ]].to_csv(risk_path, index_label="Date")
            forecast_df.to_csv(forecast_path, index=False)
            evaluation_df.to_csv(validation_path, index=False)
            saved_paths.extend([summary_path, history_path, risk_path, forecast_path, validation_path])
            if not scored_news.empty:
                news_path = output_dir / f"{safe_ticker}_news_sentiment.csv"
                scored_news.to_csv(news_path, index=False)
                saved_paths.append(news_path)
            st.success("Saved current reports: " + ", ".join(path.name for path in saved_paths))
            st.info("These newly generated files use the instrument selected in the sidebar. Old TCS-prefixed files remain historical TCS reports.")

    except Exception as error:
        st.error(f"Could not generate the complete report pack for {selected_stock}.")
        st.caption(f"Technical details: {error}")

    st.divider()
    with st.expander("📚 Historical / previously saved project reports"):
        st.caption("Files with a tcs_ prefix are from earlier TCS-only analysis. They remain useful as an archive, but they do not represent the currently selected stock.")
        if output_dir.exists():
            files = sorted(
                file for file in output_dir.iterdir()
                if file.is_file() and file.suffix.lower() in [".csv", ".png"]
            )
            if files:
                for file in files:
                    st.write(f"📄 {file.name}")
                    if file.suffix.lower() == ".png":
                        st.image(str(file), caption=file.name, width="stretch")
                    elif file.suffix.lower() == ".csv":
                        st.download_button(
                            label=f"Download {file.name}",
                            data=file.read_bytes(),
                            file_name=file.name,
                            mime="text/csv",
                            key=f"archive_download_{file.name}"
                        )
            else:
                st.info("No saved CSV reports or PNG charts found yet.")
        else:
            st.info("The project_outputs folder has not been created yet. Use the report-pack save button to create reports for this selection.")

st.divider()

st.caption(
    "Stock Vission AI | Experimental market analytics | "
    "Historical results do not guarantee future performance."
)


print("STEP 291 COMPLETED!")