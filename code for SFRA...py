STOCK_SYMBOL = "AAPL"
START_DATE = "2015-01-01"
FORECAST_DAYS = 180  # 3–6 months
RISK_FREE_RATE = 0.05

import yfinance as yf

def fetch_stock_data(symbol, start_date):
    data = yf.download(symbol, start=start_date)
    data.reset_index(inplace=True)
    return data

def preprocess_data(df):
    """
    Converts raw stock data into Prophet-compatible format
    """
    processed_df = df[["Date", "Close"]].copy()
    processed_df.columns = ["ds", "y"]
    return processed_df

def calculate_returns(df):
    """
    Calculates daily and cumulative returns
    """
    df = df.copy()
    df["daily_return"] = df["y"].pct_change()
    df["cumulative_return"] = (1 + df["daily_return"]).cumprod() - 1
    return df

from prophet import Prophet

def build_model():
    model = Prophet(
        yearly_seasonality=True,
        weekly_seasonality=True,
        daily_seasonality=False,
        changepoint_prior_scale=0.05
    )
    return model

def train_model(model, data):
    model.fit(data)
    return model

def make_prediction(model, periods):
    future = model.make_future_dataframe(periods=periods)
    forecast = model.predict(future)
    return forecast

import numpy as np

def volatility(returns, trading_days=252):
    return np.std(returns) * np.sqrt(trading_days)

def sharpe_ratio(returns, risk_free_rate):
    excess_returns = returns - (risk_free_rate / 252)
    return (np.mean(excess_returns) / np.std(excess_returns)) * np.sqrt(252)

def value_at_risk(returns, confidence_level=0.95):
    return np.percentile(returns, (1 - confidence_level) * 100)

def max_drawdown(cumulative_returns):
    peak = cumulative_returns.cummax()
    drawdown = (cumulative_returns - peak) / peak
    return drawdown.min()


def evaluate_performance(df, risk_free_rate):
    returns = df["daily_return"].dropna()

    results = {
        "Expected Return (%)": round(df["cumulative_return"].iloc[-1] * 100, 2),
        "Volatility (%)": round(volatility(returns) * 100, 2),
        "Sharpe Ratio": round(sharpe_ratio(returns, risk_free_rate), 2),
        "Value at Risk (95%) (%)": round(value_at_risk(returns) * 100, 2),
        "Max Drawdown (%)": round(max_drawdown(df["cumulative_return"]) * 100, 2),
    }

    return results

import matplotlib.pyplot as plt

def main():
    print(f"\n📥 Fetching data for {STOCK_SYMBOL}...")
    raw_data = fetch_stock_data(STOCK_SYMBOL, START_DATE)

    print("🧹 Preprocessing data...")
    processed_data = preprocess_data(raw_data)

    print("📈 Calculating return indicators...")
    processed_data = calculate_returns(processed_data)

    print("🤖 Training forecasting model...")
    model = build_model()
    trained_model = train_model(model, processed_data[["ds", "y"]])

    print("🔮 Generating forecast...")
    forecast = make_prediction(trained_model, FORECAST_DAYS)

    print("\n📊 Risk & Return Metrics")
    print("-" * 35)
    metrics = evaluate_performance(processed_data, RISK_FREE_RATE)
    for key, value in metrics.items():
        print(f"{key}: {value}")

    print("\n📉 Plotting forecast...")
    trained_model.plot(forecast)
    plt.title(f"{STOCK_SYMBOL} Price Forecast (Next 3–6 Months)")
    plt.xlabel("Date")
    plt.ylabel("Price")
    plt.show()


if __name__ == "__main__":
    main()
