import yfinance as yf
import requests


# Retrieve name based on ticker
def get_name(ticker):
    url = f"https://api.polygon.io/v3/reference/tickers/{ticker.upper()}?apiKey=VFSwKNWbH7pv7Yp98ayguccA6KVAJYjr"
    request_json = requests.get(url).json()

    return request_json["results"]["name"]


