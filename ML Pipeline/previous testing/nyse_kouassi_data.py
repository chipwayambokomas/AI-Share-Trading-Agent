import yfinance as yf
import pandas as pd

ticker = '^NYA'
start_date = '1965-12-31'
end_date = '2019-11-15'

# Add threads=False to improve stability
data = yf.download(ticker, start=start_date, end=end_date, threads=False)

# Check if data is empty
if data.empty:
    print("Download failed or returned empty data.")
else:
    data.to_csv('nyse_composite_1965_2019.csv')
    print(data.head())
