# data.py
# אחראי למשיכת נתונים מ-yfinance והחזרת DataFrame מוכן לשימוש

from __future__ import annotations
import yfinance as yf
import pandas as pd

def download(symbol: str, period: str, interval: str) -> pd.DataFrame:
    # מושך נתונים מיואהו ומסדר שמות עמודות
    df = yf.download(symbol, period=period, interval=interval, auto_adjust=False)
    if df.empty:
        raise RuntimeError("Empty dataframe from yfinance")
    df = df.rename(columns={
        "Open":"open","High":"high","Low":"low","Close":"close","Adj Close":"adj_close","Volume":"volume"
    })
    # דואג לאינדקס עם אזור זמן Asia/Jerusalem
    if df.index.tz is None:
        df.index = df.index.tz_localize("UTC")
    df.index = df.index.tz_convert("Asia/Jerusalem")
    return df
