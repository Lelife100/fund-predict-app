import os
import json
import numpy as np
import pandas as pd
import requests
from datetime import datetime
from flask import Flask, jsonify, send_from_directory
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, precision_score, recall_score

app = Flask(__name__, static_folder='.', static_url_path='')

FUND_CODE = "025209"
PRED_DAYS = 5
BUY_THRESHOLD = 0.6
SELL_THRESHOLD = 0.5
HOLD_DAYS = 5

def fetch_nav_data():
    csv_file = "025209_netvalue.csv"   # 只看这个文件
    if not os.path.exists(csv_file):
        print(f"找不到文件: {csv_file}")
        return None

    try:
        df = pd.read_csv(csv_file, header=None, encoding="utf-8")
        df = df.iloc[:, :3]
        df.columns = ["date", "nav", "acc_nav"]
        for col in ["nav", "acc_nav"]:
            df[col] = df[col].astype(str).str.replace(",", "").str.replace("%", "")
            df[col] = pd.to_numeric(df[col], errors="coerce")
        df["date"] = pd.to_datetime(df["date"], errors="coerce")
        df.dropna(subset=["date", "nav"], inplace=True)
        df = df.sort_values("date").reset_index(drop=True)
        print(f"✅ CSV 加载成功，{len(df)} 条记录")
        return df
    except Exception as e:
        print(f"CSV 解析失败: {e}")
        return None

def create_features(df):
    df = df.copy()
    df["return_1d"] = df["nav"].pct_change()
    df["return_5d"] = df["nav"].pct_change(5)
    df["return_10d"] = df["nav"].pct_change(10)
    df["ma_5"] = df["nav"].rolling(5).mean()
    df["ma_10"] = df["nav"].rolling(10).mean()
    df["ma_20"] = df["nav"].rolling(20).mean()
    df["volatility_5"] = df["return_1d"].rolling(5).std()
    df["volatility_10"] = df["return_1d"].rolling(10).std()
    df["future_return"] = df["nav"].shift(-PRED_DAYS) / df["nav"] - 1
    df["target"] = (df["future_return"] > 0).astype(int)
    return df.dropna()

def train_and_backtest(df):
    feature_cols = ["return_1d","return_5d","return_10d","ma_5","ma_10","ma_20","volatility_5","volatility_10"]
    df_model = create_features(df)
    if len(df_model) < 100:
        return None, None, None, None
    split = int(len(df_model)*0.8)
    train, test = df_model.iloc[:split], df_model.iloc[split:].copy()
    X_tr, y_tr = train[feature_cols].values, train["target"].values
    X_te, y_te = test[feature_cols].values, test["target"].values
    scaler = StandardScaler(); X_tr = scaler.fit_transform(X_tr); X_te = scaler.transform(X_te)
    model = LogisticRegression(max_iter=1000).fit(X_tr, y_tr)
    test["prob"] = model.predict_proba(X_te)[:,1]
    test["pred"] = (test["prob"]>=0.5).astype(int)
    acc = accuracy_score(y_te, test["pred"])
    prec = precision_score(y_te, test["pred"], zero_division=0)
    rec = recall_score(y_te, test["pred"], zero_division=0)
    cap, s_nav, b_nav = 1.0, [1.0], [1.0]
    pos, buy_d, buy_n, hc, trades = False, None, None, 0, []
    dates, probs, navs = test["date"].values, test["prob"].values, test["nav"].values
    for i,(d,p,n) in enumerate(zip(dates,probs,navs)):
        if i==0: base = n
        b_nav.append(n/base)
        if not pos:
            if p>=BUY_THRESHOLD: pos,buy_d,buy_n,hc = True,d,n,0
        else:
            hc+=1
            if hc>=HOLD_DAYS or p<SELL_THRESHOLD:
                ret = (n-buy_n)/buy_n
                cap *= (1+ret)
                trades.append({"buy_date":buy_d.strftime("%Y-%m-%d"),"sell_date":d.strftime("%Y-%m-%d"),
                               "hold_days":hc,"buy_nav":round(buy_n,4),"sell_nav":round(n,4),"return_pct":round(ret,4)})
                pos=False
        s_nav.append(cap*(n/buy_n) if pos and buy_n else cap)
    s_nav, b_nav = np.array(s_nav), np.array(b_nav)
    tot_ret = cap-1
    days = (dates[-1]-dates[0]).days
    ann_ret = (1+tot_ret)**(365/days)-1 if days>0 else 0
    mdd = (s_nav - np.maximum.accumulate(s_nav)).min() / np.maximum.accumulate(s_nav)[0]
    win = sum(1 for t in trades if t["return_pct"]>0)
    met = {"accuracy":acc,"precision":prec,"recall":rec,"strategy_return":tot_ret,
           "annual_return":ann_ret,"max_drawdown":mdd,"win_rate":win/len(trades) if trades else 0,
           "total_trades":len(trades)}
    eq = {"dates":[d.strftime("%Y-%m-%d") for d in dates],"strategy_nav":s_nav.tolist(),"buyhold_nav":b_nav.tolist()}
    ph = {"dates":[d.strftime("%Y-%m-%d") for d in test["date"].iloc[-90:]],
          "probabilities":test["prob"].iloc[-90:].tolist()}
    return model, scaler, met, trades, eq, ph, test.iloc[-1]

def generate_signal(model, scaler, latest):
    cols = ["return_1d","return_5d","return_10d","ma_5","ma_10","ma_20","volatility_5","volatility_10"]
    if latest is None or latest[cols].isnull().any():
        return {"action_class":"hold","icon":"⏸️","title":"数据不足","probability":None,"subtitle":""}
    X = latest[cols].values.reshape(1,-1)
    prob = model.predict_proba(scaler.transform(X))[0,1]
    if prob>=BUY_THRESHOLD: a,i,t = "buy","🟢","建议买入"
    elif prob<SELL_THRESHOLD: a,i,t = "sell","🔴","建议卖出"
    else: a,i,t = "hold","🟡","继续观望"
    return {"action_class":a,"icon":i,"title":t,"probability":prob,
            "subtitle":f"未来{PRED_DAYS}日上涨概率: {prob*100:.1f}%"}

@app.route('/api/all')
def api_all():
    df = fetch_nav_data()
    if df is None or len(df)<120:
        return jsonify({"error":"数据不足，请确保 025209_netvalue.csv 已上传到根目录"})
    res = train_and_backtest(df)
    if res[0] is None:
        return jsonify({"error":"训练数据量不足"})
    model, scaler, met, trades, eq, ph, latest = res
    sig = generate_signal(model, scaler, latest)
    return jsonify({"signal":sig,"metrics":met,"equity_curve":eq,"prob_history":ph,"trades":trades,
                    "data_start":df["date"].min().strftime("%Y-%m-%d"),
                    "data_end":df["date"].max().strftime("%Y-%m-%d"),"n_samples":len(df)})

@app.route('/')
def index():
    return send_from_directory('.', 'index.html')

if __name__ == "__main__":
    port = int(os.environ.get("PORT",5000))
    app.run(host="0.0.0.0",port=port)
