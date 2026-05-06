import os
import numpy as np
import pandas as pd
from datetime import datetime
from flask import Flask, jsonify, send_from_directory
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, precision_score, recall_score

app = Flask(__name__, static_folder='.', static_url_path='')

PRED_DAYS = 5
BUY_THRESHOLD = 0.6
SELL_THRESHOLD = 0.5
HOLD_DAYS = 5

def fetch_nav_data():
    if not os.path.exists("data.csv"):
        return None
    # 关键修改：用空白分隔符（兼容空格、制表符）
    df = pd.read_csv("data.csv", header=None, encoding="utf-8", sep=r'\s+')
    # 可能有些行末尾有空白导致读取了空列，只取前三列
    df = df.iloc[:, :3]
    df.columns = ["date", "nav", "acc_nav"]
    for col in ["nav", "acc_nav"]:
        df[col] = df[col].astype(str).str.replace(",", "", regex=True).str.replace("%", "", regex=True)
        df[col] = pd.to_numeric(df[col], errors="coerce")
    df["date"] = pd.to_datetime(df["date"], errors="coerce")
    df.dropna(subset=["date", "nav"], inplace=True)
    df = df.sort_values("date").reset_index(drop=True)
    return df

def create_features(df):
    df = df.copy()
    df["r1"] = df["nav"].pct_change()
    df["r5"] = df["nav"].pct_change(5)
    df["r10"] = df["nav"].pct_change(10)
    df["ma5"] = df["nav"].rolling(5).mean()
    df["ma10"] = df["nav"].rolling(10).mean()
    df["ma20"] = df["nav"].rolling(20).mean()
    df["vol5"] = df["r1"].rolling(5).std()
    df["vol10"] = df["r1"].rolling(10).std()
    df["target"] = (df["nav"].shift(-PRED_DAYS) / df["nav"] - 1 > 0).astype(int)
    return df.dropna()

def train_and_backtest(df):
    feats = ["r1","r5","r10","ma5","ma10","ma20","vol5","vol10"]
    dfm = create_features(df)
    if len(dfm) < 120:
        return None
    split = int(len(dfm)*0.8)
    tr, te = dfm.iloc[:split], dfm.iloc[split:].copy()
    sc = StandardScaler()
    Xtr = sc.fit_transform(tr[feats].values)
    Xte = sc.transform(te[feats].values)
    mdl = LogisticRegression(max_iter=1000).fit(Xtr, tr["target"].values)
    te["prob"] = mdl.predict_proba(Xte)[:,1]
    te["pred"] = (te["prob"]>=0.5).astype(int)
    acc = accuracy_score(te["target"], te["pred"])
    prec = precision_score(te["target"], te["pred"], zero_division=0)
    rec = recall_score(te["target"], te["pred"], zero_division=0)

    cap, s_nav, b_nav = 1.0, [1.0], [1.0]
    pos, bd, bn, hc, trades = False, None, None, 0, []
    for i,(d,p,n) in enumerate(zip(te["date"].values, te["prob"], te["nav"])):
        if i==0: base=n
        b_nav.append(n/base)
        if not pos:
            if p>=BUY_THRESHOLD: pos,bd,bn,hc = True,d,n,0
        else:
            hc+=1
            if hc>=HOLD_DAYS or p<SELL_THRESHOLD:
                ret = (n-bn)/bn
                cap *= (1+ret)
                trades.append({"buy_date":bd.strftime("%Y-%m-%d"),"sell_date":d.strftime("%Y-%m-%d"),
                               "hold_days":hc,"buy_nav":round(bn,4),"sell_nav":round(n,4),"return_pct":round(ret,4)})
                pos=False
        s_nav.append(cap*(n/bn) if pos and bn else cap)
    s_nav, b_nav = np.array(s_nav), np.array(b_nav)
    ret_total = cap-1
    days = (te["date"].iloc[-1] - te["date"].iloc[0]).days
    ret_ann = (1+ret_total)**(365.0/days)-1 if days>0 else 0
    mdd = (s_nav - np.maximum.accumulate(s_nav)).min() / np.maximum.accumulate(s_nav)[0]
    win_rate = sum(1 for t in trades if t["return_pct"]>0) / len(trades) if trades else 0

    metrics = {"accuracy":acc,"precision":prec,"recall":rec,"strategy_return":ret_total,
               "annual_return":ret_ann,"max_drawdown":mdd,"win_rate":win_rate,"total_trades":len(trades)}
    eq = {"dates":[d.strftime("%Y-%m-%d") for d in te["date"]],
          "strategy_nav":s_nav.tolist(),"buyhold_nav":b_nav.tolist()}
    ph = {"dates":[d.strftime("%Y-%m-%d") for d in te["date"].iloc[-90:]],
          "probabilities":te["prob"].iloc[-90:].tolist()}
    return mdl, sc, metrics, trades, eq, ph, te.iloc[-1]

def generate_signal(mdl, sc, latest):
    feats = ["r1","r5","r10","ma5","ma10","ma20","vol5","vol10"]
    if latest is None or latest[feats].isnull().any():
        return {"action_class":"hold","icon":"⏸️","title":"数据不足","probability":None,"subtitle":""}
    prob = mdl.predict_proba(sc.transform(latest[feats].values.reshape(1,-1)))[0,1]
    if prob>=BUY_THRESHOLD: a,i,t = "buy","🟢","建议买入"
    elif prob<SELL_THRESHOLD: a,i,t = "sell","🔴","建议卖出"
    else: a,i,t = "hold","🟡","继续观望"
    return {"action_class":a,"icon":i,"title":t,"probability":prob,
            "subtitle":f"未来{PRED_DAYS}日上涨概率: {prob*100:.1f}%"}

@app.route('/api/all')
def api_all():
    try:
        df = fetch_nav_data()
        if df is None or len(df)<100:
            return jsonify({"error":"找不到 data.csv，请确认文件已上传"})
        res = train_and_backtest(df)
        if res is None:
            return jsonify({"error":"数据量不足，至少需要120个交易日"})
        mdl, sc, met, trades, eq, ph, latest = res
        sig = generate_signal(mdl, sc, latest)
        return jsonify({"signal":sig,"metrics":met,"equity_curve":eq,"prob_history":ph,"trades":trades,
                        "data_start":df["date"].min().strftime("%Y-%m-%d"),
                        "data_end":df["date"].max().strftime("%Y-%m-%d"),"n_samples":len(df)})
    except Exception as e:
        return jsonify({"error": f"服务器内部错误：{str(e)}"})

@app.route('/')
def index():
    return send_from_directory('.', 'index.html')

if __name__ == "__main__":
    port = int(os.environ.get("PORT",5000))
    app.run(host="0.0.0.0", port=port)
