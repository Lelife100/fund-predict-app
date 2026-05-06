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

# ====== 数据读取（本地 CSV 优先，适配东方财富无表头格式） ======
def fetch_nav_data():
    # 尝试多个常用文件名
    possible_files = ["025209_netvalue.csv", "nav_data.csv", "history.csv"]
    csv_file = None
    for f in possible_files:
        if os.path.exists(f):
            csv_file = f
            break

    if csv_file:
        try:
            # 东方财富导出默认无表头，列顺序：日期, 单位净值, 累计净值, ...
            df = pd.read_csv(csv_file, header=None, encoding="utf-8")
            # 只取前三列
            df = df.iloc[:, :3]
            df.columns = ["date", "nav", "acc_nav"]
            # 清洗数值
            for col in ["nav", "acc_nav"]:
                df[col] = df[col].astype(str).str.replace(",", "").str.replace("%", "")
                df[col] = pd.to_numeric(df[col], errors="coerce")
            df["date"] = pd.to_datetime(df["date"], errors="coerce")
            df.dropna(subset=["date", "nav"], inplace=True)
            df = df.sort_values("date").reset_index(drop=True)
            print(f"✅ 从本地CSV加载数据成功，共 {len(df)} 条")
            return df
        except Exception as e:
            print(f"读取CSV失败: {e}")

    # 备用网络请求（如果CSV不存在）
    try:
        url = "https://api.fund.eastmoney.com/f10/lsjz"
        params = {
            "fundCode": FUND_CODE,
            "pageIndex": 1,
            "pageSize": 2000,
            "startDate": "2021-01-01",
            "endDate": datetime.now().strftime("%Y-%m-%d"),
            "_": int(datetime.now().timestamp() * 1000)
        }
        headers = {
            "User-Agent": "Mozilla/5.0",
            "Referer": "https://fundf10.eastmoney.com/",
        }
        resp = requests.get(url, params=params, headers=headers, timeout=10)
        data_text = resp.text
        if "jsonpgz" in data_text or "jQuery" in data_text:
            start = data_text.find("{")
            end = data_text.rfind("}")+1
            json_str = data_text[start:end]
        else:
            json_str = data_text
        data = json.loads(json_str)
        items = data.get("Data", {}).get("LSJZList", [])
        records = []
        for item in items:
            records.append({
                "date": item["FSRQ"],
                "nav": float(item["DWJZ"]),
                "acc_nav": float(item["LJJZ"])
            })
        df = pd.DataFrame(records)
        df["date"] = pd.to_datetime(df["date"])
        df = df.sort_values("date").reset_index(drop=True)
        print("网络数据获取成功")
        return df
    except Exception as e:
        print(f"网络也获取失败: {e}")
        return None

# ====== 特征与模型代码完全相同（略，可直接沿用你之前的版本） ======
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
    feature_cols = ["return_1d", "return_5d", "return_10d",
                    "ma_5", "ma_10", "ma_20", "volatility_5", "volatility_10"]
    df_model = create_features(df)
    if len(df_model) < 100:
        return None, None, None, None
    split_idx = int(len(df_model) * 0.8)
    train = df_model.iloc[:split_idx]
    test = df_model.iloc[split_idx:].copy()
    X_train = train[feature_cols].values
    y_train = train["target"].values
    X_test = test[feature_cols].values
    y_test = test["target"].values
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)
    model = LogisticRegression(max_iter=1000)
    model.fit(X_train, y_train)
    test["prob_up"] = model.predict_proba(X_test)[:, 1]
    test["pred"] = (test["prob_up"] >= 0.5).astype(int)
    accuracy = accuracy_score(y_test, test["pred"])
    precision = precision_score(y_test, test["pred"], zero_division=0)
    recall = recall_score(y_test, test["pred"], zero_division=0)
    capital = 1.0
    strategy_nav = [1.0]
    buyhold_nav = [1.0]
    in_position = False
    buy_date = buy_nav = None
    hold_counter = 0
    trades = []
    dates = test["date"].values
    probs = test["prob_up"].values
    navs = test["nav"].values
    for i, (date, prob, nav) in enumerate(zip(dates, probs, navs)):
        if i == 0:
            buyhold_base_nav = nav
        buyhold_nav.append(nav / buyhold_base_nav)
        if not in_position:
            if prob >= BUY_THRESHOLD:
                in_position = True
                buy_date = date
                buy_nav = nav
                hold_counter = 0
        else:
            hold_counter += 1
            if hold_counter >= HOLD_DAYS or prob < SELL_THRESHOLD:
                sell_nav = nav
                ret = (sell_nav - buy_nav) / buy_nav
                capital *= (1 + ret)
                trades.append({
                    "buy_date": buy_date.strftime("%Y-%m-%d"),
                    "sell_date": date.strftime("%Y-%m-%d"),
                    "hold_days": hold_counter,
                    "buy_nav": round(buy_nav, 4),
                    "sell_nav": round(sell_nav, 4),
                    "return_pct": round(ret, 4)
                })
                in_position = False
        if in_position and buy_nav is not None:
            strategy_nav.append(capital * (nav / buy_nav))
        else:
            strategy_nav.append(capital)
    strategy_nav = np.array(strategy_nav)
    buyhold_nav = np.array(buyhold_nav)
    total_return = capital - 1
    days_total = (dates[-1] - dates[0]).days
    annual_return = (1 + total_return) ** (365.0 / days_total) - 1 if days_total > 0 else 0
    peak = np.maximum.accumulate(strategy_nav)
    max_drawdown = (strategy_nav - peak).min() / peak[0]
    win_trades = sum(1 for t in trades if t["return_pct"] > 0)
    win_rate = win_trades / len(trades) if trades else 0
    metrics = {
        "accuracy": accuracy, "precision": precision, "recall": recall,
        "strategy_return": total_return, "annual_return": annual_return,
        "max_drawdown": max_drawdown, "win_rate": win_rate,
        "total_trades": len(trades)
    }
    eq = {
        "dates": [d.strftime("%Y-%m-%d") for d in dates],
        "strategy_nav": strategy_nav.tolist(),
        "buyhold_nav": buyhold_nav.tolist()
    }
    prob_hist = {
        "dates": [d.strftime("%Y-%m-%d") for d in test["date"].iloc[-90:]],
        "probabilities": test["prob_up"].iloc[-90:].tolist()
    }
    return model, scaler, metrics, trades, eq, prob_hist, test.iloc[-1] if len(test)>0 else None

def generate_signal(model, scaler, latest):
    cols = ["return_1d","return_5d","return_10d","ma_5","ma_10","ma_20","volatility_5","volatility_10"]
    if latest is None or latest[cols].isnull().any():
        return {"action_class":"hold","icon":"⏸️","title":"数据不足","probability":None,"subtitle":""}
    X = latest[cols].values.reshape(1, -1)
    prob = model.predict_proba(scaler.transform(X))[0, 1]
    if prob >= BUY_THRESHOLD:
        a, i, t = "buy", "🟢", "建议买入"
    elif prob < SELL_THRESHOLD:
        a, i, t = "sell", "🔴", "建议卖出"
    else:
        a, i, t = "hold", "🟡", "继续观望"
    return {"action_class":a,"icon":i,"title":t,"probability":prob,"subtitle":f"未来{PRED_DAYS}日上涨概率: {prob*100:.1f}%"}

@app.route('/api/all')
def api_all():
    df = fetch_nav_data()
    if df is None or len(df) < 120:
        return jsonify({"error": "数据不足，请确认 025209_netvalue.csv 已上传"})
    result = train_and_backtest(df)
    if result[0] is None:
        return jsonify({"error": "训练数据量不足"})
    model, scaler, metrics, trades, eq_curve, prob_hist, latest_row = result
    signal = generate_signal(model, scaler, latest_row)
    return jsonify({
        "signal": signal,
        "metrics": metrics,
        "equity_curve": eq_curve,
        "prob_history": prob_hist,
        "trades": trades,
        "data_start": df["date"].min().strftime("%Y-%m-%d"),
        "data_end": df["date"].max().strftime("%Y-%m-%d"),
        "n_samples": len(df)
    })

@app.route('/')
def index():
    return send_from_directory('.', 'index.html')

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port)
