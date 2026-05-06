import os
import json
import numpy as np
import pandas as pd
import requests
from datetime import datetime, timedelta
from flask import Flask, jsonify, send_from_directory
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, precision_score, recall_score

app = Flask(__name__, static_folder='.', static_url_path='')

# ====== 配置参数 ======
FUND_CODE = "025209"
FUND_NAME = "永赢先锋半导体智选混合发起C"
LOOKBACK_DAYS = 60       # 模型使用的历史天数
PRED_DAYS = 5            # 预测未来几天
BUY_THRESHOLD = 0.6
SELL_THRESHOLD = 0.5
HOLD_DAYS = 5

# ====== 数据爬取（东方财富 API） ======
def fetch_nav_data():
    """获取基金历史净值"""
    url = "https://api.fund.eastmoney.com/f10/lsjz"
    params = {
        "callback": "jQuery",
        "fundCode": FUND_CODE,
        "pageIndex": 1,
        "pageSize": 1000,
        "startDate": "2021-01-01",
        "endDate": datetime.now().strftime("%Y-%m-%d"),
        "_": int(datetime.now().timestamp() * 1000)
    }
    headers = {
        "User-Agent": "Mozilla/5.0",
        "Referer": "http://fund.eastmoney.com/",
    }
    try:
        resp = requests.get(url, params=params, headers=headers, timeout=15)
        # 处理 JSONP
        data_text = resp.text
        if "jQuery" in data_text:
            json_str = data_text[data_text.index("(")+1:data_text.rindex(")")]
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
        df = df.sort_values("date", ascending=True).reset_index(drop=True)
        return df
    except Exception as e:
        print(f"数据获取失败: {e}")
        return None

# ====== 特征工程 ======
def create_features(df):
    """构造预测特征（滚动窗口）"""
    df = df.copy()
    df["return_1d"] = df["nav"].pct_change()
    df["return_5d"] = df["nav"].pct_change(5)
    df["return_10d"] = df["nav"].pct_change(10)
    df["ma_5"] = df["nav"].rolling(5).mean()
    df["ma_10"] = df["nav"].rolling(10).mean()
    df["ma_20"] = df["nav"].rolling(20).mean()
    df["volatility_5"] = df["return_1d"].rolling(5).std()
    df["volatility_10"] = df["return_1d"].rolling(10).std()
    # 未来 PRED_DAYS 天的累计收益（标签）
    df["future_return"] = df["nav"].shift(-PRED_DAYS) / df["nav"] - 1
    df["target"] = (df["future_return"] > 0).astype(int)
    return df.dropna()

# ====== 训练模型 & 回测 ======
def train_and_backtest(df):
    """训练逻辑回归，同时返回回测指标和交易记录"""
    feature_cols = ["return_1d", "return_5d", "return_10d",
                    "ma_5", "ma_10", "ma_20", "volatility_5", "volatility_10"]
    df_model = create_features(df)
    if len(df_model) < 100:
        return None, None, None, None

    # 划分训练/测试（这里使用滚动窗口，最后一整段做回测更加合理）
    # 简单起见，用前80%数据训练，后20%回测
    split_idx = int(len(df_model) * 0.8)
    train = df_model.iloc[:split_idx]
    test = df_model.iloc[split_idx:].copy()

    X_train = train[feature_cols].values
    y_train = train["target"].values
    X_test = test[feature_cols].values
    y_test = test["target"].values

    # 标准化
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)

    # 训练
    model = LogisticRegression(max_iter=1000)
    model.fit(X_train, y_train)

    # 预测概率
    test["prob_up"] = model.predict_proba(X_test)[:, 1]
    test["pred"] = (test["prob_up"] >= 0.5).astype(int)

    # 分类指标
    accuracy = accuracy_score(y_test, test["pred"])
    precision = precision_score(y_test, test["pred"], zero_division=0)
    recall = recall_score(y_test, test["pred"], zero_division=0)

    # ====== 回测策略 ======
    capital = 1.0          # 初始净值
    strategy_nav = [1.0]
    buyhold_nav = [1.0]
    in_position = False
    buy_date = None
    buy_nav = None
    trades = []
    hold_counter = 0

    # 需要从原始 df 中提取净值序列，以便计算持有期收益
    # 这里我们使用 test 对应的原始净值（注意 test 索引对应原始 df）
    original_navs = df.set_index("date")["nav"]

    dates = test["date"].values
    prob_ups = test["prob_up"].values
    navs = test["nav"].values

    for i, (date, prob, nav) in enumerate(zip(dates, prob_ups, navs)):
        # 买入持有净值（以第一天为基准）
        if i == 0:
            buyhold_base_nav = nav
        buyhold_nav.append(nav / buyhold_base_nav)

        # 策略逻辑
        if not in_position:
            if prob >= BUY_THRESHOLD:
                # 买入
                in_position = True
                buy_date = date
                buy_nav = nav
                hold_counter = 0
        else:
            hold_counter += 1
            # 卖出条件：持仓满5天 或 预测概率低于卖出阈值
            if hold_counter >= HOLD_DAYS or prob < SELL_THRESHOLD:
                # 卖出
                sell_nav = nav
                return_pct = (sell_nav - buy_nav) / buy_nav
                capital *= (1 + return_pct)
                trades.append({
                    "buy_date": buy_date.strftime("%Y-%m-%d"),
                    "sell_date": date.strftime("%Y-%m-%d"),
                    "hold_days": hold_counter,
                    "buy_nav": round(buy_nav, 4),
                    "sell_nav": round(sell_nav, 4),
                    "return_pct": round(return_pct, 4)
                })
                in_position = False
                buy_date = None
                buy_nav = None
                hold_counter = 0

        # 当天策略净值（如果持仓，净值跟随买入成本计算当日浮动）
        if in_position and buy_nav is not None:
            strategy_nav.append(capital * (nav / buy_nav))
        else:
            strategy_nav.append(capital)

    # 如果回测结束时还持仓，不强制平仓（按最后一天净值计算）
    # 这里简单处理，直接沿用最后净值
    strategy_nav = np.array(strategy_nav)
    buyhold_nav = np.array(buyhold_nav)

    # 计算更多指标
    total_return = capital - 1
    # 年化收益（按回测期间实际天数计算）
    days_total = (dates[-1] - dates[0]).days
    annual_return = (1 + total_return) ** (365.0 / days_total) - 1 if days_total > 0 else 0

    # 最大回撤
    peak = np.maximum.accumulate(strategy_nav)
    drawdown = (strategy_nav - peak) / peak
    max_drawdown = drawdown.min()

    # 胜率
    win_trades = [t for t in trades if t["return_pct"] > 0]
    win_rate = len(win_trades) / len(trades) if trades else 0

    metrics = {
        "accuracy": accuracy,
        "precision": precision,
        "recall": recall,
        "strategy_return": total_return,
        "annual_return": annual_return,
        "max_drawdown": max_drawdown,
        "win_rate": win_rate,
        "total_trades": len(trades)
    }

    # 准备曲线数据
    equity_curve = {
        "dates": [d.strftime("%Y-%m-%d") for d in dates],
        "strategy_nav": strategy_nav.tolist(),
        "buyhold_nav": buyhold_nav.tolist()
    }

    # 概率历史（最近90天）
    prob_df = test.iloc[-90:]
    prob_history = {
        "dates": [d.strftime("%Y-%m-%d") for d in prob_df["date"]],
        "probabilities": prob_df["prob_up"].tolist()
    }

    return model, scaler, metrics, trades, equity_curve, prob_history, test.iloc[-1] if len(test) > 0 else None

# ====== 生成最新信号 ======
def generate_signal(model, scaler, latest_data):
    """根据最新数据生成当日信号"""
    feature_cols = ["return_1d", "return_5d", "return_10d",
                    "ma_5", "ma_10", "ma_20", "volatility_5", "volatility_10"]
    if latest_data is None or latest_data[feature_cols].isnull().any():
        return {
            "action_class": "hold",
            "icon": "⏸️",
            "title": "数据不足，无法判断",
            "probability": None,
            "subtitle": "请等待更多历史数据"
        }

    X = latest_data[feature_cols].values.reshape(1, -1)
    X_scaled = scaler.transform(X)
    prob = model.predict_proba(X_scaled)[0, 1]

    # 假设当前没有持仓（真实场景需记录持仓状态）
    # 这里我们只演示预测概率，不自动判断买卖（用户可以自己看提示）
    if prob >= BUY_THRESHOLD:
        action = "buy"
        icon = "🟢"
        title = "建议买入"
    elif prob < SELL_THRESHOLD:
        action = "sell"
        icon = "🔴"
        title = "建议卖出"
    else:
        action = "hold"
        icon = "🟡"
        title = "继续观望"

    return {
        "action_class": action,
        "icon": icon,
        "title": title,
        "probability": prob,
        "subtitle": f"未来{PRED_DAYS}个交易日上涨概率: {prob*100:.1f}%"
    }

# ====== 主 API 接口 ======
@app.route('/api/all')
def api_all():
    df = fetch_nav_data()
    if df is None or len(df) < 120:
        return jsonify({"error": "无法获取足够的历史数据"})

    result = train_and_backtest(df)
    if result[0] is None:
        return jsonify({"error": "训练数据不足"})

    model, scaler, metrics, trades, equity_curve, prob_history, latest_row = result

    # 生成最新信号
    signal = generate_signal(model, scaler, latest_row)

    return jsonify({
        "signal": signal,
        "metrics": metrics,
        "equity_curve": equity_curve,
        "prob_history": prob_history,
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
