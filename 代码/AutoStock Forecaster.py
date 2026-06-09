import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import xgboost as xgb
import lightgbm as lgb
import matplotlib.pyplot as plt
import os
import shutil
import warnings
warnings.filterwarnings('ignore')

# ===== 解决 matplotlib 中文显示问题 =====
plt.rcParams['font.sans-serif'] = ['SimHei', 'Arial Unicode MS', 'DejaVu Sans', 'Bitstream Vera Sans',
                                   'Lucida Grande', 'Verdana', 'Geneva', 'Lucid',
                                   'Arial', 'Helvetica', 'Avant Garde', 'sans-serif']
plt.rcParams['axes.unicode_minus'] = False  # 正常显示负号
# =====================================

# 时间序列与深度学习导入
from statsmodels.tsa.statespace.sarimax import SARIMAX
import torch
import torch.nn as nn

# 文件名列表
file_names = [
    "Tesla Stock Price History.csv",
    "Ford Motor Stock Price History.csv",
    "Toyota Motor Stock Price History.csv",
    "BYD A Stock Price History.csv",
    "Mercedes Benz Group Stock Price History.csv"
]

# 创建文件夹
original_dir = "原数据集"
processed_dir = "新数据集"
os.makedirs(original_dir, exist_ok=True)
os.makedirs(processed_dir, exist_ok=True)

# 将原始文件移动到“原数据集”（仅第一次运行时）
for file_name in file_names:
    if os.path.exists(file_name) and not os.path.exists(os.path.join(original_dir, file_name)):
        shutil.move(file_name, os.path.join(original_dir, file_name))
        print(f"已将原始文件移动到 {original_dir}: {file_name}")

# 更新路径
file_paths = [os.path.join(original_dir, fname) for fname in file_names]

# ==================== LSTM 模型定义 ====================
class LSTMModel(nn.Module):
    def __init__(self, input_size=1, hidden_size=50, num_layers=1, output_size=1):
        super(LSTMModel, self).__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size, output_size)
    
    def forward(self, x):
        out, _ = self.lstm(x)
        out = self.fc(out[:, -1, :])  # 取最后一个时间步
        return out

# ==================== 数据预处理函数 ====================
def preprocess_stock_data(file_path):
    company_name = os.path.basename(file_path).split(' Stock')[0]
    print(f"\n=== Processing {company_name} ===")
    
    df = pd.read_csv(file_path)
    print("Original shape:", df.shape)
    
    df.columns = df.columns.str.strip()
    
    df['Date'] = pd.to_datetime(df['Date'], format='%m/%d/%Y', errors='coerce')
    df = df.sort_values('Date').reset_index(drop=True)
    
    price_cols = ['Price', 'Open', 'High', 'Low']
    for col in price_cols:
        df[col] = df[col].astype(str).str.replace(',', '').astype(float)
    
    if 'Vol.' in df.columns:
        def parse_vol(v):
            if pd.isna(v): return np.nan
            v = str(v).strip().upper()
            if 'M' in v: return float(v.replace('M', ''))
            elif v in ['-', '']: return np.nan
            else:
                try: return float(v) / 1e6
                except: return np.nan
        df['Vol.'] = df['Vol.'].apply(parse_vol)
    
    if 'Change %' in df.columns:
        df['Change %'] = df['Change %'].astype(str).str.replace('%', '').astype(float) / 100
    
    # 处理缺失值
    for col in price_cols:
        df[col] = df[col].ffill().bfill()
    if 'Vol.' in df.columns:
        df['Vol.'] = df['Vol.'].fillna(df['Vol.'].median())
    df['Change %'] = df['Price'].pct_change().fillna(0)
    
    # 异常值 Winsorizing
    def winsorize_series(series):
        Q1, Q3 = series.quantile(0.25), series.quantile(0.75)
        IQR = Q3 - Q1
        lower, upper = Q1 - 1.5*IQR, Q3 + 1.5*IQR
        return np.clip(series, lower, upper)
    for col in price_cols:
        df[col] = winsorize_series(df[col])
    
    # 标准化与归一化（仅 Price）
    scaler_std = StandardScaler()
    df['Price_Standardized'] = scaler_std.fit_transform(df[['Price']])
    scaler_minmax = MinMaxScaler()
    df['Price_Normalized'] = scaler_minmax.fit_transform(df[['Price']])
    
    # 特征工程（用于 ML 模型）
    df['Return'] = df['Price'].pct_change()
    df['MA5'] = df['Price'].rolling(5).mean()
    df['MA20'] = df['Price'].rolling(20).mean()
    df['Volatility'] = df['Return'].rolling(20).std()
    df = df.dropna().reset_index(drop=True)
    
    print(f"{company_name} 预处理完成 - 最终形状: {df.shape}")
    
    # 保存预处理数据
    processed_file_name = os.path.basename(file_path).replace('.csv', '_processed.csv')
    processed_path = os.path.join(processed_dir, processed_file_name)
    df.to_csv(processed_path, index=False)
    print(f"已保存预处理数据到: {processed_path}")
    
    return df

# ==================== 预测模型训练与评估 ====================
def train_and_evaluate_models(df, company_name):
    print(f"\n=== 训练预测模型：{company_name} ===")
    
    # 目标：下一日 Price
    df['Target'] = df['Price'].shift(-1)
    df = df.dropna().reset_index(drop=True)
    
    # 特征列
    feature_cols = ['Open', 'High', 'Low', 'Vol.', 'MA5', 'MA20', 'Volatility', 'Return']
    X = df[feature_cols]
    y = df['Target']
    
    # 训练/测试划分（80% 训练）
    split_idx = int(len(df) * 0.8)
    X_train, X_test = X[:split_idx], X[split_idx:]
    y_train, y_test = y[:split_idx], y[split_idx:]
    
    results = {}
    predictions = {}  # 用于绘图
    
    # 统一的评估函数
    def evaluate_model(y_true, y_pred, model_name):
        mse = mean_squared_error(y_true, y_pred)
        rmse = np.sqrt(mse)
        mae = mean_absolute_error(y_true, y_pred)
        r2 = r2_score(y_true, y_pred)
        results[model_name] = {
            'MSE': mse,
            'RMSE': rmse,
            'MAE': mae,
            'R2': r2
        }
        predictions[model_name] = y_pred
        print(f"  {model_name:12s} | MSE: {mse:.4f} | RMSE: {rmse:.4f} | MAE: {mae:.4f} | R²: {r2:.4f}")

    # (1) SARIMA
    print("训练 SARIMA 模型...")
    price_series_train = df['Price'][:split_idx].values
    try:
        sarima_model = SARIMAX(price_series_train, order=(5,1,0), seasonal_order=(1,1,1,12))
        sarima_fit = sarima_model.fit(disp=False)
        sarima_pred = sarima_fit.forecast(steps=len(y_test))
        evaluate_model(y_test, sarima_pred, 'SARIMA')
    except Exception as e:
        print("SARIMA 训练失败:", e)
        results['SARIMA'] = {'MSE': np.nan, 'RMSE': np.nan, 'MAE': np.nan, 'R2': np.nan}

    # (2) 机器学习模型
    print("训练 RandomForest...")
    rf = RandomForestRegressor(n_estimators=100, random_state=42)
    rf.fit(X_train, y_train)
    rf_pred = rf.predict(X_test)
    evaluate_model(y_test, rf_pred, 'RandomForest')

    print("训练 XGBoost...")
    xgb_model = xgb.XGBRegressor(n_estimators=100, random_state=42)
    xgb_model.fit(X_train, y_train)
    xgb_pred = xgb_model.predict(X_test)
    evaluate_model(y_test, xgb_pred, 'XGBoost')

    print("训练 LightGBM...")
    lgb_model = lgb.LGBMRegressor(n_estimators=100, random_state=42)
    lgb_model.fit(X_train, y_train)
    lgb_pred = lgb_model.predict(X_test)
    evaluate_model(y_test, lgb_pred, 'LightGBM')
    predictions['LightGBM'] = lgb_pred

    # (3) LSTM（保持原逻辑不变）
    print("训练 LSTM 模型...")
    seq_length = 20
    
    def create_sequences(data, seq_length):
        xs, ys = [], []
        for i in range(len(data) - seq_length):
            xs.append(data[i:i + seq_length])
            ys.append(data[i + seq_length])
        return np.array(xs), np.array(ys)
    
    full_price = df['Price'].values.reshape(-1, 1)
    scaler = MinMaxScaler()
    price_scaled = scaler.fit_transform(full_price)
    
    train_scaled = price_scaled[:split_idx]
    test_scaled = price_scaled[split_idx:]
    
    X_lstm_train, y_lstm_train = create_sequences(train_scaled, seq_length)
    
    if len(train_scaled) >= seq_length:
        test_input = np.concatenate([train_scaled[-seq_length:], test_scaled])
    else:
        test_input = test_scaled
    
    lstm_pred_aligned = None
    plot_dates_lstm = None
    
    if len(test_input) < seq_length + 1:
        print("数据不足，无法训练 LSTM")
        results['LSTM'] = {'MSE': np.nan, 'RMSE': np.nan, 'MAE': np.nan, 'R2': np.nan}
    else:
        X_lstm_test, y_lstm_test_scaled = create_sequences(test_input, seq_length)
        
        X_lstm_train_t = torch.from_numpy(X_lstm_train).float()
        y_lstm_train_t = torch.from_numpy(y_lstm_train).float()
        X_lstm_test_t = torch.from_numpy(X_lstm_test).float()
        
        model = LSTMModel(input_size=1, hidden_size=50, num_layers=1)
        criterion = nn.MSELoss()
        optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
        
        epochs = 50
        for epoch in range(epochs):
            model.train()
            optimizer.zero_grad()
            output = model(X_lstm_train_t)
            loss = criterion(output, y_lstm_train_t)
            loss.backward()
            optimizer.step()
            if (epoch + 1) % 10 == 0:
                print(f"  Epoch {epoch+1}/{epochs}, Loss: {loss.item():.6f}")
        
        model.eval()
        with torch.no_grad():
            lstm_pred_scaled = model(X_lstm_test_t).numpy()
        
        lstm_pred = scaler.inverse_transform(lstm_pred_scaled).flatten()
        
        actual_test_prices = full_price[split_idx + seq_length : split_idx + seq_length + len(lstm_pred)]
        min_len = min(len(actual_test_prices), len(lstm_pred))
        actual_test_prices = actual_test_prices[:min_len].flatten()
        lstm_pred_aligned = lstm_pred[:min_len]
        
        evaluate_model(actual_test_prices, lstm_pred_aligned, 'LSTM')
        plot_dates_lstm = df['Date'][split_idx + seq_length : split_idx + seq_length + min_len]
        predictions['LSTM'] = lstm_pred_aligned

    # 打印完整评估表格（保持原样）
    print("\n" + "="*60)
    print(f"{company_name} 模型性能对比（测试集）")
    print("="*60)
    print(f"{'模型':12s} | {'MSE':>10s} | {'RMSE':>10s} | {'MAE':>10s} | {'R²':>10s}")
    print("-"*60)
    for model_name in ['SARIMA', 'RandomForest', 'XGBoost', 'LightGBM', 'LSTM']:
        m = results.get(model_name, {'MSE': np.nan, 'RMSE': np.nan, 'MAE': np.nan, 'R2': np.nan})
        print(f"{model_name:12s} | {m['MSE']:10.4f} | {m['RMSE']:10.4f} | {m['MAE']:10.4f} | {m['R2']:10.4f}")
    print("="*60)

    # ==================== 新增：模型指标对比柱状图 ====================
    models = ['SARIMA', 'RandomForest', 'XGBoost', 'LightGBM', 'LSTM']
    mse_vals = [results.get(m, {}).get('MSE', np.nan) for m in models]
    rmse_vals = [results.get(m, {}).get('RMSE', np.nan) for m in models]
    mae_vals = [results.get(m, {}).get('MAE', np.nan) for m in models]
    r2_vals = [results.get(m, {}).get('R2', np.nan) for m in models]

    x = np.arange(len(models))
    width = 0.2

    fig, ax1 = plt.subplots(figsize=(12, 7))

    ax1.bar(x - 1.5*width, mse_vals, width, label='MSE', color='#FF9999')
    ax1.bar(x - 0.5*width, rmse_vals, width, label='RMSE', color='#66B2FF')
    ax1.bar(x + 0.5*width, mae_vals, width, label='MAE', color='#99FF99')
    ax1.set_ylabel('误差指标值', fontsize=14)
    ax1.set_xlabel('模型', fontsize=14)
    ax1.set_title(f'{company_name} 各模型性能指标对比', fontsize=16)
    ax1.set_xticks(x)
    ax1.set_xticklabels(models)
    ax1.legend(loc='upper left')
    ax1.grid(True, axis='y', alpha=0.3)

    # R² 使用右侧坐标轴
    ax2 = ax1.twinx()
    ax2.bar(x + 1.5*width, r2_vals, width, label='R²', color='#FFCC99')
    ax2.set_ylabel('R² 得分', fontsize=14)
    ax2.legend(loc='upper right')

    # 在柱子上显示数值（可选，美观）
    def autolabel(bars, is_r2=False):
        for bar in bars:
            height = bar.get_height()
            if not np.isnan(height):
                ax = ax2 if is_r2 else ax1
                ax.annotate(f'{height:.4f}',
                            xy=(bar.get_x() + bar.get_width() / 2, height),
                            xytext=(0, 3),
                            textcoords="offset points",
                            ha='center', va='bottom', fontsize=9)

    # 如果需要显示数值，可取消下面注释
    # autolabel(ax1.patches[:5], False)  # MSE
    # autolabel(ax1.patches[5:10], False) # RMSE
    # autolabel(ax1.patches[10:15], False)# MAE
    # autolabel(ax2.patches, True)       # R2

    plt.tight_layout()
    metrics_plot_path = os.path.join(processed_dir, f"{company_name}_model_metrics_comparison.png")
    plt.savefig(metrics_plot_path, dpi=200, bbox_inches='tight')
    plt.close()
    print(f"模型性能指标对比图已保存至: {metrics_plot_path}\n")
    # ============================================================

    # 原有的价格预测对比图（保持不变）
    plt.figure(figsize=(14, 7))
    plt.plot(df['Date'][split_idx:split_idx + len(y_test)], y_test.values, 
             label='实际价格', linewidth=2, color='black')
    plt.plot(df['Date'][split_idx:], predictions.get('LightGBM', []), 
             label='LightGBM 预测', alpha=0.8, linewidth=2)
    
    if lstm_pred_aligned is not None:
        plt.plot(plot_dates_lstm, lstm_pred_aligned, 
                 label='LSTM 预测', alpha=0.8, linestyle='--', linewidth=2)
    
    plt.title(f'{company_name} 股票价格预测对比（测试集）', fontsize=16)
    plt.xlabel('日期')
    plt.ylabel('价格')
    plt.legend(fontsize=12)
    plt.grid(True, alpha=0.3)
    plt.xticks(rotation=45)
    plot_path = os.path.join(processed_dir, f"{company_name}_prediction_plot.png")
    plt.tight_layout()
    plt.savefig(plot_path, dpi=200)
    plt.close()
    print(f"预测对比图已保存至: {plot_path}\n")
    
    return results

# ==================== 主程序 ====================
processed_dfs = {}
all_results = {}

for file_path in file_paths:
    if os.path.exists(file_path):
        df = preprocess_stock_data(file_path)
        processed_dfs[file_path] = df
        results = train_and_evaluate_models(df, os.path.basename(file_path).split(' Stock')[0])
        all_results[os.path.basename(file_path)] = results
    else:
        print(f"文件未找到: {file_path}")

print("所有股票处理与模型训练完成！")
print(f"原始数据位于：{original_dir}")
print(f"预处理数据与预测图位于：{processed_dir}")