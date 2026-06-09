import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from sklearn.ensemble import RandomForestRegressor
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
import xgboost as xgb
import lightgbm as lgb
import torch
import torch.nn as nn
import streamlit as st
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import os
import joblib
import jieba
from transformers import pipeline


# ==================== 配置 ====================
st.set_page_config(page_title="AutoStock Forecaster + 金融情感分析", layout="wide")
st.title("🚗 AutoStock Forecaster & 金融文本情感分析仪表盘")

# 页面选择
page = st.sidebar.radio("选择功能模块", ["股票价格预测仪表盘", "金融文本情感分析"])

processed_dir = "新数据集"
model_dir = "情感模型"
os.makedirs(model_dir, exist_ok=True)

# ==================== LSTM 模型定义 ====================
class LSTMModel(nn.Module):
    def __init__(self, input_size=1, hidden_size=50, num_layers=1, output_size=1):
        super(LSTMModel, self).__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size, output_size)
    
    def forward(self, x):
        out, _ = self.lstm(x)
        out = self.fc(out[:, -1, :])
        return out

# ==================== 侧边栏选择股票 ====================
if page == "股票价格预测仪表盘":
    companies = ["Tesla", "Ford Motor", "Toyota Motor", "BYD A", "Mercedes Benz Group"]
    company = st.sidebar.selectbox("请选择股票公司", companies, key="stock_select")

    # 查找对应的预处理文件
    processed_files = [f for f in os.listdir(processed_dir) if f.endswith("_processed.csv")]
    company_file = None
    for f in processed_files:
        if company.replace(" ", "") in f.replace(" ", ""):  # 模糊匹配
            company_file = f
            break

    if company_file is None:
        st.error(f"未找到 {company} 的预处理数据！请先运行 AutoStock Forecaster.py 生成数据。")
        st.stop()

    df_path = os.path.join(processed_dir, company_file)
    df = pd.read_csv(df_path)
    df['Date'] = pd.to_datetime(df['Date'])
    df = df.sort_values('Date').reset_index(drop=True)

    # ==================== 模型训练与预测（复用原逻辑） ====================
    split_ratio = 0.8
    split_idx = int(len(df) * split_ratio)

    feature_cols = ['Open', 'High', 'Low', 'Vol.', 'MA5', 'MA20', 'Volatility', 'Return']

    # 树模型训练（用于特征重要性与预测）
    X = df[feature_cols].iloc[:-1]  # 去掉最后一行（Target 为 NaN）
    y = df['Price'].shift(-1).dropna()  # 下一日价格作为目标

    rf = RandomForestRegressor(n_estimators=100, random_state=42)
    xgb_model = xgb.XGBRegressor(n_estimators=100, random_state=42)
    lgb_model = lgb.LGBMRegressor(n_estimators=100, random_state=42, verbosity=-1)

    rf.fit(X, y)
    xgb_model.fit(X, y)
    lgb_model.fit(X, y)

    # 测试集预测（对齐日期）
    test_X = X.iloc[split_idx-1:]  # 从训练集最后一天开始预测
    pred_rf = rf.predict(test_X)
    pred_xgb = xgb_model.predict(test_X)
    pred_lgb = lgb_model.predict(test_X)

    test_dates = df['Date'].iloc[split_idx:]

    # ==================== LSTM 训练与预测 ====================
    seq_length = 20
    scaler = MinMaxScaler()
    price_scaled = scaler.fit_transform(df['Price'].values.reshape(-1, 1))

    def create_sequences(data, seq_length):
        xs, ys = [], []
        for i in range(len(data) - seq_length):
            xs.append(data[i:i + seq_length])
            ys.append(data[i + seq_length])
        return np.array(xs), np.array(ys)

    if len(price_scaled) > seq_length:
        X_seq, y_seq = create_sequences(price_scaled, seq_length)
    
        train_size = split_idx - seq_length
        X_train_seq = X_seq[:train_size]
        y_train_seq = y_seq[:train_size]
    
        X_train_t = torch.from_numpy(X_train_seq).float()
        y_train_t = torch.from_numpy(y_train_seq).float()
    
        lstm_model = LSTMModel()
        criterion = nn.MSELoss()
        optimizer = torch.optim.Adam(lstm_model.parameters(), lr=0.001)
    
        # 快速训练（Dashboard 不需要极致精度）
        lstm_model.train()
        for epoch in range(30):
            optimizer.zero_grad()
            output = lstm_model(X_train_t)
            loss = criterion(output, y_train_t)
            loss.backward()
            optimizer.step()
    
        # 测试集序列
        test_seq_input = price_scaled[split_idx - seq_length:]
        test_sequences, _ = create_sequences(test_seq_input.reshape(-1, 1), seq_length)
        test_seq_t = torch.from_numpy(test_sequences).float()
    
        lstm_model.eval()
        with torch.no_grad():
            pred_lstm_scaled = lstm_model(test_seq_t).numpy()
        pred_lstm = scaler.inverse_transform(pred_lstm_scaled).flatten()
        lstm_dates = df['Date'].iloc[split_idx + seq_length - len(pred_lstm): split_idx + seq_length]
    else:
        pred_lstm = []
        lstm_dates = []

# ==================== 主界面布局 ====================
    col1, col2 = st.columns([3, 1])

    with col1:
        st.subheader(f"{company} 股票价格走势与多模型预测（测试集）")
    
        fig_price = make_subplots()
        # 实际价格
        fig_price.add_trace(go.Scatter(x=df['Date'][split_idx:], y=df['Price'][split_idx:],
                                   name="实际价格", line=dict(width=3, color='black')))
        # 树模型预测
        fig_price.add_trace(go.Scatter(x=test_dates, y=pred_lgb, name="LightGBM 预测", line=dict(width=2)))
        fig_price.add_trace(go.Scatter(x=test_dates, y=pred_rf, name="RandomForest 预测", line=dict(width=2)))
        fig_price.add_trace(go.Scatter(x=test_dates, y=pred_xgb, name="XGBoost 预测", line=dict(width=2)))
        # LSTM 预测
        if len(pred_lstm) > 0:
            fig_price.add_trace(go.Scatter(x=lstm_dates, y=pred_lstm, name="LSTM 预测",
                                       line=dict(dash='dash', width=2)))
    
        fig_price.update_layout(height=600, xaxis_title="日期", yaxis_title="价格",
                            legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1))
        st.plotly_chart(fig_price, use_container_width=True)

    with col2:
        st.subheader("技术指标与成交量")
        fig_ind = make_subplots(rows=2, cols=1, shared_xaxes=True,
                            vertical_spacing=0.05,
                            subplot_titles=("价格与移动均线", "成交量"),
                            row_heights=[0.7, 0.3])
    
        fig_ind.add_trace(go.Scatter(x=df['Date'], y=df['Price'], name="收盘价", line=dict(color='black')), row=1, col=1)
        fig_ind.add_trace(go.Scatter(x=df['Date'], y=df['MA5'], name="MA5", line=dict(color='orange')), row=1, col=1)
        fig_ind.add_trace(go.Scatter(x=df['Date'], y=df['MA20'], name="MA20", line=dict(color='blue')), row=1, col=1)
    
        vol_col = 'Vol.' if 'Vol.' in df.columns else None
        if vol_col and df[vol_col].notna().any():
            fig_ind.add_trace(go.Bar(x=df['Date'], y=df[vol_col], name="成交量", marker_color='lightgray'), row=2, col=1)
    
        fig_ind.update_layout(height=600, showlegend=True)
        st.plotly_chart(fig_ind, use_container_width=True)

    # ==================== 特征重要性对比 ====================
    st.subheader("树模型特征重要性对比（基于 Gain/Importance）")

    feat_df = pd.DataFrame({
        'Feature': feature_cols,
        'RandomForest': rf.feature_importances_,
        'XGBoost': xgb_model.feature_importances_,
        'LightGBM': lgb_model.feature_importances_
    }).melt(id_vars='Feature', var_name='Model', value_name='Importance')

    fig_feat = go.Figure()
    colors = {'RandomForest': '#FF9999', 'XGBoost': '#66B2FF', 'LightGBM': '#99FF99'}

    for model in ['RandomForest', 'XGBoost', 'LightGBM']:
        df_m = feat_df[feat_df['Model'] == model].sort_values('Importance', ascending=True)
        fig_feat.add_trace(go.Bar(
            y=df_m['Feature'],
            x=df_m['Importance'],
            name=model,
            orientation='h',
            marker_color=colors[model]
        ))

    fig_feat.update_layout(
        height=500,
        xaxis_title="特征重要性得分",
        yaxis_title="特征",
        barmode='group',
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
    )
    st.plotly_chart(fig_feat, use_container_width=True)

    # ==================== LSTM 隐藏状态热力图（已修复） ====================
    st.subheader("LSTM 模型内部解释：隐藏状态激活热力图（最近时间步）")

    lookback = min(50, len(price_scaled))
    recent_scaled = price_scaled[-lookback:]

    class LSTMWithHidden(nn.Module):
        def __init__(self):
            super().__init__()
            self.lstm = nn.LSTM(1, 50, batch_first=True)
            self.fc = nn.Linear(50, 1)
        def forward(self, x):
            out, (h_n, c_n) = self.lstm(x)
            pred = self.fc(out[:, -1, :])
            return pred, h_n[-1]  # 返回最后一层最后一个时间步的隐藏状态 (batch, hidden)

    # 复制权重
    lstm_hidden = LSTMWithHidden()
    lstm_hidden.lstm.load_state_dict(lstm_model.lstm.state_dict())
    lstm_hidden.fc.load_state_dict(lstm_model.fc.state_dict())

    # 输入：(batch=1, seq=lookback, feature=1)
    input_seq = torch.from_numpy(recent_scaled).float().unsqueeze(0)  # 正确3维

    with torch.no_grad():
        _, final_hidden = lstm_hidden(input_seq)  # (1, 50)

    # 可视化最后一个时间步的50个神经元激活（更简洁直观）
    neuron_activation = final_hidden.squeeze(0).numpy()

    fig_neuron = go.Figure(go.Bar(
        x=[f"Neuron {i+1}" for i in range(50)],
        y=neuron_activation,
        marker_color='purple'
    ))
    fig_neuron.update_layout(
        height=400,
        title="LSTM 最后一层隐藏状态（最近一天的50个神经元激活值）",
        xaxis_title="隐藏神经元",
        yaxis_title="激活强度"
    )
    st.plotly_chart(fig_neuron, use_container_width=True)

    st.caption("""
    **说明**：  
    - 树模型特征重要性基于内置 importance（Gain）。  
    - LSTM 部分展示最近一天的隐藏状态激活值（强度越高表示该神经元对当前预测贡献越大）。  
    - 请确保已运行 `AutoStock Forecaster.py` 生成预处理数据。
    """)

    st.success("Dashboard 加载完成！享受交互式分析吧 🚀")

# ==================== 金融文本情感分析页面 ====================
else:
    st.header("🧠 金融文本情感分析系统")
    st.markdown("""
    支持三种主流方法同时分析金融新闻/研报/评论的情感倾向：
    - **情绪词典法**：基于姜富伟等（2021）中文金融情感词典
    - **机器学习**：TF-IDF + 分类器（自动训练并保存）
    - **深度学习**：中文 BERT 情感分析模型
    """)

    # ==================== 加载姜富伟金融词典 ====================
    @st.cache_data
    def load_financial_dictionary():
        excel_file = "中文金融情感词典_姜富伟等(2021).xlsx"
        if not os.path.exists(excel_file):
            st.error(f"未找到词典文件：{excel_file}，请放在脚本同目录下！")
            st.stop()

        df_neg = pd.read_excel(excel_file, sheet_name="negative", header=0)
        negative_words = df_neg.iloc[:, 0].dropna().astype(str).str.strip().tolist()

        df_pos = pd.read_excel(excel_file, sheet_name="positive", header=0)
        positive_words = df_pos.iloc[:, 0].dropna().astype(str).str.strip().tolist()

        return set(positive_words), set(negative_words)

    positive_words, negative_words = load_financial_dictionary()

    def dictionary_sentiment_score(text):
        if not isinstance(text, str) or not text.strip():
            return 0.0
        words = jieba.lcut(text)
        pos_count = sum(1 for w in words if w in positive_words)
        neg_count = sum(1 for w in words if w in negative_words)
        total = pos_count + neg_count
        if total == 0:
            return 0.0
        return round((pos_count - neg_count) / total, 4)

    # ==================== 机器学习模型训练与加载 ====================
    @st.cache_resource
    def load_or_train_ml_model():
        vectorizer_path = os.path.join(model_dir, "tfidf_vectorizer.pkl")
        model_path = os.path.join(model_dir, "sentiment_classifier.pkl")

        if os.path.exists(vectorizer_path) and os.path.exists(model_path):
            vectorizer = joblib.load(vectorizer_path)
            model = joblib.load(model_path)
            st.success("已加载预训练的机器学习情感模型")
            return model, vectorizer
        else:
            st.warning("未找到预训练模型，正在使用示例数据训练...")
            # 示例训练数据（可替换为真实标注数据）
            sample_texts = [
                "公司业绩大幅增长，净利润超预期", "股价暴涨，市场情绪高涨",
                "新能源板块强势，资金大幅流入", "政策利好，长期看好",
                "业绩爆雷，净利润大幅亏损", "股价连续下跌，投资者恐慌",
                "高管减持，市场信心受挫", "大盘调整，存在下行风险"
            ]
            labels = ['积极', '积极', '积极', '积极', '消极', '消极', '消极', '消极']

            vectorizer = TfidfVectorizer(max_features=5000, ngram_range=(1, 3))
            X = vectorizer.fit_transform(sample_texts)
            X_train, X_test, y_train, y_test = train_test_split(X, labels, test_size=0.3, random_state=42)

            model = RandomForestClassifier(n_estimators=200, random_state=42, n_jobs=-1)
            model.fit(X_train, y_train)
            pred = model.predict(X_test)
            st.info(f"训练完成，测试准确率：{accuracy_score(y_test, pred):.3f}")

            joblib.dump(vectorizer, vectorizer_path)
            joblib.dump(model, model_path)
            st.success("模型训练完成并已保存！")
            return model, vectorizer

    ml_model, tfidf_vectorizer = load_or_train_ml_model()

    def ml_sentiment_predict(texts):
        X = tfidf_vectorizer.transform(texts)
        preds = ml_model.predict(X)
        probs = ml_model.predict_proba(X)
        scores = [max(prob) for prob in probs]
        return preds.tolist(), scores

    # ==================== BERT 模型加载 ====================
    @st.cache_resource
    def load_bert_pipeline():
        with st.spinner("正在加载中文 BERT 情感分析模型（首次较慢）..."):
            pipe = pipeline(
                "sentiment-analysis",
                model="uer/roberta-base-finetuned-dianping-chinese",
                device=0 if torch.cuda.is_available() else -1
            )
        return pipe

    bert_pipeline = load_bert_pipeline()

    def bert_sentiment_predict(texts):
        results = []
        for text in texts:
            text = text[:500]  # BERT 输入限制
            try:
                res = bert_pipeline(text)[0]
                label = res['label']
                score = res['score']
                sentiment = '积极' if 'POS' in label.upper() else '消极'
            except:
                sentiment, score = '中性', 0.0
            results.append((sentiment, round(score, 4)))
        return results

    # ==================== 用户输入 ====================
    st.subheader("输入金融文本进行情感分析")
    input_mode = st.radio("输入方式", ["单条文本", "多条文本（每行一条）"])

    if input_mode == "单条文本":
        user_text = st.text_area("请输入金融新闻/评论/研报文本：", height=150,
                                 value="公司三季度业绩大幅增长，净利润同比增长超过50%，市场前景看好。")
        texts = [user_text.strip()] if user_text.strip() else []
    else:
        user_texts = st.text_area("请输入多条文本（每行一条）：", height=300,
                                  value="公司业绩超预期\n股价承压下跌\n新能源板块强势")
        texts = [t.strip() for t in user_texts.split('\n') if t.strip()]

    if st.button("开始情感分析") and texts:
        with st.spinner("正在分析，请稍等..."):
            # 1. 词典法
            dict_scores = [dictionary_sentiment_score(t) for t in texts]
            dict_sentiments = ['积极' if s > 0.1 else ('消极' if s < -0.1 else '中性') for s in dict_scores]

            # 2. 机器学习
            ml_labels, ml_scores = ml_sentiment_predict(texts)

            # 3. BERT
            bert_results = bert_sentiment_predict(texts)
            bert_labels = [r[0] for r in bert_results]
            bert_scores = [r[1] for r in bert_results]

            # 结果表格
            results_df = pd.DataFrame({
                '文本': texts,
                '词典法得分': dict_scores,
                '词典法情感': dict_sentiments,
                '机器学习预测': ml_labels,
                '机器学习置信度': [round(s, 4) for s in ml_scores],
                'BERT 情感': bert_labels,
                'BERT 置信度': bert_scores
            })

            st.success("分析完成！")
            st.dataframe(results_df, use_container_width=True)

            # ==================== 可视化对比 ====================
            st.subheader("三种方法情感得分对比（归一化后）")
            # 将三种方法得分映射到 [-1, 1] 范围便于对比
            norm_dict = np.array(dict_scores)
            norm_ml = np.array([s if l == '积极' else -s for s, l in zip(ml_scores, ml_labels)])
            norm_bert = np.array([s if l == '积极' else -s for s, l in zip(bert_scores, bert_labels)])

            fig = go.Figure()
            fig.add_trace(go.Bar(name='词典法', x=list(range(len(texts))), y=norm_dict, marker_color='gray'))
            fig.add_trace(go.Bar(name='机器学习', x=list(range(len(texts))), y=norm_ml, marker_color='orange'))
            fig.add_trace(go.Bar(name='BERT', x=list(range(len(texts))), y=norm_bert, marker_color='purple'))

            fig.update_layout(
                barmode='group',
                height=500,
                xaxis_title="文本序号",
                yaxis_title="情感强度（正=积极，负=消极）",
                title="三种情感分析方法对比"
            )
            st.plotly_chart(fig, use_container_width=True)

    st.caption("""
    **说明**：
    - 词典法：基于姜富伟（2021）金融情感词典，解释性最强。
    - 机器学习：首次运行自动训练并保存模型，后续秒载。
    - BERT：使用中文点评领域微调模型，泛化能力强。
    """)

st.sidebar.markdown("---")
st.sidebar.info("确保以下文件在同目录：\n- `中文金融情感词典_姜富伟等(2021).xlsx`\n- 已运行股票预测脚本生成数据")