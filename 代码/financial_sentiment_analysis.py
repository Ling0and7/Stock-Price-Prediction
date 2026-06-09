import pandas as pd
import numpy as np
import jieba
import os
import joblib
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report
from transformers import pipeline
import torch
import matplotlib.pyplot as plt

# ==================== 修复中文显示问题 ====================
# 使用系统可用字体（如 Arial Unicode MS 或 Microsoft YaHei），避免缺失字体警告
plt.rcParams['font.sans-serif'] = ['Arial Unicode MS', 'Microsoft YaHei', 'SimHei', 'DejaVu Sans']  # 多备选字体
plt.rcParams['axes.unicode_minus'] = False  # 正常显示负号

# ==================== 创建文件夹 ====================
model_dir = "情感模型"
os.makedirs(model_dir, exist_ok=True)

# ==================== 1. 从 Excel 加载姜富伟金融情绪词典 ====================
excel_file = "中文金融情感词典_姜富伟等(2021).xlsx"

print("正在加载姜富伟中文金融情感词典...")

# 读取 negative sheet
df_neg = pd.read_excel(excel_file, sheet_name="negative", header=0)
negative_words = df_neg.iloc[:, 0].dropna().astype(str).tolist()  # 第一列
negative_words = [word.strip() for word in negative_words if word.strip()]

# 读取 positive sheet
df_pos = pd.read_excel(excel_file, sheet_name="positive", header=0)
positive_words = df_pos.iloc[:, 0].dropna().astype(str).tolist()
positive_words = [word.strip() for word in positive_words if word.strip()]

print(f"成功加载负向词：{len(negative_words)} 个")
print(f"成功加载正向词：{len(positive_words)} 个")
print(f"示例正向词：{positive_words[:10]}")
print(f"示例负向词：{negative_words[:10]}")

# ==================== 情绪词典法（使用完整金融词典） ====================
def dictionary_sentiment_score(text):
    """
    使用姜富伟金融情绪词典计算文本情绪得分
    得分 = (正向词数 - 负向词数) / (正向词数 + 负向词数 + 1)  # 加1避免除零
    范围约 -1 到 +1
    """
    if not isinstance(text, str) or not text.strip():
        return 0.0
    
    words = jieba.lcut(text)
    pos_count = sum(1 for word in words if word in positive_words)
    neg_count = sum(1 for word in words if word in negative_words)
    
    total = pos_count + neg_count
    if total == 0:
        return 0.0
    score = (pos_count - neg_count) / total
    return round(score, 4)

# ==================== 2. 机器学习方法（TF-IDF + 分类器） ====================
def train_ml_sentiment_model(texts, labels, save_path=model_dir):
    """
    训练 TF-IDF + 机器学习模型，并保存到 情感模型 文件夹
    """
    print("\n正在训练机器学习情绪分类模型...")
    
    vectorizer = TfidfVectorizer(
        max_features=10000,
        ngram_range=(1, 3),
        min_df=1,
        stop_words=None
    )
    
    X = vectorizer.fit_transform(texts)
    X_train, X_test, y_train, y_test = train_test_split(X, labels, test_size=0.2, random_state=42, stratify=labels)
    
    # 训练多个模型，取表现最好的（这里默认用 RandomForest）
    models = {
        'NaiveBayes': MultinomialNB(),
        'SVM': SVC(kernel='linear', probability=True),
        'RandomForest': RandomForestClassifier(n_estimators=200, random_state=42, n_jobs=-1)
    }
    
    best_model = None
    best_acc = 0
    results = {}
    
    for name, model in models.items():
        model.fit(X_train, y_train)
        pred = model.predict(X_test)
        acc = accuracy_score(y_test, pred)
        results[name] = acc
        print(f"{name} 准确率: {acc:.4f}")
        
        if acc > best_acc:
            best_acc = acc
            best_model = model
    
    print(f"\n最佳模型：{list(models.keys())[list(results.values()).index(best_acc)]}，准确率：{best_acc:.4f}")
    
    # 保存向量化器和最佳模型
    vectorizer_path = os.path.join(save_path, "tfidf_vectorizer.pkl")
    model_path = os.path.join(save_path, "sentiment_classifier.pkl")
    
    joblib.dump(vectorizer, vectorizer_path)
    joblib.dump(best_model, model_path)
    
    print(f"已保存 TF-IDF 向量化器到：{vectorizer_path}")
    print(f"已保存情绪分类模型到：{model_path}")
    
    return best_model, vectorizer

# ==================== 3. 深度学习方法：BERT 金融情绪分析 ====================
def bert_sentiment_analysis(texts):
    """
    使用中文金融领域优化的 BERT 模型（推荐使用专业 FinBERT）
    这里使用一个支持中文的通用情感分析模型
    """
    print("\n正在使用 BERT 进行情绪分析...")
    
    # 推荐模型（支持中文情感）
    sentiment_pipeline = pipeline(
        "sentiment-analysis",
        model="uer/roberta-base-finetuned-dianping-chinese",  # 中文点评情感模型，效果较好
        # 或者使用：nlptown/bert-base-multilingual-uncased-sentiment
        device=0 if torch.cuda.is_available() else -1
    )
    
    results = []
    for i, text in enumerate(texts):
        if len(text) > 500:
            text = text[:500]
        try:
            res = sentiment_pipeline(text)[0]
            label = res['label']
            score = res['score']
            if label == 'POSITIVE' or 'positive' in label.lower():
                sentiment = '积极'
            else:
                sentiment = '消极'
            results.append({
                'text': texts[i],
                'sentiment': sentiment,
                'score': round(score, 4),
                'model': 'BERT'
            })
        except:
            results.append({'text': texts[i], 'sentiment': '中性', 'score': 0.0, 'model': 'BERT'})
    
    return pd.DataFrame(results)

# ==================== 示例运行 ====================
if __name__ == "__main__":
    # 示例金融新闻/评论文本（可替换为真实数据）
    sample_texts = [
        "公司三季度业绩大幅增长，净利润同比增长超过50%，市场前景看好。",
        "受大盘调整影响，股价连续下跌，投资者信心受挫，存在下行风险。",
        "今日A股震荡走势，成交量萎缩，资金观望情绪较重。",
        "新能源板块强势上涨，多只个股涨停，机构资金大幅流入。",
        "公司发布高管减持公告，市场反应冷淡，股价承压。",
        "宏观经济数据超预期，政策宽松信号明显，利好股市。",
        "年报业绩爆雷，净利润大幅亏损，股价跌停。"
    ]
    
    df_sample = pd.DataFrame({'text': sample_texts})
    
    # === 1. 情绪词典法 ===
    print("\n=== 1. 情绪词典法（姜富伟金融词典） ===")
    df_sample['dict_score'] = df_sample['text'].apply(dictionary_sentiment_score)
    df_sample['dict_sentiment'] = df_sample['dict_score'].apply(
        lambda x: '积极' if x > 0.1 else ('消极' if x < -0.1 else '中性')
    )
    print(df_sample[['text', 'dict_score', 'dict_sentiment']])
    
    # 可视化词典得分（已修复中文显示）
    plt.figure(figsize=(12, 6))
    bars = plt.bar(range(len(df_sample)), df_sample['dict_score'], 
                   color=np.where(df_sample['dict_score'] > 0, 'green', 
                                np.where(df_sample['dict_score'] < 0, 'red', 'gray')))
    plt.axhline(0, color='black', linewidth=0.8)
    plt.title('金融新闻情绪词典得分（姜富伟词典）', fontsize=16)
    plt.ylabel('情绪得分')
    plt.xlabel('新闻样本')
    plt.xticks([])
    for i, bar in enumerate(bars):
        score = df_sample['dict_score'].iloc[i]
        plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + (0.01 if score >= 0 else -0.02),
                 f'{score:.3f}', ha='center', va='bottom' if score >= 0 else 'top', fontsize=10)
    plt.tight_layout()
    plt.show()
    
    # === 2. 机器学习模型训练（需提供标签）===
    print("\n=== 2. 机器学习模型训练 ===")
    # 注意：真实使用时请提供标注好的标签数据
    # 这里用词典结果模拟标签（仅演示）
    fake_labels = df_sample['dict_sentiment'].tolist()
    
    if len(set(fake_labels)) >= 2:  # 至少有两个类别才能训练
        model, vectorizer = train_ml_sentiment_model(df_sample['text'].tolist(), fake_labels)
    else:
        print("标签类别不足，无法训练模型（演示数据太少）")
    
    # === 3. BERT 深度学习情绪分析 ===
    bert_results = bert_sentiment_analysis(df_sample['text'].tolist())
    print("\n=== 3. BERT 深度学习情绪分析结果 ===")
    print(bert_results[['text', 'sentiment', 'score']])
    
    print(f"\n所有模型训练完成！")
    print(f"机器学习模型已保存至文件夹：{model_dir}")