~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~重要的内容~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
安装库文件语句：
pip install pandas numpy matplotlib scikit-learn xgboost lightgbm statsmodels torch torchvision torchaudio

前端运行代码：streamlit run dashboard.py
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~关于代码的说明~~~~~~~~~~~~~~~~~~~~~~~~~~~~
三大类预测模型：
传统时间序列：使用 SARIMA（比 ARIMA 更强大，支持季节性）。
机器学习：RandomForest、XGBoost、LightGBM（使用新增特征如移动均线、波动率）。
深度学习：LSTM（使用 PyTorch，支持长期依赖）。

每个股票：
训练所有模型。
输出 MAE、RMSE、R² 评估指标。
生成“实际 vs LSTM vs LightGBM”预测对比图，保存到“新数据集”文件夹。