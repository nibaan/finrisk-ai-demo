# FinRisk AI

信用卡次月违约分析看板。三个页面覆盖数据质量、风险分群和模型评估。右下角助手提供五个基于随包汇总的本地示例。

应用入口为 `app.py`。Streamlit Community Cloud 使用仓库根目录的 `requirements.txt`，部署时在 Advanced settings 中选择 Python 3.14。

上线后的演示可从「风险分析」查看明确延迟月份分群，再点右下角助手查询同一个分组。2–6 个月组为 1,826／3,441，53.07%，口径为冻结训练集。

数据来源为 2005 年台湾信用卡历史样本。Yeh, I. (2009). Default of Credit Card Clients. 本应用展示清洗、分组和建模后生成的汇总结果。数据许可为 CC BY 4.0。[UCI 数据及许可](https://archive.ics.uci.edu/dataset/350/default+of+credit+card+clients)
