# G2 数据字典

来源：本地只读 `data/raw/default of credit card clients.xls`，UCI [Default of Credit Card Clients](https://archive.ics.uci.edu/dataset/350/default+of+credit+card+clients)，Yeh, I. (2009), DOI: 10.24432/C55S3H，CC BY 4.0。原件 SHA-256：`30c6be3abd8dcfd3e6096c828bad8c2f011238620f5369220bd60cfc82700933`。两张业务表均由同一 XLS 宽表转换，非原生多源。原始 25 列缺失数为 0；未解释编码仍是原值，不是缺失。

| 表.字段 | 来源 | 含义、类型与单位 | 缺失／处理／入模 |
| --- | --- | --- | --- |
| customer_master.customer_id | ID | 客户唯一标识，INTEGER | 非空主键；不入模 |
| customer_master.limit_bal | X1 / LIMIT_BAL | 信用额度，INTEGER，新台币 | 非空原值；后续是否入模待 G3 训练集决定 |
| customer_master.sex | X2 / SEX | 官方性别编码，INTEGER | 非空原值；待 G3 决定 |
| customer_master.education | X3 / EDUCATION | 教育编码，INTEGER | 非空原值；0/5/6 官方未解释，单列保留；待 G3 决定 |
| customer_master.marriage | X4 / MARRIAGE | 婚姻编码，INTEGER | 非空原值；0 官方未解释，单列保留；待 G3 决定 |
| customer_master.age | X5 / AGE | 年龄，INTEGER，岁 | 非空原值；待 G3 决定 |
| customer_master.target_default_payment_next_month | Y / default payment next month | 次月违约还款事件，INTEGER，1 是、0 否 | 非空二值，仅主表；仅作标签，不入模 |
| monthly_credit_history.customer_id | ID | 客户标识，INTEGER | 非空外键；不入模 |
| monthly_credit_history.history_month | 下表映射 | 历史月份，TEXT，YYYY-MM | 非空，与客户 ID 构成主键；不直接入模 |
| monthly_credit_history.history_month_index | 下表映射 | 2005-04 至 09 升序 1—6，INTEGER | 非空；不直接入模 |
| monthly_credit_history.repayment_status | 下表映射 | 官方还款状态原编码，INTEGER | 非空；仅 1—9 计为官方解释的延迟；-2/0 未解释，-1 为官方解释的其他状态；原值不求均值，后续入模待定 |
| monthly_credit_history.bill_statement_amt | 下表映射 | 账单金额，INTEGER，新台币 | 非空；负数保留，客户级均值仅供 G2 查询，入模待定 |
| monthly_credit_history.previous_payment_amt | 下表映射 | 前期付款金额，INTEGER，新台币 | 非空；客户级均值仅供 G2 查询，入模待定；不构造同月还款率 |
| customer_split.customer_id / split | 按唯一客户及 Y 分层 | 冻结的实际 ID 集合；train/validation/test | 种子 42，60/20/20；不是特征 |

| 月份 / index | repayment_status | bill_statement_amt | previous_payment_amt |
| --- | --- | --- | --- |
| 2005-04 / 1 | X11 / PAY_6 | X17 / BILL_AMT6 | X23 / PAY_AMT6 |
| 2005-05 / 2 | X10 / PAY_5 | X16 / BILL_AMT5 | X22 / PAY_AMT5 |
| 2005-06 / 3 | X9 / PAY_4 | X15 / BILL_AMT4 | X21 / PAY_AMT4 |
| 2005-07 / 4 | X8 / PAY_3 | X14 / BILL_AMT3 | X20 / PAY_AMT3 |
| 2005-08 / 5 | X7 / PAY_2 | X13 / BILL_AMT2 | X19 / PAY_AMT2 |
| 2005-09 / 6 | X6 / PAY_0 | X12 / BILL_AMT1 | X18 / PAY_AMT1 |

`history_month` 是字段映射，不证明付款偿还同月账单；具体目标日历月若称 2005-10，仅是根据末次历史月及“次月”作出的推断。每客户六个月是完整性检查，不默认作有区分度的特征。质量详情见 `data/processed/g2_results/quality.json`。
