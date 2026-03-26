# SHAP 单地址案例解释

- 使用模型：`best_xgboost_optimized.joblib`
- 采用阈值：`0.50`

## 钓鱼诈骗与资金清洗倾向

- 地址：`0xb615a63688104f93e59c44c2e8f2ed9ac0401ee2`
- 真实标签 `FLAG`：`1`
- 模型预测标签：`1`
- 欺诈概率：`0.9999`
- 主要正向推动特征：`ERC20 min val rec=1.5509; ERC20_most_rec_token_type_freq=1.3978; total transactions (including tnx to create contract)=1.2611; total ether balance=1.1157; total ether received=0.7631; min value received=0.6170`
- 主要负向拉回特征：`Unique Received From Addresses=-0.5519; Avg min between sent tnx=-0.1745; max value received=-0.1022; Unique Sent To Addresses=-0.0662`
- 对应图表：`case_103_phishing_money_laundering.png`

## 庞氏骗局倾向

- 地址：`0x536a6ba0d913d5d6a4ce2c6eb7ed0de3c0f0b89e`
- 真实标签 `FLAG`：`1`
- 模型预测标签：`1`
- 欺诈概率：`0.7618`
- 主要正向推动特征：`Unique Received From Addresses=2.9970; avg val received=0.5758; Sent tnx=0.5099; min value received=0.3302; ERC20 min val sent=0.3169; Avg min between received tnx=0.2776`
- 主要负向拉回特征：`total transactions (including tnx to create contract)=-2.0511; Received Tnx=-0.6248; Number of Created Contracts=-0.5029; total ether received=-0.4708`
- 对应图表：`case_788_ponzi_scheme.png`

## 发币骗局与跑路盘倾向

- 地址：`0xd808259ca07fdf4d8fa825c4704f624352e2dc14`
- 真实标签 `FLAG`：`1`
- 模型预测标签：`1`
- 欺诈概率：`0.8230`
- 主要正向推动特征：`Unique Received From Addresses=3.2267; min value received=0.7205; ERC20 most sent token type_freq=0.3856; ERC20 max val sent=0.3681; ERC20_most_rec_token_type_freq=0.3480; ERC20 max val rec=0.3140`
- 主要负向拉回特征：`Avg min between received tnx=-0.8190; total ether received=-0.6902; total Ether sent=-0.4121; Sent tnx=-0.4010`
- 对应图表：`case_482_ico_rugpull.png`

