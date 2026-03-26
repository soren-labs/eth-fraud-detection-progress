# SHAP 结果汇总

## 1. 说明

本次 SHAP 分析基于：

- 优化后的 XGBoost 模型
- 推荐测试阈值：`0.50`
- 特征方案：`numeric_textfreq`

SHAP 分析分为两部分：

1. **全局解释**：看模型整体最依赖哪些字段
2. **单地址案例解释**：看某个具体地址为什么被判为高风险，以及它更接近哪类欺诈倾向

## 2. 全局解释结论

从全局 SHAP 结果来看，当前模型最重视的字段主要包括：

1. `Unique Received From Addresses`
2. `Time Diff between first and last (Mins)`
3. `avg val received`
4. `total ether received`
5. `Avg min between received tnx`
6. `ERC20_most_rec_token_type_freq`
7. `Sent tnx`
8. `total transactions (including tnx to create contract)`
9. `ERC20 min val rec`
10. `total Ether sent`

### 这些结果说明什么

- 模型非常重视**交互地址数量**，说明“从多少不同地址收款”是重要风险信号。
- 模型非常重视**生命周期长短**，说明“短平快”地址确实更可疑。
- 模型非常重视**收款规模与收款模式**，说明资金归集行为很关键。
- 模型也重视 **ERC20 相关特征**，说明发币骗局 / 代币异常交互对风险识别有明显帮助。

### 与三类欺诈的对应关系

- **钓鱼诈骗与资金清洗**：`Time Diff between first and last (Mins)`、`Avg min between sent tnx)`、`total ether balance`
- **庞氏骗局**：`Unique Received From Addresses`、`avg val received`、`Received Tnx`
- **发币骗局与跑路盘**：`ERC20_most_rec_token_type_freq`、`ERC20 min val rec`、`Total ERC20 tnxs`

## 3. 单地址案例解释

本次从测试集中选取了 3 个被模型正确识别为欺诈地址的案例，并根据业务规则分别对应三类欺诈倾向。

### 3.1 钓鱼诈骗与资金清洗倾向案例

- 地址：`0xb615a63688104f93e59c44c2e8f2ed9ac0401ee2`
- 真实标签：`FLAG = 1`
- 模型预测：`1`
- 欺诈概率：`0.9999`

主要正向推动特征包括：

- `ERC20 min val rec`
- `ERC20_most_rec_token_type_freq`
- `total transactions (including tnx to create contract)`
- `total ether balance`
- `total ether received`
- `min value received`

可解释结论：

> 该地址表现出非常高的欺诈风险。模型认为它在交易活跃度、资金收取规模、净流量状态以及代币相关行为上都明显偏离正常地址，因此更接近“钓鱼诈骗与资金快速转移”的行为模式。

对应图表：

- `case_103_phishing_money_laundering.png`

### 3.2 庞氏骗局倾向案例

- 地址：`0x536a6ba0d913d5d6a4ce2c6eb7ed0de3c0f0b89e`
- 真实标签：`FLAG = 1`
- 模型预测：`1`
- 欺诈概率：`0.7618`

主要正向推动特征包括：

- `Unique Received From Addresses`
- `avg val received`
- `Sent tnx`
- `min value received`
- `ERC20 min val sent`
- `Avg min between received tnx`

可解释结论：

> 该地址的风险主要来自多源收款特征和收款金额模式。模型尤其重视其“来自较多不同地址的收款行为”，这与庞氏骗局中的持续吸金逻辑较为一致，因此该地址更接近“庞氏骗局倾向”。

对应图表：

- `case_788_ponzi_scheme.png`

### 3.3 发币骗局与跑路盘倾向案例

- 地址：`0xd808259ca07fdf4d8fa825c4704f624352e2dc14`
- 真实标签：`FLAG = 1`
- 模型预测：`1`
- 欺诈概率：`0.8230`

主要正向推动特征包括：

- `Unique Received From Addresses`
- `min value received`
- `ERC20 most sent token type_freq`
- `ERC20 max val sent`
- `ERC20_most_rec_token_type_freq`
- `ERC20 max val rec`

可解释结论：

> 该地址的异常主要集中在 ERC20 相关特征和异常收款模式上。模型捕捉到了它在代币交互上的显著偏离，因此更接近“发币骗局与跑路盘倾向”。

对应图表：

- `case_482_ico_rugpull.png`

## 4. 论文中可以如何表述

### 4.1 全局解释表述

可以写成：

> 全局 SHAP 结果表明，模型主要依赖交互地址数量、生命周期、平均收款金额、累计收款规模以及 ERC20 相关特征进行判别，说明链上资金归集行为、短周期活跃模式与代币交互异常是欺诈识别的重要依据。

### 4.2 案例分析表述

可以写成：

> 单地址案例解释进一步表明，模型不仅能够识别高风险地址，还能通过关键特征贡献揭示其更接近的欺诈业务模式。例如，生命周期短且资金转移迅速的地址更接近钓鱼与资金清洗场景，而具有多源收款特征的地址则更接近庞氏骗局场景。

### 4.3 方法边界说明

一定要补一句：

> 需要说明的是，本文中的“欺诈类型倾向”分析是基于模型解释结果与业务规则的综合判断，而非基于真实三分类标签的监督学习结论。

## 5. 对应文件

- 全局 SHAP 条形图：`global_shap_bar_top20.png`
- 全局 SHAP 总结图：`global_shap_summary_top15.png`
- 全局特征表：`global_shap_importance.csv`
- 案例说明：`case_explanations.csv`
- 单地址案例图：
  - `case_103_phishing_money_laundering.png`
  - `case_788_ponzi_scheme.png`
  - `case_482_ico_rugpull.png`

## 6. 结论

当前 SHAP 结果已经足够支撑论文中的“可解释性分析”部分，并且和你前面定义的三类欺诈业务逻辑基本一致。下一步如果继续推进，最合适的是：

1. 从这 3 个案例里选 2 到 3 个写进论文正文
2. 在论文中加入全局 SHAP 图和单地址案例图
3. 基于这些结果继续做三类欺诈类型的文字分析和答辩讲解准备
