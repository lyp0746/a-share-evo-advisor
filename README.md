# A股“进化式”投研助手（进化优化版，单文件 GUI）使用说明

本文档基于提供的代码文件 a_share_evo_advisor.py，整理为一份完整、细致的使用与实现指南，适合作为 README/用户手册。内容覆盖安装与运行、界面功能、数据与模型、回测与绩效评估、数据库结构、扩展开发建议与常见问题排查。

- 项目定位：面向 A 股的“进化式”投研工具，数据源为 baostock，支持因子构建、Ridge 回归与遗传算法进化选因子、市场扫描、回测、建议生成和历史绩效评估。
- 运行形态：单文件 GUI 应用（tkinter），SQLite 持久化，matplotlib 集成绘图。
- 使用门槛低：不要求预置股票池，可从指数/文件导入，离线也可用（需本地已缓存数据）。

---

## 1. 功能一览

- 数据与持久化
  - baostock 接入，支持“增量拉取”K 线（避免全量重复下载）
  - SQLite 数据库保存价格、名称、建议、权重、用户操作等
  - 代码归一化和名称缓存
- 指标与因子
  - 技术指标：SMA/EMA/MACD/RSI/ATR/BOLL/KDJ/CCI/WR/OBV/MFI/CMF/BBP/Range/Gap/动量/波动等
  - 特征裁剪与标准化；目标收益剪裁
- 模型训练与进化
  - Ridge 回归（时间序列 CV 自动选 λ；可选优化目标 IC 或 MSE）
  - 遗传算法选择“特征子集+λ”，以验证集 Rank IC 为目标最大化
- 建议与扩展建议
  - 支持“无模型时”的基线打分策略
  - 扩展建议（仓位/止损/止盈/建议数量）
- 市场扫描与筛选
  - HS300/ZZ500/SZ50/ALL/CUSTOM
  - 增强质量分 qscore；高级筛选：趋势、接近 55 日高、波动 Z 阈值、RSI 区间、流动性阈值
  - ALL 扫描批处理限速与“跳过增量更新”
- 回测
  - TopN 选股、等权持有 H 日、单边费率、输出 CAGR/回撤/Sharpe/胜率，绘制净值
- 绩效评估与可视化
  - 历史建议筛选（日期/类型/H）、分位统计、收益分布直方图
- 用户操作记录
  - BUY/SELL/ADJ 记录，便于后续行为分析与“连续化进化”

---

## 2. 环境与安装

- 必需
  - Python 3.8+（建议）
  - tkinter（GUI 依赖）
  - SQLite（Python 原生 sqlite3 库即可）
- Python 包
  - pip install baostock pandas numpy matplotlib

系统环境补充：
- Windows：一般自带 Tk 支持
- macOS/Linux：若启动报 Tk/TkAgg 错，可安装 Tk 支持
  - Ubuntu/Debian: `sudo apt-get install python3-tk`
  - macOS（Homebrew）: `brew install python-tk@3.x`（按环境调整）

中文字体：
- matplotlib 使用 SimHei/WenQuanYi 等中文字体显示；若字体缺失可能导致中文乱码，可安装相应字体或改用系统已装字体。

---

## 3. 启动与首次使用

- 启动
  ```bash
  python a_share_evo_advisor.py
  ```
- 首次推荐流程
  1) “数据/股票池”页：
     - 填写“起始/结束日期”和“复权”方式
     - 通过“从指数填充（HS300）”或“导入股票列表文件”获取股票池
     - 点击“增量更新数据”
  2) “训练/进化”页：
     - 设置 H（日）、训练模式（ridge_cv/ga）、目标（IC/MSE）
     - 开始训练，自动选择 λ（或遗传进化选因子+λ），保存权重
  3) “自选建议”或“市场扫描”页：
     - 选择指数/自定义池，调整筛选与风险参数
     - 生成扩展建议并保存，或扫描全市场并导出
  4) “回测”页：
     - 设置 TopN 和持有 H 日、费率与流动性阈值，回测策略表现
  5) “绩效/历史”页：
     - 按日期/类型/H 评估历史建议表现，查看分位统计与直方图

说明：
- 未安装 baostock 或登录失败时，程序仍可离线使用“本地已存数据”（advisor.db）。
- 首次需联网更新数据，否则离线无数据无法训练/扫描。

---

## 4. GUI 页面详解

### 4.1 数据 / 股票池
- 股票代码输入：支持逗号/换行分隔，格式自动归一化：
  - 600000 → sh.600000
  - 000001 → sz.000001
  - 600000.SH / 000001.SZ 均可
- 导入股票列表文件：
  - .txt：每行一个代码
  - .csv：默认首列或列名为 code/证券代码/ts_code/wind_code
- 从指数填充：HS300
- 复权：1-后复权 / 2-前复权 / 3-不复权（默认 2）
- 增量更新：对已存在的 code 从“数据库最新日期+1”拉取至“结束日期”

### 4.2 训练 / 进化
- 参数
  - H（日）：预测视野（未来 H 日收益）
  - 训练模式：ridge_cv 或 ga
  - 目标指标：IC（横截面 Rank IC 最大化）或 MSE（均方误差最小化）
  - CV 折数：时间序列交叉验证
  - GA 参数：种群大小、代数
- 训练结果展示：timestamp、H、method、lambda、IC、ICIR、MSE、有效交易天数 n_days

建议：
- 初学者：先用 `ridge_cv + IC`，H=10~20，CV=4
- GA 模式耗时更长，适用于更大样本与更强泛化的需求

### 4.3 自选建议
- 输入自选股票，设置资金与单笔风险％（如 1%）
- “生成扩展建议并保存”：持久化到 advice 表
- 每条包含：
  - 预测分（或基线分），建议（买/卖/强烈买入/强烈卖出/观望）
  - 扩展建议：建议仓位、止损、止盈、建议数量（按 100 股一手）
  - 理由简述（常规信号如站上 SMA20、MACD 红柱、RSI 区间等）

### 4.4 市场扫描
- 指数池：HS300/ZZ500/SZ50/ALL/CUSTOM
- 高级筛选：
  - 趋势：价>SMA20>SMA60 且 MACD>0 且 SMA20 上行
  - 接近 55 日高：≥98%
  - 波动 Z 阈值：限制高波动
  - RSI 区间
  - 流动性阈值：近 20 日均额
- 无权重也可扫描：使用“基线打分”排序
- 建议支持导出 CSV

ALL 扫描性能建议：
- 勾选“扫描前跳过增量更新”
- 调整批处理 batch_size 与 sleep 间隔以降低接口压力

### 4.5 回测
- 策略：每 H 日调仓一次，按当日打分选 TopN 等权持有 H 日
- 支持无权重（“基线打分”）
- 交易费率：单边 bps（如 1=万分之一）
- 输出：
  - 期收益表（date, ret）
  - 净值曲线与指标（CAGR/MaxDD/Sharpe/胜率）
- 自动弹窗绘制净值曲线

### 4.6 绩效 / 历史
- 历史建议检索：按日期/类型（买/卖/全部）/H（日）
- 统计：
  - 分类（买/卖/强烈买入/强烈卖出/观望）绩效
  - 按 score 分位组（q 组）平均收益与胜率
  - 签名收益分布直方图
- 查看与导出历史记录

### 4.7 图表
- 绘制单股指标图：价格+SMA/BOLL、MACD、RSI

### 4.8 用户操作
- 记录 BUY/SELL/ADJ 行为（日期/价格/数量/备注）
- 支持刷新查看

---

## 5. 因子与模型原理

### 5.1 技术指标与特征列

- 趋势/均线：sma5, sma10, sma20, sma60, ema12, ema26
- MACD 系：dif, dea, macd
- 震荡类：rsi14, kdj_k, kdj_d, kdj_j, cci14, wr14
- 布林带：bb_mid, bbp
- 成交/资金：v_surge（量比）、obv_ema、mfi14、cmf20
- 波动/动量：vol20（年化近 20 日波动）、mom10、ret5、ret20、range_pct、gap_pct
- 位置特征：above_sma20、dif_pos、macd_up
- 风险：atr14
- 其它中间量（用于质量分与扩展建议）：amt20、near_high55、dd60、sma20_slope5 等

特征预处理：
- y（未来 H 日收益）裁剪到 [-25%, 25%]
- 若干特征进行 1%/99% 分位剪裁（去极值）
- 标准化：\( x' = (x - \mu) / \sigma \)，保存 \(\mu,\sigma\)

### 5.2 目标与评估

- 横截面 Rank IC（斯皮尔曼相关）
  - 对每个交易日 d，计算当日所有股票预测分与真实未来收益的秩相关
  - 汇总得到平均 IC、标准差、ICIR
- 时间序列交叉验证（ts_cv_splits）
  - 按日期顺序切分：前若干段为训练，后一段为验证，保持时间因果

IC 计算公式（简化表示）：
\[
\text{IC}_d = \rho(\text{rank}(\hat{y}_d),\, \text{rank}(y_d))
\]
\[
\text{ICIR} = \frac{\overline{\text{IC}}}{\sigma(\text{IC}) + \varepsilon}
\]

- Ridge 回归（带截距，截距不正则）：
  - \( \min\limits_{w,b} \|y - (Xw + b)\|_2^2 + \lambda \|w\|_2^2 \)

### 5.3 遗传算法（GA）选特征 + λ

- 个体表示：特征子集（布尔掩码）+ λ
- 初始随机族群（受 min_feat/max_feat 约束）
- 适应度：时间序列 CV 上的平均横截面 Rank IC
- 进化操作：
  - 选择（锦标赛或排序挑选）
  - 交叉（单点）与变异（按位翻转）
  - 弹性修正（保证特征数在 [min_feat, max_feat]）
- 输出：最佳子集与 λ；在全样本重训并评估

建议：
- 初期使用 `min_feat=6 ~ 12, max_feat≤20`，`n_pop=24, n_gen=8`，平衡速度与效果

---

## 6. 建议与“扩展建议”策略

### 6.1 基线打分（无模型时）
- 鼓励中短期动量、低波动、接近 55 日高、趋势成立
- 形式（简化）：
  \[
  s \approx 0.8 \cdot \text{mom10} - 0.15 \cdot \text{vol20} + 0.2 \cdot (\text{near\_high55}-0.9) + 0.05 \cdot \tanh(\text{macd}) + 0.1 \cdot \text{trend}
  \]

### 6.2 文本建议分级（score_to_advice）
- 阈值：pos_thr=0.02，neg_thr=-0.02
- 规则：
  - ≥1.5×pos_thr：强烈买入
  - ≥pos_thr：买入
  - ≤1.5×neg_thr：强烈卖出
  - ≤neg_thr：卖出
  - 否则：观望

### 6.3 扩展建议（位置、止损、止盈、数量）
- 止损：
  - 若 ATR14>0：`stop = min(close*0.95, close - 1.5*ATR14)`（至少 -5%）
  - 否则：`stop = close * 0.95`
- 目标：
  - `target = close + max(2.5*ATR14, (close - stop) * 1.5)`
- 仓位（单笔风险控制）：
  - 跌幅％：`drop_pct = (close - stop)/close`
  - \( \text{pos\_pct} = \min(30\%, \text{risk\_pct} / \text{drop\_pct}) \)
- 数量：按 100 股为一手向下取整
- 建议类型：若 score>0 且 mom10≥-2% → 买入，否则观望

---

## 7. 市场扫描与质量分

- 质量分 qscore：
  \[
  \text{qscore} = 0.5 \cdot z(\text{score}) + 0.25 \cdot z(\text{mom10}) - 0.25 \cdot z(\text{vol20}) + 0.12 \cdot \text{trend} + 0.08 \cdot \text{breakout55} + 0.05 \cdot \text{dd60}
  \]
  - 其中 z(·) 为当期截面标准化
  - dd60 ∈ [-1,0] 回撤项，负值相当于惩罚深回撤（直接相加）
- 排序：按 qscore、score、mom10 三重排序
- 高级筛选：可先以 vol_z 和 RSI 范围预筛后排序

---

## 8. 回测框架与指标

- 步骤
  1) 预载所有股票的指标数据
  2) 生成调仓日期序列：每 hold_days 取一个 d，确保未来 d+H 存在数据
  3) 每期 d：
     - 流动性过滤：amt20≥阈值
     - 打分：模型预测或基线打分
     - 选 TopN 等权
     - 收益：d→d+H 的收盘收益，减去进出双边费用
  4) 聚合为期收益表 pr (date, ret)，累计为净值 nav

- 指标
  - 日/期胜率：收益>0 的占比
  - CAGR（以 252 交易日年化）：
    \[
    \text{CAGR} = \text{NAV}_{\text{last}}^{\frac{252}{\text{total\_days}}} - 1
    \]
  - 最大回撤 MaxDD：
    \[
    \text{MaxDD} = \min_t \left( \frac{\text{NAV}_t}{\max_{s\le t}\text{NAV}_s} - 1 \right)
    \]
  - Sharpe（以持有期为一个“周期”近似年化）：
    \[
    \text{Sharpe} = \frac{\overline{r}}{\sigma(r) + \varepsilon} \cdot \sqrt{\frac{252}{\text{period\_days}}}
    \]

提示：
- 该回测为简化版，未包含成交/停牌/滑点等微观约束，仅用于快速迭代评估

---

## 9. 数据库结构（SQLite）

默认库：`advisor.db`

- prices（K 线）
  - 主键：(code, date)
  - 字段：open, high, low, close, volume, amount
  - 索引：date、code
- advice（建议记录）
  - id, date, code, score, advice, reasoning, horizon
- weights（最新模型权重）
  - timestamp 主键，horizon, features(json), weights(json), intercept, mu(json), sigma(json), lambda, notes
- meta
  - key 主键, value（如 latest_weights_ts）
- stock_names
  - code 主键, name
- user_ops（用户操作）
  - id, date, code, action(BUY/SELL/ADJ), price, qty, note

备份建议：
- 直接拷贝 `advisor.db` 文件即可备份/迁移

---

## 10. 实操建议与常见用法

- 股票池选择
  - 训练与回测尽量使用“稳定、流动性较好”的成分（如 HS300/ZZ500）
  - 自定义池可按行业/主题分类使用
- 训练窗口
  - 建议至少覆盖 3~5 年交易日，样本越多越稳定
- H（日）选择
  - 短中期（10~20 日）更适合技术因子；更长期需引入基本面/财务因子
- ALL 扫描优化
  - 勾选“跳过增量更新”，加快扫描，后续再批量更新
  - 适当提高“近20日均额阈值”，减少长尾小票
- 风险控制
  - risk_pct 建议 ≤ 1%，避免大波动 ATR 导致过大仓位
  - 止损与目标价仅为参考，实盘需结合个股特性与风控制度

---

## 11. 已知问题与修复建议

- 扫描结果插入 Treeview 时，values 末尾重复附加了 `advice` 和 `reasoning`（导致列数与值数不对齐风险）
  - 位置：`on_scan_market_async → self.scan_tree.insert(...)`
  - 建议修复：删除重复的两个参数，仅保留一次
  - 修复示例（仅保留一次 advice/reasoning）：
    ```python
    self.scan_tree.insert("", tk.END, values=(
        r["date"], r["code"], r.get("name",""), f"{r['close']:.2f}",
        f"{r['score']:.4f}", f"{r['qscore']:.4f}",
        f"{r['mom10']:.2%}", f"{r['vol20']:.3f}",
        f"{r['atr14']:.3f}", f"{r['amt20']:.0f}",
        f"{r.get('trend',0):.0f}", f"{r.get('breakout55',0):.0f}", f"{r.get('dd60',0):.2%}",
        f"{r['pos_pct'] * 100:.1f}%", f"{r['stop']:.2f}", f"{r['target']:.2f}", int(r["qty"]),
        r["advice"], r["reasoning"]
    ))
    ```
- 字体/后端
  - 若 matplotlib TkAgg 报错，请安装 Tk 或更换后端（需配合 tkinter GUI）

---

## 12. 扩展开发指南

- 增加/调整因子
  - 在 `compute_indicators` 中新增列，并在 `feature_columns()` 中选择是否纳入模型
  - 记得在训练数据构造与剪裁环节考虑新因子的极值处理
- 增加目标与评估指标
  - 可添加信息系数的“稳定性”约束、分行业中性化等
- 强化回测
  - 引入开盘交易、成交量/停牌过滤、分红送配处理、仓位动态调整
- 策略集成
  - 将 “质量分 + 风险控制 + 动量” 结合，形成多因子打分与分层持仓
- 更换数据源
  - 如需更换为 tushare/wind 等，替换“数据获取”和“名称缓存”层逻辑

---

## 13. 故障排查（FAQ）

- baostock 登录失败/未安装
  - 执行：`pip install baostock`，重启程序
  - 若接口报错/限频，可稍后再试；程序内部已做重试与 sleep
- 无法拉取指数成分（离线/未登录）
  - 离线模式无法从指数拉取，建议先在有网环境更新并缓存数据
- ALL 扫描很慢
  - 勾选“跳过增量更新”
  - 减少股票池或提高 amt20 门槛
  - 调整 batch_size、sleep_ms，降低接口压力
- 中文乱码
  - 安装中文字体或者修改 `plt.rcParams["font.family"]` 为系统已装字体

---

## 14. 免责声明

- 本程序仅用于教育与研究，不构成任何投资建议。
- 历史回测与模拟不代表未来表现，实盘有交易成本、流动性、滑点、停牌等诸多因素影响。
- 使用者需自行承担交易风险。

---

## 15. 快速命令备忘

- 安装依赖
  ```bash
  pip install baostock pandas numpy matplotlib
  # Linux/macOS 若缺少 Tk:
  # Ubuntu/Debian: sudo apt-get install python3-tk
  # macOS: brew 安装或使用 conda 的 tk 包
  ```
- 运行
  ```bash
  python a_share_evo_advisor.py
  ```