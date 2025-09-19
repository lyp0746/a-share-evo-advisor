# A股进化式投研助手（Baostock 重构升级版）说明文档

本项目提供一个基于 baostock 数据源的“进化式”量化选股与建议生成工具，集数据增量更新、特征工程、模型训练（岭回归/遗传算法）、市场扫描与优质股打分、仓位/止损/止盈建议、历史绩效评估于一体。采用 SQLite 持久化，内置 GUI（tkinter）便于交互使用。

免责声明：仅用于教育与研究，不构成任何投资建议。

---

## 功能总览

- 数据
  - baostock 增量拉取 A 股日线 K 线
  - 本地 SQLite 落库，支持增量更新、按区间读取
- 特征工程
  - SMA/EMA/MACD/RSI/ATR/BOLL/KDJ/CCI/WR/OBV/MFI/CMF/BBP/波动/动量/区间与缺口等
  - 标准化处理、极值剪裁（1%/99% 分位）
- 模型训练
  - 岭回归（含截距，b 不正则），时间序列 CV 选 λ
  - 目标指标可选 IC 或 MSE
  - 遗传算法同时搜索特征子集与 λ，适合更强表达
- 市场扫描与优选
  - 指数池：HS300/ZZ500/SZ50/ALL/CUSTOM
  - 综合质量分 qscore，支持趋势/突破/波动Z/RSI 筛选
  - 可导出 CSV
- 建议与风险控制
  - 扩展建议：仓位、止损、止盈、股数（手数）基于 ATR、动量与风险预算
  - 结果写入 advice 表，支持历史查询、分位统计、分布图
- 评估
  - 历史建议按 horizon 评估平均收益与胜率
  - 分位统计（按 score 分组）与收益直方图
- GUI
  - 多页签：数据/训练/自选建议/扫描/评估/图表/说明/日志
  - 异步执行与日志输出

---

## 快速开始

### 1) 环境与依赖

- Python 3.8+
- 依赖安装：
  ```bash
  pip install baostock pandas numpy matplotlib
  ```
- GUI 环境（tkinter）:
  - Windows/macOS 通常自带
  - Linux 需安装 Tk（例如 Ubuntu: `sudo apt-get install python3-tk`）
- 字体：程序已配置中文字体优先级（SimHei/WenQuanYi 等）

### 2) 运行

```bash
python a_share_evo_advisor.py
```

首次运行会初始化数据库 `advisor.db`，并弹出 GUI。

### 3) 基本流程

1. 在“数据/股票池”页输入股票代码或导入自定义股票池（.txt/.csv）
2. 点击“增量更新数据”
3. 在“训练/进化”页设置 `H`、训练模式与目标指标，开始训练
4. 在“自选建议”页生成扩展建议（会写入 SQLite）
5. 在“市场扫描”页选指数/筛选条件，开始扫描并导出结果
6. 在“绩效/历史”页评估过往建议

---

## 数据与持久化

- SQLite 文件：`advisor.db`
- 表结构：
  - prices
    | 列名   | 类型 | 说明 |
    |-------|------|------|
    | code  | TEXT | 证券代码（主键1）|
    | date  | TEXT | 交易日（主键2，YYYY-MM-DD）|
    | open/high/low/close | REAL | OHLC |
    | volume | REAL | 成交量 |
    | amount | REAL | 成交额 |

  - advice
    | 列名   | 类型 | 说明 |
    |-------|------|------|
    | id        | INTEGER | 自增主键 |
    | date      | TEXT    | 建议生成日期 |
    | code      | TEXT    | 股票代码 |
    | score     | REAL    | 预测得分 |
    | advice    | TEXT    | 文本化建议（含仓位/止损/止盈/数量） |
    | reasoning | TEXT    | 理由（基础+扩展） |
    | horizon   | INTEGER | 当时模型的 H |

  - weights
    | 列名   | 类型 | 说明 |
    |-------|------|------|
    | timestamp | TEXT | 唯一时间戳（主键） |
    | horizon   | INTEGER | 训练的 H |
    | features  | TEXT | JSON 序列化特征名列表 |
    | weights   | TEXT | JSON 序列化权重数组 |
    | intercept | REAL | 截距 b |
    | mu        | TEXT | JSON 均值向量 |
    | sigma     | TEXT | JSON 标准差向量 |
    | lambda    | REAL | 岭回归 λ |
    | notes     | TEXT | 训练说明/评估 JSON |

  - meta
    | 列名 | 类型 | 说明 |
    |-----|------|------|
    | key   | TEXT | 主键 |
    | value | TEXT | 值（例如 latest_weights_ts） |

- 数据更新
  - 函数：`fetch_k_data_incremental` 仅从 DB 最新日期+1 开始拉取，减少重复
  - baostock 频率限制已通过 sleep 控制（谨慎 ALL 扫描，建议跳过更新）

---

## 特征工程

- 基础与趋势：`sma5/10/20/60`、`ema12/26`、布林带（20,2）
- MACD 族：`dif`, `dea`, `macd`
- 震荡：`rsi14`, `kdj_k/d/j`, `cci14`, `wr14`
- 成交与资金：`v_surge`（量/5日量均）、`obv_ema`, `mfi14`, `cmf20`
- 波动与动量：`vol20`（年化）、`mom10`, `ret5/20/60`
- 结构/风险：`above_sma20`, `dif_pos`, `macd_up`, `atr14`
- 其它：`bbp`, `range_pct`, `gap_pct`, `amt20`, `near_high55`, `dd60`, `sma20_slope5`

特征列用于训练由 `feature_columns()` 定义。训练前进行：
- 目标收益剪裁：\( y \in [-0.25, 0.25] \)
- 关键特征 1%/99% 分位剪裁，鲁棒性更强
- 标准化：\( X'=(X-\mu)/\sigma \)

---

## 模型训练与进化

### 目标定义

- 训练样本：多股票、多交易日的面板数据
- 目标收益：未来 \( H \) 日收盘收益
  \[
  y_t = \frac{\text{close}_{t+H}}{\text{close}_t}-1
  \]
- 防未来函数：`shift(-H)` 且丢弃最后 \( H \) 条

### 岭回归（带截距）
- 模型：\( y \approx Xw + b \)（b 不正则）
- 问题：最小化 \( \|y - Xw - b\|^2 + \lambda \|w\|_2^2 \)
- 求解：正规方程增强一列常数，解奇异时退化为 `pinv`

### 时间序列 CV（ts_cv_splits）
- 按 unique 日期等分段，保持时间顺序
- 折内：前若干段为训练、后一段为验证
- 评价指标：
  - Rank IC（Spearman 横截面）：在每个验证日内按股票截面相关
    \[
    \text{IC} = \text{corr}_{\text{Spearman}}(\text{pred}, \text{real})
    \]
  - 选择策略：平均 IC 最大或 MSE 最小（内部转换为得分）

### 遗传算法（GA）
- 搜索空间：特征子集 + λ
- 个体表示：布尔掩码 + λ
- 变异/交叉：随机翻转、切点交叉；限制最小/最大特征数
- 适应度：时间序列 CV 的平均 Rank IC
- 精英保留 + 锦标赛选择

### 训练入口

- `train_and_save_weights(codes, horizon, mode="ridge_cv"|"ga", ...)`
  - `ridge_cv`：自动网格 λ（默认 `[1e-4,3e-4,1e-3,3e-3,1e-2,3e-2,1e-1]`）
  - `ga`：可配 `n_pop/n_gen/n_folds/min_feat/max_feat`
- 训练完成后自动写表 `weights`，并记录元信息到 `meta.latest_weights_ts`
- `load_latest_weights()` 读取最新一版权重与标准化参数

---

## 打分、信号与建议

### 预测得分与文本化建议

- 预测分：\( s = (x-\mu)/\sigma \cdot w + b \)
- 文本化建议阈值：
  - `pos_thr = 0.02`, `neg_thr = -0.02`
  - s ≥ 1.5×pos_thr → “强烈买入”；s ≥ pos_thr → “买入”
  - s ≤ 1.5×neg_thr → “强烈卖出”；s ≤ neg_thr → “卖出”
  - 其他 → “观望”
- 理由摘要：`reasoning_from_signals` 基于趋势、MACD、RSI、动量、放量等拼接人类可读的提示

### 市场扫描与综合质量分

- 指数池：`get_market_codes(index_flag, date)` 支持 HS300/ZZ500/SZ50/ALL
- 扫描主函数：`scan_market_and_rank(...)`
  - 更新数据（可跳过）、加载近端数据、计算特征、过滤流动性（`amt20`）
  - 计算预测分 `score`
  - 质量信号：
    - 趋势 `_trend_flag`: 价 > SMA20 > SMA60 且 MACD>0 且 SMA20 上行
    - 突破 `_breakout_flag`: 近高55 ≥ 98%
  - 综合质量分（更适合排序）：
    \[
    \text{qscore} = 0.5\,z(\text{score}) + 0.25\,z(\text{mom10}) - 0.25\,z(\text{vol20})
    + 0.12\,\text{trend} + 0.08\,\text{breakout55} + 0.05\,\text{dd60}
    \]
  - 可选筛选：trend、breakout、max vol_z、RSI 区间
  - 最终按 `qscore/score/mom10` 排序取 TopN

### 扩展建议（风控与仓位）

- 止损、目标价与仓位：
  \[
  \begin{aligned}
  \text{stop} &= \min(0.95 \cdot \text{close}, \ \text{close} - 1.5\cdot \text{ATR14}) \\
  \text{target} &= \text{close} + \max(2.5\cdot \text{ATR14}, \ (\text{close}-\text{stop})\cdot 1.5) \\
  \text{drop\_pct} &= \frac{\text{close} - \text{stop}}{\text{close}} \\
  \text{pos\_pct} &= \min(0.3, \ \frac{\text{risk\_pct}}{\text{drop\_pct}}) \\
  \end{aligned}
  \]
- 股票数量：100 股一手，按 `pos_pct * capital / close` 向下取整到手数
- 扩展建议与理由由 `gen_extended_advice` 生成

---

## GUI 使用指南

### 1. 数据 / 股票池
- 输入框支持 `sh.600000, sz.000001` 或换行分隔
- “导入股票列表文件”：
  - .txt：每行一代码（支持 600000、sh.600000、600000.SH）
  - .csv：默认首列；如存在 `code/证券代码/ts_code/wind_code` 优先
- 选项：
  - 勾选“训练/扫描使用自定义股票池”
  - 设定起止日期与 `adjustflag`（1 后复权/2 前复权/3 不复权）
  - “增量更新数据”写入 SQLite

### 2. 训练 / 进化
- 设置 `H（日）`、训练模式（`ridge_cv` 或 `ga`）、目标（IC/MSE）、CV 折数
- GA 参数：种群大小与代数
- 训练完成显示 IC/ICIR/MSE 与 n_days，并保存最新权重

### 3. 自选建议
- 输入/填充股票、设定资金与风险比例 `risk_pct`
- 点击“生成扩展建议并保存”，落表 advice
- 表格展示 date/code/score/advice/reasoning/horizon

### 4. 市场扫描
- 选择指数或 CUSTOM（使用导入的股票池）
- 可选“扫描前跳过增量更新（快）”（推荐 ALL 时勾选）
- 高级筛选：趋势/接近55日高/波动Z上限/RSI 区间
- 设置 TopN、流动性阈值（近20日均额）
- 完成后可导出 CSV

### 5. 绩效 / 历史
- 筛选开始/结束日期、类型（ALL/BUY/SELL）、H（留空用当时 H）
- “评估历史建议绩效”：按类型统计 n/avg_ret/win_rate
- “分位统计”：按 score 分 q 组统计
- “绘制收益分布图”：签名收益直方图（买→正；卖→负；观望→0）

### 6. 图表
- 若选中自选建议表中某一行就画该股，否则取输入框首个
- 绘制价格/均线/布林、MACD、KDJ、ATR、近高55与回撤等

---

## 编码规范与扩展点

- 代码组织：单文件 GUI，函数区分清晰（数据层/特征/训练/扫描/评估/GUI）
- 核心函数（便于二开）：
  - 数据层：
    - `load_price_df(code, start_date=None, end_date=None)`
    - `upsert_prices(code, df)`、`query_advice(...)`
  - 特征：
    - `compute_indicators(df)`：新增特征在此扩展
    - `feature_columns()`：训练特征列清单
  - 训练：
    - `build_training_data(codes, horizon, ...)`
    - `train_ridge_with_cv(...)`、`evolve_ga_feature_selection(...)`
    - `save_weights_record(...)`、`load_latest_weights()`
  - 扫描/建议：
    - `scan_market_and_rank(...)`
    - `gen_extended_advice(row, risk_pct, capital)`
  - 评估：
    - `evaluate_history_performance(...)`
    - `evaluate_quantile_performance(...)`
- 代码归一化：
  - `normalize_code("600000") -> "sh.600000"`
  - `parse_codes_from_text(...)` 支持逗号/空白/分号分隔

---

## 程序化调用示例（非 GUI）

```python
from a_share_evo_advisor import (
    init_db, bs_safe_login, update_data_for_codes, train_and_save_weights,
    load_latest_weights, score_latest_for_codes, scan_market_and_rank
)

# 1) 初始化与登录
init_db()
bs_safe_login()

# 2) 更新数据
codes = ["sh.600000", "sz.000001", "sz.300750"]
update_data_for_codes(codes, start_date="2015-01-01", end_date="2025-01-01", adjustflag="2", logger=print)

# 3) 训练（岭回归 + IC）
pack = train_and_save_weights(codes, horizon=10, mode="ridge_cv", target_metric="IC")

# 4) 最新权重打分
w = load_latest_weights()
recs = score_latest_for_codes(codes, w, start_date="2020-01-01")
for r in recs:
    print(r)  # (date, code, score, advice, reasoning)

# 5) 扫描指数
df = scan_market_and_rank("HS300", "2019-01-01", "2025-01-01", "2", w, min_amt20=2e8, topN=30)
print(df.head())
```

---

## 参数与默认值（要点）

- 训练
  - `lam_grid`: [1e-4, 3e-4, 1e-3, 3e-3, 1e-2, 3e-2, 1e-1]
  - CV 折数：默认 4
  - 目标：IC（Rank Spearman，按日横截面）
- 建议阈值
  - `pos_thr = 0.02`，`neg_thr = -0.02`
  - 强烈档 = 1.5 × 基础阈值
- 扫描与筛选
  - `min_amt20 = 2e8`
  - `topN = 30`
  - RSI 筛选默认关闭（范围 20~85 可选）
  - 波动 Z 阈值默认启用，`max_vol_z = 2.5`
- 风险参数
  - `risk_pct = 1%`（单笔风险预算）
  - `pos_pct ≤ 30%`（最大仓位上限）

---

## 评估说明

- 历史建议评估基于建议当日后的 \( H \) 日收盘价变化：
  - 买/强买 → 直接收益
  - 卖/强卖 → 收益取相反数
  - 观望 → 0
- 统计维度：平均收益、胜率（>0 的比例）
- 分位统计：按 score 分 q 组（`pd.qcut`），可观察分层效果

注意：这不是完整的回测框架，不涉及资金曲线/持仓冲突/交易成本等。

---

## 性能与稳定性建议

- ALL 扫描时：
  - 勾选“扫描前跳过增量更新”
  - 提高 `min_amt20`（减少小票）
  - 控制 `batch_size` 与 `sleep_ms`，分批更新行情
- DB 维护：
  - 大量写入后如遇变慢，可手动 VACUUM（外部工具）
- baostock 登录
  - 若异常，重试或检查网络；程序已在增量拉取处做了重登与 sleep

---

## 常见问题（FAQ）

- Q: 启动提示 baostock 未安装？
  - A: `pip install baostock` 并重启程序
- Q: 字体乱码或图形不显示中文？
  - A: 安装中文字体；Matplotlib 已配置中文字体优先
- Q: Linux 没有 tkinter？
  - A: 安装 `python3-tk` 包；或使用虚拟环境确保 tk 库可用
- Q: ALL 扫描耗时较长？
  - A: 勾选“跳过增量更新”、提高流动性阈值、调小 TopN、分批更新
- Q: CSV 导入无法识别代码列？
  - A: 优先识别列名 `code/证券代码/ts_code/wind_code`，否则使用首列；代码自动归一化

---

## 目录与关键对象速览

- 全局状态：`app_state = {"horizon": 10}`
- 数据库路径：`DB_PATH = "advisor.db"`
- 日期格式：`DATE_FMT = "%Y-%m-%d"`

关键方法（选摘）：
- 数据
  - `init_db()`、`upsert_prices()`、`load_price_df()`
- 指标
  - `compute_indicators()`、`feature_columns()`
- 训练
  - `build_training_data()`、`train_ridge_with_cv()`、`evolve_ga_feature_selection()`
- 评估
  - `groupby_date_ic()`、`rank_spearman_ic()`、`evaluate_history_performance()`、`evaluate_quantile_performance()`
- 扫描/建议
  - `scan_market_and_rank()`、`gen_extended_advice()`、`score_latest_for_codes()`
- GUI
  - `EvoAdvisorApp`：各页签构建与事件回调

---

## 开发与二次扩展

- 添加新特征：
  - 在 `compute_indicators` 计算，并加入 `feature_columns`
  - 注意在 `build_training_data` 中的极值剪裁列表中按需添加
- 自定义质量分：
  - 修改 `quality_score_advanced` 的组合权重
- 更换模型：
  - 参考 `ridge_regression` 管线，增加新模型并在 `train_and_save_weights` 纳入分支
- 更多指数/股票池：
  - 在 `get_market_codes` 增加分支逻辑

---

## 重要提示

- 所有分析与建议均基于历史数据和简单模型，不保证未来表现
- 请谨慎使用仓位建议，必要时加入自有风控框架
- 数据依赖第三方接口（baostock），请注意稳定性与频率限制

---

## 许可证与声明

- 本项目用于学习与研究。作者不对任何因使用本工具导致的损失负责
- 若在研究中引用本工具，请注明来源

如需更详细的技术支持或二次开发指导，请提出你的具体需求与场景。