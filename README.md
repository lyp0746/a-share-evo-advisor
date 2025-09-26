# A股进化式投研助手（进化版）用户与技术文档

> 版本：v1.3.0  
> 项目名称：A股进化式投研助手（单文件 GUI）  
> 数据源：baostock  
> 数据持久化：SQLite  
> 图形界面：tkinter + matplotlib（TkAgg）

本项目提供一套集 数据增量更新、特征工程、模型训练（含进化/遗传特征选择）、市场扫描、建议生成、回测与绩效评估、用户操作记录 于一体的 A 股投研工作台。即使无已训练权重亦可运行（基线打分模式），适合快速探索与持续迭代。

本文档覆盖安装部署、核心概念、功能说明、操作流程、数据与模型细节、数据库结构、导入导出、常见问题与性能优化等，力求详细清晰、开箱即用。

---

## 目录

- [快速开始](#快速开始)
- [运行环境与安装](#运行环境与安装)
- [整体架构与数据流](#整体架构与数据流)
- [数据库结构](#数据库结构)
- [功能详解](#功能详解)
  - [股票池与数据更新](#股票池与数据更新)
  - [特征工程](#特征工程)
  - [训练与进化](#训练与进化)
  - [自选建议（扩展建议/仓位止损止盈）](#自选建议扩展建议仓位止损止盈)
  - [市场扫描（无权重亦可）](#市场扫描无权重亦可)
  - [回测（TopN 等权 + H 日持有）](#回测topn-等权--h-日持有)
  - [绩效评估与历史](#绩效评估与历史)
  - [用户操作记录](#用户操作记录)
  - [图表与可视化](#图表与可视化)
- [模型与指标细节](#模型与指标细节)
- [权重导入导出与格式](#权重导入导出与格式)
- [配置与日志](#配置与日志)
- [性能优化建议](#性能优化建议)
- [常见问题 FAQ](#常见问题-faq)
- [风险提示与免责声明](#风险提示与免责声明)

---

## 快速开始

1. 安装依赖
   - Python 3.8+（建议 3.9/3.10）
   - pip 安装：`pip install baostock pandas numpy matplotlib`

2. 运行程序
   - 命令行：`python a_share_evo_advisor.py`
   - 首次启动会初始化本地 SQLite 数据库 `advisor.db`，并尝试 baostock 登录（失败也可离线使用本地数据）

3. 快速使用
   - 页签“数据/股票池”：导入/输入股票池，设置日期与复权，点击“增量更新数据”
   - 页签“训练/进化”：设置 H（日）、训练模式与目标指标，点击“开始训练/进化”
   - 页签“自选建议”：填写资金与风险参数，生成扩展建议并保存（写入历史 advice 表）
   - 页签“市场扫描”：选择指数/池与过滤条件，开始扫描并导出 CSV
   - 页签“回测”：设定 TopN/H/费率/流动性阈值，运行回测查看年化、回撤、Sharpe、胜率
   - 页签“绩效/历史”：评估历史建议、做分位统计与收益分布
   - 页签“用户操作”：记录每日 BUY/SELL/ADJ 以留痕

---

## 运行环境与安装

- Python：3.8+（Win/Mac/Linux 均可）
- 依赖库：
  - baostock（数据源）
  - pandas / numpy（数据处理与数值计算）
  - matplotlib（绘图，TkAgg 后端）
  - tkinter（GUI，Linux 需单独安装：Ubuntu/Debian `sudo apt install python3-tk`）
- 字体：内置中文字体优先级 ["SimHei", "WenQuanYi Micro Hei", "Heiti TC", "sans-serif"]，确保中文图表显示正常

安装示例：
```bash
pip install baostock pandas numpy matplotlib
# Linux 若缺 tkinter：
# sudo apt-get update && sudo apt-get install -y python3-tk
python a_share_evo_advisor.py
```

---

## 整体架构与数据流

- 数据获取
  - 通过 baostock 拉取日线 K 数据，增量写入本地 SQLite（表：`prices`）
  - 支持复权选项：1-后复权、2-前复权、3-不复权

- 特征工程
  - 对本地数据计算技术指标与派生特征（SMA/EMA/MACD/RSI/ATR/BOLL/KDJ/CCI/WR/OBV/MFI/CMF/BBP/波动/动量/Gap/Range 等）

- 模型训练
  - 岭回归（带拦截，拦截不正则化），时间序列交叉验证选择 λ，目标可选 IC 或 MSE
  - 遗传算法（GA）进化特征子集 + λ，以验证 Rank IC 为适应度

- 生成建议与市场扫描
  - 无权重时也可使用“基线打分”排序
  - 支持扩展建议生成（仓位、止损、止盈、建议数量）

- 回测与评估
  - TopN 等权，H 日持有，计算期收益并年化
  - 历史建议绩效/分位统计/收益分布

- 持久化
  - SQLite：价格、建议、权重、元数据、股票名称缓存、用户操作记录
  - 最新权重自动标记于 meta.latest_weights_ts

---

## 数据库结构

数据库文件：`advisor.db`

- 表 `prices`（行情）
  - 主键：(code, date)
  - 字段：open/high/low/close/volume/amount
  - 索引：`idx_prices_date`、`idx_prices_code`

- 表 `advice`（建议历史）
  - 字段：date, code, score, advice, reasoning, horizon
  - 用于绩效评估与导出

- 表 `weights`（模型权重）
  - 字段：timestamp（主键）, horizon, features(JSON), weights(JSON), intercept, mu(JSON), sigma(JSON), lambda, notes
  - 元信息 `notes` 会保存方法、CV 结果、评估指标等

- 表 `meta`
  - 键值对存储，如 `('latest_weights_ts', timestamp)`

- 表 `stock_names`
  - 股票代码与中文名称缓存

- 表 `user_ops`
  - 用户操作记录：date, code, action(BUY/SELL/ADJ), price, qty, note

---

## 功能详解

### 股票池与数据更新

- 输入代码格式（自动归一化）：
  - 支持 `sh.600000` / `sz.000001`
  - 纯 6 位数字：`600000 -> sh.600000`；`000001/300xxx -> sz.000001`
  - 支持 `600000.SH / 000001.SZ`
- 股票池来源：
  - 手工输入（逗号/换行分隔）
  - 导入 `.txt`（每行一个）或 `.csv`（首列或名为 code/证券代码/ts_code/wind_code）
  - 从指数填充（HS300）；亦支持 ZZ500/SZ50/ALL/CUSTOM（扫描页）
- 增量更新：
  - 根据 DB 最新日期 +1 作为真实起始拉取，避免重复写入
  - 登录失败或未安装 baostock 时进入离线模式，仅使用本地数据
- 参数：
  - 日期范围（起始/结束）
  - 复权方式（1/2/3）

提示：指标计算与训练对样本量有最低需求。实践上，建议单票至少 80 根日线以上。

---

### 特征工程

核心函数：`compute_indicators(df)`

- 均线与 EMA：`sma5/10/20/60`、`ema12/26`
- MACD 系列：`dif, dea, macd`
- 振荡与通道：
  - RSI(14)：`rsi14`
  - KDJ(9,3,3)：`kdj_k, kdj_d, kdj_j`
  - CCI(14)：`cci14`
  - Williams%R(14)：`wr14`
  - Bollinger(20,2)：`bb_mid, bb_up, bb_low, bbp`
- 成交/资金：`obv, obv_ema, mfi14, cmf20`
- 波动与动量：`ret1, vol20(年化), mom10, ret5/20/60`
- 位置关系与状态：`above_sma20, dif_pos, macd_up`
- 振幅与缺口：`range_pct, gap_pct`
- 流动性与风险：`amt20（近20日均额）, atr14`
- 趋势/突破相关：`near_high55, dd60, sma20_slope5` 等

训练使用的特征列（`feature_columns()`）：
- 趋势/均线：sma5,sma10,sma20,sma60, ema12,ema26
- MACD：dif, dea, macd
- 震荡：rsi14, kdj_k, kdj_d, kdj_j, cci14, wr14
- 布林：bb_mid, bbp
- 成交/资金：v_surge, obv_ema, mfi14, cmf20
- 波动/动量：vol20, mom10, ret5, ret20, range_pct, gap_pct
- 位置：above_sma20, dif_pos, macd_up
- 风险：atr14

训练前处理：
- 目标收益：未来 \(H\) 日收益 \(y = \frac{Close_{t+H}}{Close_t} - 1\)，并裁剪到 [-0.25, 0.25]
- 特征极值剪裁（1/99 分位）：rsi14, v_surge, vol20, mom10, macd, dif, dea, bbp, kdj_j, cci14, wr14, range_pct, gap_pct, mfi14, cmf20
- 标准化：\(X_{std} = (X - \mu) / \sigma\)，保存 \(\mu, \sigma\) 以在线打分

数据最低需求：
- 训练：单票长度 ≥ max(80, H+40)
- 扫描：单票长度 ≥ 80
- 回测：单票长度 ≥ max(60, hold_days+10)

---

### 训练与进化

支持两种模式（页签“训练/进化”）：

1) 岭回归 + 时间序列 CV（`ridge_cv`）
- 带截距（拦截项不参与正则）。目标可选：
  - IC：最大化横截面 Rank IC 的验证均值
  - MSE：最小化均方误差
- 时间序列 CV：
  - 按交易日分段，使用前 i 段训练，第 i+1 段验证
- λ 搜索网格：`[1e-4, 3e-4, 1e-3, 3e-3, 1e-2, 3e-2, 1e-1]`
- 评估并持久化：保存权重与评估指标（IC 均值/标准差/ICIR/MSE/有效天数）

2) 遗传算法（`ga`）特征子集 + λ 进化
- 个体为特征选择掩码，适应度为验证集 Rank IC 均值
- 进化参数：种群 `n_pop`、代数 `n_gen`、子集大小约束 `min_feat/max_feat`
- 交叉、变异并保证子集大小约束，搜索最佳子集与 λ
- 拟合全样本并持久化

Rank IC 定义（横截面 Spearman）：
\[
IC_d = \rho\big(\mathrm{rank}(\hat{y}_d), \mathrm{rank}(y_d)\big), \quad
IC_{mean} = \frac{1}{D} \sum_{d=1}^{D} IC_d
\]
其中 \(d\) 为交易日维度；每个交易日要求样本数 ≥ 5。

模型方程（岭回归）：
\[
y \approx X w + b, \quad \arg\min_{w,b}\|y - Xw - b\mathbf{1}\|_2^2 + \lambda \|w\|_2^2
\]
拦截 \(b\) 不正则化。

---

### 自选建议（扩展建议/仓位止损止盈）

生成过程（可无权重）：
- 分数优先使用训练权重，否则使用“基线打分”
- 生成扩展建议项：
  - 止损：`stop = min(close*0.95, close - 1.5*ATR14)`（至少 -5%）
  - 止盈：`target = close + max(2.5*ATR, (close-stop)*1.5)`
  - 仓位比例：`pos_pct = min(30%, risk_pct / 跌幅%)`，其中 `跌幅% = (close - stop)/close`
  - 数量：按 100 股一手取整
  - 建议类型：`score>0 且 mom10≥-2%` -> “买入”，否则“观望”  
- 同时写入历史表 `advice`，便于后续绩效评估

“基线打分”（无模型时）核心思路：鼓励中短期动量、低波动、接近 55 日高、趋势成立，抑制高波动。

---

### 市场扫描（无权重亦可）

- 股票池：HS300/ZZ500/SZ50/ALL/CUSTOM
- 控制项：
  - 是否跳过增量更新（ALL 场景建议开启以提速）
  - 批处理大小与间隔（ALL 扫描限流）
  - 流动性阈值：近 20 日均额 `amt20 >= min_amt20`
  - 高级过滤：趋势（价>SMA20>SMA60 且 MACD>0 且 SMA20 上行）、接近 55 日高（≥98%）、波动 z 分上限、RSI 区间
- 打分：
  - 若有权重：标准化后线性打分
  - 否则：基线打分
- 质量分 qscore（用于排序，默认 TopN）：
\[
qscore = 0.5\cdot score\_z + 0.25\cdot mom\_z - 0.25\cdot vol\_z + 0.12\cdot trend + 0.08\cdot breakout55 + 0.05\cdot dd60
\]
其中各项为：score、mom10、vol20 的 z 分，trend（布尔）、breakout55（布尔）、dd60（[-1,0]，负值为回撤）

- 扩展建议：对扫描结果逐条生成并展示仓位/止损/止盈/数量，支持导出 CSV，同时写入 `advice` 历史

---

### 回测（TopN 等权 + H 日持有）

- 每持有期（`hold_days`）调仓一次
- 当日根据打分（权重或基线）选 TopN 等权持有下一期
- 期收益：简单 `Close_{t+H}/Close_t - 1`，扣除双边手续费 `fee_bps/10000 * 2`
- 流动性过滤：`amt20 >= min_amt20`
- 输出：
  - 期收益表（date, ret）
  - 净值曲线 `nav = cumprod(1+ret)`
  - 指标（按持有期为一个周期年化）：
    - 年化收益 CAGR
    - 最大回撤 MaxDD
    - 夏普比率 Sharpe
    - 胜率 WinRate

指标计算要点：
- 年化：以 252 交易日换算，周期为 hold_days
- MaxDD：按净值序列对峰值回撤
- Sharpe：使用周期收益的均值与标准差，按年化换算

---

### 绩效评估与历史

- 历史建议绩效（可按日期、类型 BUY/SELL、H 日）：
  - 将“卖出”建议对应收益取负，统计平均收益与胜率
- 分位统计：
  - 将 score 分为 q 分位，统计每组平均收益与胜率（买卖混合为签名收益）
- 收益分布图：
  - 绘制历史建议 H 日后收益直方图（便于观察偏度与尾部）

同时支持导出建议历史 CSV（附带股票名称）。

---

### 用户操作记录

- 记录日常交易行为：BUY/SELL/ADJ、价格、数量、备注
- 展示最近 300 条，长期留痕便于复盘与数据对齐

---

### 图表与可视化

- 单票绘图：收盘 + SMA20/60 + BOLL（上下轨），下方 MACD 柱 + DIF/DEA
- 操作入口：页签“图表”

---

## 模型与指标细节

- Rank Spearman IC（横截面）：
  - 对同一交易日的预测与真实未来收益进行排名，再计算皮尔逊相关
  - 稳健性：当标准差近零（所有排名相同）返回 NaN（训练中会跳过）
- 时间序列 CV 切分：
  - 将全部交易日均分为 `n_folds` 段
  - 第 i 折用前 i 段合并做训练，第 i+1 段作为验证，保持时间顺序
- 标准化与在线打分：
  - 训练时保存的 `mu/sigma` 用于推断期标准化，确保一致性
- 建议信号说明（示例）：
  - 规则型解释：如“股价站上SMA20”“MACD红柱”“DIF>0”“RSI超卖/超买”“动量10日：xx%”“放量”

分数→建议（用于“买入/卖出/观望”标签）：
- `score_to_advice` 默认阈值：`pos_thr=0.02`，`neg_thr=-0.02`
  - 分数 ≥ 0.03：“强烈买入”，≥ 0.02：“买入”
  - 分数 ≤ -0.03：“强烈卖出”，≤ -0.02：“卖出”
  - 其他：“观望”

---

## 权重导入导出与格式

- 导出最新权重：菜单“文件 → 导出最新权重…（JSON）”
- 导入权重：菜单“文件 → 导入权重…（JSON）”，会写入 `weights` 并更新 `meta.latest_weights_ts`

JSON 结构示例：
```json
{
  "timestamp": "2025-09-25 14:30:00",
  "horizon": 10,
  "features": ["sma5", "sma10", "..."],
  "weights": [0.01, -0.02, "..."],
  "intercept": 0.0005,
  "mu": [1.23, 4.56, "..."],
  "sigma": [0.78, 0.12, "..."],
  "lambda": 0.01,
  "notes": "imported or training metadata json"
}
```

注意：`features/mu/sigma/weights` 维度必须一致；导入后会作为新的最新权重参与扫描/建议/回测。

---

## 配置与日志

- 配置文件：`advisor_config.json`
  - 自动保存绝大多数 GUI 参数（日期、H、训练/扫描/回测配置、主题等）
  - 自定义股票池文件路径仅记录但不自动重新载入（避免路径失效）
- 日志文件：`advisor.log`
  - 记录关键操作与错误，便于排查

---

## 性能优化建议

- ALL 扫描时：
  - 勾选“扫描前跳过增量更新（快）”
  - 调整批大小与批间隔（接口限流）
  - 提高 `min_amt20` 限制，减少低流动性标的
- 训练/进化：
  - 先用 `ridge_cv` 粗选，IC 目标 + 合理折数
  - 再用 `ga` 在较小/精选股票池上搜索特征子集，提升性价比
- 数据：
  - 定期增量更新本地 DB，避免每次都拉全量
- 绘图：
  - 数据不足（如滚动窗口前期 NaN）会被自动过滤，保证图表可读性

---

## 常见问题 FAQ

- Q：baostock 登录失败或未安装？
  - A：`pip install baostock`，网络不佳可重试。未安装/未登录时进入离线模式，仅使用本地数据。
- Q：ALL 扫描太慢？
  - A：勾选“跳过增量更新”，提高 `min_amt20`，调大批大小（同时适当延时），必要时改用指数成分列表。
- Q：代码格式校验失败？
  - A：支持 `600000 / sh.600000 / 600000.SH`，会自动归一化；确保 6 位数字与交易所匹配（6→sh，0/3→sz）。
- Q：训练数据不足？
  - A：单票至少 ~80 根日线；训练期尽量覆盖 horizon + 40 以上；检查日期范围与复权方式一致性。
- Q：回测无结果？
  - A：确保股票池在回测日期内有足够样本；`min_amt20` 不要设置过高；`hold_days` 不宜超过总样本周期。
- Q：中文不显示或乱码？
  - A：确认系统已安装中文字体；或修改 `plt.rcParams["font.family"]` 指定可用中文字体。
- Q：如何清空或重建数据库？
  - A：关闭程序后删除 `advisor.db`（会丢失历史与权重记录），重启会自动初始化表结构。

---

## 风险提示与免责声明

- 本程序仅用于教育与研究，不构成任何投资建议。
- 历史回测与模拟结果不代表未来表现，市场有风险，投资需谨慎。
- 使用 baostock 等第三方数据源请遵守其服务条款与访问限制。

---

## 附录：关键公式与实现要点

- 横截面 Rank IC（每个交易日）：
\[
IC_d = \frac{\sum_i (r_i - \bar{r})(q_i - \bar{q})}{\sqrt{\sum_i (r_i-\bar{r})^2}\sqrt{\sum_i (q_i-\bar{q})^2}}
\]
其中 \(r_i\) 为预测值的名次，\(q_i\) 为真实收益的名次。

- qscore 组合：
\[
qscore = 0.5\cdot score\_z + 0.25\cdot mom\_z - 0.25\cdot vol\_z + 0.12\cdot trend + 0.08\cdot breakout55 + 0.05\cdot dd60
\]

- 头寸建议：
  - 跌幅比：\(\mathrm{drop\_pct} = \frac{close - stop}{close}\)
  - 仓位：\(\mathrm{pos\_pct} = \min(0.3, \frac{\mathrm{risk\_pct}}{\mathrm{drop\_pct}})\)

- 回测指标（周期 = hold_days）：
  - 年化收益：\(\mathrm{CAGR} = \mathrm{NAV}_{end}^{252 / N} - 1\)，\(N\) 为净值点数
  - 最大回撤：\(\min(\mathrm{NAV}/\mathrm{cummax(NAV)} - 1)\)
  - 夏普比率：\(\mathrm{Sharpe} = \frac{\mu \cdot (252/hold\_days)}{\sigma \cdot \sqrt{252/hold\_days}}\)

---

如需二次开发或部署到服务器侧（无 GUI 场景），可将核心函数（数据、特征、训练、扫描、回测、评估）封装为服务接口或脚本任务，GUI 相关类 `EvoAdvisorApp` 可按需裁剪。欢迎基于此框架继续进化（特征工程、目标函数、交易逻辑、风险控制）。祝研究顺利！