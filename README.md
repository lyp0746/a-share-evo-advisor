## A股进化式投研助手（baostock + Ridge 回归 + GUI）

本项目提供一个基于 baostock 的 A 股“进化式”投研助手，支持数据增量更新、特征工程、Ridge 回归自我进化训练、个股/市场扫描打分、ATR 扩展交易建议（仓位/止损/止盈/下单数量）、历史建议绩效评估、结果持久化（SQLite）以及图形界面交互（tkinter + matplotlib）。

- 文件：`a_share_evo_advisor.py`（单文件 GUI 程序）
- 数据库：`advisor.db`（SQLite，自动创建）
- 环境：Python 3.9+（推荐 3.10–3.12）

请注意：本程序仅用于教育与研究目的，不构成任何投资建议。

---

## 功能亮点

- 增量数据更新：基于本地 DB 最新交易日，仅拉取新增数据，减少调用与等待
- 特征工程：SMA、EMA、MACD、RSI、波动率、动量、成交量放大、ATR 等常见指标
- 自我进化：以历史特征拟合未来 \(H\) 日收益，Ridge 回归带截距、标准化
- 个股打分与建议：对自选股给出“预测分”，并生成扩展建议（仓位/止损/止盈/数量）
- 市场扫描：支持 HS300 / ZZ500 / SZ50 / 全部A股，按“质量分”排序筛选优质标的
- 绩效评估：根据历史建议与实际未来收益统计类别表现（均值、胜率）
- 可视化：收盘价与均线、MACD/DIF/DEA、ATR 走图
- 持久化：价格、建议与权重均落库（SQLite）

---

## 运行截图（功能概览）

- 顶部参数区：股票代码、起止日期、预测视野 \(H\)、岭回归 \(\lambda\)、复权方式
- 市场扫描/风险参数：指数范围、流动性门槛、TopN、资金规模、单笔风险占比
- 操作按钮：更新数据、训练权重、自选股建议、市场扫描、评估历史、查看历史、绘图
- 建议表格：显示 date、code、score、advice、reasoning、horizon
- 日志区：实时输出操作日志/错误

---

## 安装与环境准备

### 1) 准备 Python

- 推荐 Python 3.10–3.12（64 位）
- 确保安装了 tkinter（GUI 所需）
  - Windows：使用 python.org 官方安装包，默认包含 tkinter
  - macOS：
    - 推荐使用 python.org 的官方安装包（自带 Tk）
    - 或使用 Conda（`conda install tk`）
  - Ubuntu/Debian：`sudo apt-get install python3-tk`
  - CentOS/RHEL：`sudo yum install python3-tkinter`

- Matplotlib 使用 `TkAgg` 后端，如遇 GUI 后端错误，请确认 tkinter 已安装

### 2) Python 依赖安装

- 使用虚拟环境（推荐）

  - venv
    ```
    python -m venv .venv
    # Windows
    .venv\Scripts\activate
    # macOS / Linux
    source .venv/bin/activate
    ```

  - pip 安装依赖
    ```
    pip install -r requirements.txt
    ```

  - 国内镜像（可选）
    ```
    pip install -r requirements.txt -i https://pypi.tuna.tsinghua.edu.cn/simple
    ```

### 3) 中文字体（可选但推荐）

本程序在 matplotlib 中启用了中文字体配置。如系统缺少中文字体，建议安装：
- Ubuntu/Debian：
  ```
  sudo apt-get install fonts-wqy-microhei fonts-wqy-zenhei
  ```
- Windows/macOS 通常已具备中文字体（或自行安装）

---

## 快速开始

1. 激活虚拟环境并安装依赖（见上节）
2. 运行程序
   ```
   python a_share_evo_advisor.py
   ```
3. GUI 操作流程
   - 在“股票代码”中输入代码列表，例如：`sh.600000,sz.000001,sz.300750`
   - 点击“1) 增量更新数据”拉取历史 K 线
   - 点击“2) 训练/进化权重”得到模型权重
   - 点击“3) 自选股生成建议(扩展)”获取仓位/止损/止盈/数量建议
   - 点击“市场扫描优质股”对所选指数范围进行扫描排序并持久化建议
   - 点击“评估历史建议”查看建议类别的收益统计
   - 点击“查看建议历史”浏览历史建议详情
   - 点击“绘图(所选股票)”展示技术指标图表

提示：
- 首次使用建议将起始日期设为较早时间，例如 `2015-01-01`
- 复权方式：`1=后复权`、`2=前复权(默认)`、`3=不复权`

---

## 参数说明与策略细节

### 训练目标与回归
- 目标收益：未来 \(H\) 日收益 \(r_{t,H} = \frac{Close_{t+H}}{Close_t} - 1\)
- 特征：
  - 均线：`sma5, sma10, sma20, sma60`
  - 指数均线：`ema12, ema26`
  - MACD 及衍生：`dif, dea, macd`
  - RSI：`rsi14`
  - 波动率：`vol20`（年化近 20 日）
  - 动量：`mom10`
  - 成交量相关：`v_surge`（相对 5 日均量的放大倍数）
  - 位置/信号：`above_sma20, dif_pos, macd_up`
- 训练细节：
  - 对目标收益与部分特征进行剪裁以增强稳健性
  - 标准化：对每列计算均值/标准差并应用
  - Ridge 回归（带截距），权重正则化，截距不正则化
  - 训练完成后将权重、标准化参数、特征列、\(\lambda\) 与备注保存到 DB

### 个股打分与建议
- 预测分 `score = xz · w + b`，近似为未来 \(H\) 日收益估计
- 基础建议门槛：`pos_thr=0.02`、`neg_thr=-0.02` → 强烈买入/买入/观望/卖出/强烈卖出
- 扩展建议（ATR 驱动）：
  - 止损：`min(收盘价*0.95, 收盘价 - 1.5*ATR14)`
  - 止盈：`收盘价 + max(2.5*ATR14, (收盘价-止损)*1.5)`
  - 仓位上限：30%，并按“单笔风险占比 / 止损跌幅”折算
  - 数量：按 100 股一手向下取整

### 市场扫描与质量分
- 指数范围：`HS300`、`ZZ500`、`SZ50`、`ALL(全A)`
- 流动性过滤：近 20 日平均成交额 `amt20 ≥ min_amt20`
- 质量分：
  \[
  qscore = 0.7 \cdot score + 0.2 \cdot mom10 - 0.1 \cdot vol\_z + trend\_bonus
  \]
  - `vol_z` 为波动率标准分
  - `trend_bonus`：若 `close > sma20 > sma60` 则 +0.02

### 绩效评估
- 基于已持久化的历史“建议记录”，对`强烈买入/买入/观望/卖出/强烈卖出`聚合
- 买入类使用“正向未来收益”，卖出类使用“反向收益”
- 输出：样本数、平均收益、胜率

---

## 代码结构（关键函数）

- 数据库/持久化
  - `init_db()`：初始化 SQLite 表结构
  - `upsert_prices()`：价格数据落库（主键：code+date）
  - `insert_advice()`、`query_advice()`：建议的写入/查询
  - `save_weights_record()`、`load_latest_weights()`：模型权重的保存/读取

- 数据获取（baostock）
  - `bs_safe_login()/logout()`：安全登录/登出
  - `fetch_k_data_incremental()`：历史 K 线增量拉取

- 特征工程与训练
  - `compute_indicators()`：技术指标计算
  - `build_training_data()`：构建特征矩阵与未来收益
  - `standardize_fit()/standardize_apply()`：标准化
  - `ridge_regression()`：带截距的 Ridge 回归（闭式解）

- 打分/建议/扫描
  - `score_to_advice()`、`reasoning_from_signals()`
  - `train_and_save_weights()`：训练并保存权重
  - `score_latest_for_codes()`：对自选股打分
  - `get_market_codes()`：指数成分/全市场代码列表
  - `scan_market_and_rank()`：市场扫描、打分、排序、TopN 输出
  - `gen_extended_advice()`：扩展交易建议（ATR/仓位/数量）

- 绩效与可视化
  - `evaluate_history_performance()`：历史建议绩效
  - `_plot_matplotlib()`：价格/指标可视化

- GUI 主程序
  - `EvoAdvisorApp`：tkinter 界面、事件绑定、表格与弹窗
  - `main()`：入口（`init_db()` + GUI）

---

## 数据库结构（SQLite）

- 表 `prices`（主键：`(code, date)`）
  - `code TEXT, date TEXT, open REAL, high REAL, low REAL, close REAL, volume REAL, amount REAL`

- 表 `advice`
  - `id INTEGER PRIMARY KEY AUTOINCREMENT`
  - `date TEXT, code TEXT, score REAL, advice TEXT, reasoning TEXT, horizon INTEGER`

- 表 `weights`（主键：`timestamp`）
  - `timestamp TEXT`
  - `horizon INTEGER`
  - `features TEXT`（JSON list[str]）
  - `weights TEXT`（JSON list[float]）
  - `intercept REAL`
  - `mu TEXT`（JSON list[float]）
  - `sigma TEXT`（JSON list[float]）
  - `lambda REAL`
  - `notes TEXT`

- 表 `meta`
  - `key TEXT PRIMARY KEY, value TEXT`
  - 用途：存储 `latest_weights_ts`

- DB 文件：默认当前目录 `advisor.db`

---

## 使用技巧与实践建议

- 初训建议：
  - 股票池以流动性较好的标的为主（如 HS300 成分或你的自选核心）
  - 预测视野 \(H\)：10 或 20 日；\(\lambda\)：`1e-2` 起试
- 数据完整性：
  - 定期点击“增量更新”，保证训练/打分可用数据充足
- 风险参数：
  - `单笔风险%` 建议从小到大试，常见如 `0.5% ~ 1%`
- 结果解释：
  - `score` 更贴近“未来收益”的点估计
  - `qscore` 综合动量/波动/趋势，适于“榜单排序”

---

## 常见问题（FAQ）

- Q: 打开程序报 “ImportError: No module named 'tkinter'”
  - A: 安装 tkinter（见“安装与环境准备”）
- Q: “matplotlib is currently using agg, which is a non-GUI backend”
  - A: 确认 tkinter 安装，并使用 GUI 环境运行；或在 Conda 环境中 `conda install tk matplotlib`
- Q: “未安装 baostock，请先执行：pip install baostock”
  - A: 安装依赖：`pip install -r requirements.txt`
- Q: baostock 调用失败或超时
  - A: 重试，或切换网络；程序内部自带重登与重试逻辑
- Q: 中文乱码
  - A: 安装中文字体（文泉驿等），或在代码中将 `plt.rcParams["font.family"]` 改为你系统已有的中文字体

---

## 兼容性与系统依赖

- 操作系统：Windows / macOS / Linux（X11/Wayland）
- Python：3.9–3.12
- 图形后端：matplotlib `TkAgg`（依赖 tkinter）
- 字体：中文字体建议安装（WenQuanYi / SimHei 等）

---

## 开发与二次扩展建议

- 策略层：
  - 新增特征列（如 KDJ、布林带、换手率），在 `feature_columns()` 中一并纳入
  - 更复杂的模型（如 Lasso/ElasticNet/Tree/Boosting），注意保存/加载参数接口
- 数据层：
  - 可替换为其他数据源（Tushare/JoinQuant 等），对齐 `fetch_k_data_incremental()` 与 `get_market_codes()`
- 评估层：
  - 引入时间分层回测，或滚动训练/验证切分
- GUI 层：
  - 增加导入导出配置、批量任务、回测面板

---

## 性能与稳定性提示

- 市场扫描 ALL（全 A）数据量较大，建议合理设置日期范围与流动性门槛
- baostock 有访问频控，程序已做 `time.sleep` 节流；必要时可增大间隔
- SQLite 为单文件数据库，I/O 较轻；如并发访问请做互斥

---

## 法律与免责声明

- 本程序仅用于教育与研究目的，不构成任何投资建议
- 使用者需遵守所在地区法律法规与数据源使用条款
- 作者不对因使用本程序产生的任何损失承担责任

---

## 运行命令小抄

- 安装依赖
  ```
  pip install -r requirements.txt
  ```
- 启动 GUI
  ```
  python a_share_evo_advisor.py
  ```

---

## 变更记录（示例）

- v0.1.0
  - 初版发布：增量拉取、Ridge 训练、扩展建议、市场扫描、绩效评估、GUI

---

## 许可证

- 可根据你的项目选择合适的开源协议（如 MIT/Apache-2.0），并在仓库中添加 LICENSE 文件