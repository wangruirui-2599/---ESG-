# 📊 ESG Insight Valuator (EIV)

**ESG智能估值分析系统** — 将 ESG 因子融入企业估值全流程的量化分析框架。

[![Python](https://img.shields.io/badge/Python-3.10+-blue.svg)](https://www.python.org/)
[![License](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)

---

## 🎯 项目概述

EIV 是一个将 ESG（环境、社会、治理）因素系统化融入企业估值和投资决策的 Python 量化分析系统。它通过 **数据管道 → ESG量化 → 异常检测 → 多情景估值 → 因子融合 → 概率建议 → 回测验证** 的完整管线，输出可操作的投资建议和可视化报告。

### 核心能力

| 模块 | 功能 |
|------|------|
| 📥 数据管道 | 多源数据加载、另类数据融合（舆情/专利/供应链）、特征工程 |
| 🌱 ESG 量化 | 行业动态权重、产业链风险传导、ESG 趋势分析 |
| 🔍 异常检测 | LightGBM 财务异常预警模型 |
| 💰 估值引擎 | 多情景 DCF + 情绪校准 |
| 🧩 因子融合 | DCF × 相对估值 × ESG × 情绪，行业动态权重 |
| 🎯 投资建议 | 概率化决策、上下行风险量化 |
| 📈 回测验证 | 历史回测 + DID 因果推断 |
| 📋 报告生成 | HTML/Markdown 自动报告 + 可视化图表 |

---

## 🏗️ 项目结构

```
esg-insight-valuator/
├── config/                         # 配置文件
│   ├── settings.yaml               # 主配置
│   ├── industry_weights.yaml       # 行业ESG权重
│   ├── industry_linkages.yaml      # 行业关联矩阵
│   ├── scenario_params.yaml        # DCF情景参数
│   └── model_params.yaml           # 模型超参数
├── src/
│   ├── data_pipeline/              # 数据层
│   │   ├── loader.py               # 多源数据加载
│   │   ├── alternative_data.py     # 另类数据融合
│   │   └── feature_engineering.py  # 特征工程
│   ├── esg_quant/                  # ESG量化
│   │   ├── dynamic_weights.py      # 动态权重引擎
│   │   ├── contagion.py            # 风险传导分析
│   │   └── trend_analyzer.py       # ESG趋势分析
│   ├── anomaly/                    # 异常检测
│   │   └── predictor.py            # LightGBM预警模型
│   ├── valuation/                  # 估值引擎
│   │   ├── scenario.py             # 多情景DCF
│   │   └── sentiment_calibration.py # 情绪校准
│   ├── fusion/                     # 因子融合
│   │   └── four_factor.py          # 四因子融合
│   ├── advice/                     # 投资建议
│   │   └── probabilistic.py        # 概率建议引擎
│   ├── backtest/                   # 回测验证
│   │   ├── engine.py               # 回测引擎
│   │   └── causal.py               # DID因果推断
│   ├── reporting/                  # 报告生成
│   │   ├── report_generator.py     # 报告生成器
│   │   └── visualizer.py           # 可视化图表
│   └── utils/                      # 工具
│       ├── logger.py               # 日志配置
│       ├── config.py               # 配置加载(基础版)
│       └── config_loader.py        # 配置加载(增强版)
├── scripts/
│   ├── run_full_pipeline.py        # 主运行管线
│   ├── generate_sample_data.py     # 示例数据生成
│   └── generate_industry_reports.py # 行业综合报告生成
├── tests/                          # 测试用例
├── data/
│   ├── raw/                        # 原始数据
│   ├── processed/                  # 中间结果
│   └── external/                   # 外部另类数据
├── models/                         # 训练好的模型
├── output/
│   ├── figures/                    # 图表输出
│   └── reports/                    # 报告输出
├── logs/                           # 日志文件
├── requirements.txt                # 依赖包
└── README.md                       # 本文件
```

---

## 🚀 快速开始

### 1. 环境要求

- Python 3.10+
- pip

### 2. 安装依赖

```bash
pip install -r requirements.txt
```

### 3. 生成示例数据

```bash
python scripts/generate_sample_data.py
```

### 4. 运行完整管线

```bash
# 运行全部步骤
python scripts/run_full_pipeline.py --step all

# 分步运行（便于调试）
python scripts/run_full_pipeline.py --step load
python scripts/run_full_pipeline.py --step feature
python scripts/run_full_pipeline.py --step esg
python scripts/run_full_pipeline.py --step anomaly
python scripts/run_full_pipeline.py --step valuation
python scripts/run_full_pipeline.py --step fusion
python scripts/run_full_pipeline.py --step advice
python scripts/run_full_pipeline.py --step backtest
python scripts/run_full_pipeline.py --step report
```

### 5. 生成行业综合报告

```bash
python scripts/generate_industry_reports.py
```

### 6. 指定参数

```bash
# 自定义配置和数据目录
python scripts/run_full_pipeline.py --step all --config config/ --data-dir data/ --output-dir output/

# 只分析特定股票
python scripts/run_full_pipeline.py --step all --stock-code 000001

# 调试模式
python scripts/run_full_pipeline.py --step all --log-level DEBUG
```

---

## 📊 输出说明

运行完成后，在 `output/` 目录下生成：

| 路径 | 内容 |
|------|------|
| `output/reports/eiv_report_*.html` | 单只股票的HTML分析报告 |
| `output/reports/eiv_report_*.md` | Markdown格式报告 |
| `output/reports/eiv_batch_report.html` | 全部股票汇总报告 |
| `output/reports/eiv_industry_comprehensive_report.html` | 行业综合对比报告 |
| `output/figures/industry_esg_ranking.png` | 行业ESG排名图 |
| `output/figures/industry_esg_timeseries.png` | ESG趋势时间序列 |
| `output/figures/industry_multi_radar.png` | 多行业雷达对比 |
| `output/figures/industry_attractiveness.png` | 投资吸引力矩阵 |
| `data/processed/*.parquet` | 各步骤中间结果 |

---

## 🔧 配置说明

### settings.yaml — 主配置

```yaml
project_name: "ESG Insight Valuator"
data_sources:           # 定义所有数据源
  - name: "financials"
    type: "csv"
    path: "data/raw/financials.csv"
advice_threshold_buy: 0.15    # 买入阈值(15%上行)
advice_threshold_sell: -0.10  # 卖出阈值(-10%下行)
backtest_start: "2020-01-01"  # 回测开始日期
```

### industry_weights.yaml — 行业ESG权重

```yaml
esg_weights:
  - industry: "石油石化"
    E_weight: 0.45    # 环境权重45%(高污染行业)
    S_weight: 0.25
    G_weight: 0.30
```

### scenario_params.yaml — DCF情景

```yaml
scenarios:
  - name: "乐观"
    revenue_growth: 0.15
    wacc: 0.08
    probability: 0.25
  - name: "中性"
    revenue_growth: 0.08
    wacc: 0.10
    probability: 0.50
  - name: "悲观"
    revenue_growth: 0.02
    wacc: 0.12
    probability: 0.25
```

---

## 🧪 在 Python 中使用

```python
from src.valuation.scenario import DCFValuator
from src.fusion.four_factor import FourFactorFusion
from src.advice.probabilistic import ProbabilisticAdvisor

# DCF估值
valuator = DCFValuator()
result = valuator.value_expected(
    stock_code="000001",
    base_revenue=200,        # 200亿营收
    base_margin=0.15,        # 15%利润率
    current_price=12.50,     # 当前股价
    total_shares=50,         # 50亿股
)

# 四因子融合
fusion = FourFactorFusion()
final = fusion.fuse(
    dcf_value=result["expected_value"],
    relative_value=11.0,
    esg_adjusted_value=result["expected_value"] * 1.05,
    sentiment_adjusted_value=result["expected_value"] * 0.98,
    industry="银行",
)

# 投资建议
advisor = ProbabilisticAdvisor()
advice = advisor.evaluate(
    optimistic_value=15.0,
    neutral_value=result["expected_value"],
    pessimistic_value=9.0,
    current_price=12.50,
    esg_trend_score=2.5,
    anomaly_probability=0.1,
)

print(f"建议: {advice['advice']}")
print(f"置信度: {advice['confidence']:.0%}")
```

---

## 🤝 贡献指南

欢迎提交 Issue 和 Pull Request。主要开发分支：`main`

---

## 📄 许可证

MIT License

---

*ESG Insight Valuator — 让 ESG 因子量化可操作*
