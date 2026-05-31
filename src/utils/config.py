"""
配置加载与验证模块
==================
基于 Pydantic 的统一配置管理，支持 YAML 配置文件加载与类型校验。
"""

from pathlib import Path
from typing import Any, Dict, List, Optional

import yaml
from pydantic import BaseModel, Field, field_validator
from loguru import logger


# ============================================================================
# Pydantic 配置模型定义
# ============================================================================

class DataSourceConfig(BaseModel):
    """数据源配置"""
    name: str = Field(..., description="数据源名称")
    type: str = Field(default="csv", description="数据源类型：csv/database/api")
    path: Optional[str] = Field(default=None, description="文件路径或连接字符串")
    table: Optional[str] = Field(default=None, description="数据库表名")
    encoding: str = Field(default="utf-8", description="文件编码")


class ESGWeightItem(BaseModel):
    """ESG权重项"""
    industry: str = Field(..., description="行业名称")
    E_weight: float = Field(..., ge=0, le=1, description="环境权重")
    S_weight: float = Field(..., ge=0, le=1, description="社会权重")
    G_weight: float = Field(..., ge=0, le=1, description="治理权重")


class IndustryLinkageItem(BaseModel):
    """行业关联项"""
    source: str = Field(..., description="上游行业")
    target: str = Field(..., description="下游行业")
    coefficient: float = Field(..., ge=0, le=1, description="传导系数")


class ScenarioParams(BaseModel):
    """情景参数"""
    name: str = Field(..., description="情景名称")
    revenue_growth: float = Field(..., description="营收增长率")
    margin_change: float = Field(default=0.0, description="利润率变化")
    wacc: float = Field(..., gt=0, description="加权平均资本成本")
    terminal_growth: float = Field(..., ge=0, le=0.05, description="永续增长率")
    esg_premium: float = Field(default=0.0, description="ESG溢价/折价")


class ModelParamsConfig(BaseModel):
    """模型参数配置"""
    learning_rate: float = Field(default=0.05, gt=0, description="学习率")
    n_estimators: int = Field(default=200, gt=0, description="树的数量")
    max_depth: int = Field(default=7, gt=0, description="最大深度")
    num_leaves: int = Field(default=31, gt=0, description="叶子节点数")
    min_child_samples: int = Field(default=20, gt=0, description="最小子节点样本数")
    subsample: float = Field(default=0.8, ge=0.1, le=1.0, description="样本采样率")
    colsample_bytree: float = Field(default=0.8, ge=0.1, le=1.0, description="特征采样率")
    reg_alpha: float = Field(default=0.1, ge=0, description="L1正则化")
    reg_lambda: float = Field(default=0.1, ge=0, description="L2正则化")


class SentimentFactorWeights(BaseModel):
    """情绪因子权重"""
    northbound: float = Field(default=0.30, ge=0, le=1, description="北向资金权重")
    margin: float = Field(default=0.25, ge=0, le=1, description="两融余额权重")
    turnover: float = Field(default=0.20, ge=0, le=1, description="换手率权重")
    sentiment_text: float = Field(default=0.25, ge=0, le=1, description="舆情文本权重")


class FusionWeights(BaseModel):
    """融合权重配置"""
    dcf_weight: float = Field(default=0.35, ge=0, le=1, description="DCF估值权重")
    relative_weight: float = Field(default=0.25, ge=0, le=1, description="相对估值权重")
    esg_weight: float = Field(default=0.20, ge=0, le=1, description="ESG因子权重")
    sentiment_weight: float = Field(default=0.20, ge=0, le=1, description="市场情绪权重")


class AppSettings(BaseModel):
    """
    应用主配置模型
    ================
    所有配置项通过 Pydantic 进行类型验证，确保配置正确性。
    """
    # 基础设置
    project_name: str = Field(default="ESG Insight Valuator", description="项目名称")
    version: str = Field(default="1.0.0", description="版本号")
    data_dir: str = Field(default="data", description="数据目录")
    output_dir: str = Field(default="output", description="输出目录")
    models_dir: str = Field(default="models", description="模型存储目录")

    # 数据源配置
    data_sources: List[DataSourceConfig] = Field(default_factory=list)

    # ESG权重配置
    esg_weights: List[ESGWeightItem] = Field(default_factory=list)

    # 行业关联配置
    industry_linkages: List[IndustryLinkageItem] = Field(default_factory=list)

    # 情景参数
    scenarios: List[ScenarioParams] = Field(default_factory=list)

    # 模型参数
    model_params: Optional[ModelParamsConfig] = None

    # 情绪因子权重
    sentiment_weights: Optional[SentimentFactorWeights] = None

    # 融合权重
    fusion_weights: Optional[FusionWeights] = None

    # 投资建议参数
    advice_threshold_buy: float = Field(default=0.15, description="买入阈值（低估比例）")
    advice_threshold_sell: float = Field(default=-0.10, description="卖出阈值（高估比例）")
    confidence_level: float = Field(default=0.95, description="置信水平")

    # 回测参数
    backtest_start: str = Field(default="2020-01-01", description="回测起始日期")
    backtest_end: str = Field(default="2025-12-31", description="回测结束日期")
    rebalance_frequency: str = Field(default="monthly", description="调仓频率")


# ============================================================================
# 配置加载函数
# ============================================================================

def load_yaml(file_path: str) -> Dict[str, Any]:
    """
    加载 YAML 配置文件。

    Parameters
    ----------
    file_path : str
        YAML 文件路径

    Returns
    -------
    dict
        解析后的配置字典

    Raises
    ------
    FileNotFoundError
        配置文件不存在时抛出
    yaml.YAMLError
        YAML 格式错误时抛出
    """
    path = Path(file_path)
    if not path.exists():
        raise FileNotFoundError(f"配置文件不存在: {file_path}")

    with open(path, "r", encoding="utf-8") as f:
        data = yaml.safe_load(f)

    logger.info(f"成功加载配置文件: {file_path}")
    return data if data is not None else {}


def load_app_settings(config_dir: str = "config") -> AppSettings:
    """
    加载并验证应用配置。

    从 config/ 目录加载所有 YAML 配置文件，合并后通过 Pydantic 验证。

    Parameters
    ----------
    config_dir : str
        配置文件目录路径

    Returns
    -------
    AppSettings
        验证通过的应用配置对象
    """
    config_path = Path(config_dir)
    merged: Dict[str, Any] = {}

    # 按顺序加载各配置文件
    config_files = [
        "settings.yaml",
        "industry_weights.yaml",
        "industry_linkages.yaml",
        "scenario_params.yaml",
        "model_params.yaml",
    ]

    for file_name in config_files:
        file_path = config_path / file_name
        if file_path.exists():
            data = load_yaml(str(file_path))
            merged.update(data)
        else:
            logger.warning(f"配置文件不存在，跳过: {file_path}")

    # 通过 Pydantic 验证并构建配置对象
    try:
        settings = AppSettings(**merged)
        logger.info("应用配置验证通过")
        return settings
    except Exception as e:
        logger.error(f"配置验证失败: {e}")
        raise


def get_config_paths(config_dir: str = "config") -> Dict[str, str]:
    """
    获取各配置文件路径映射。

    Parameters
    ----------
    config_dir : str
        配置文件目录

    Returns
    -------
    dict
        配置名到文件路径的映射
    """
    base = Path(config_dir)
    return {
        "settings": str(base / "settings.yaml"),
        "industry_weights": str(base / "industry_weights.yaml"),
        "industry_linkages": str(base / "industry_linkages.yaml"),
        "scenario_params": str(base / "scenario_params.yaml"),
        "model_params": str(base / "model_params.yaml"),
    }
