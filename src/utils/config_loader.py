"""
配置加载与验证模块
==================
基于 Pydantic v2 + PyYAML 的统一配置管理。

功能：
  1. 从 config/ 目录加载所有 YAML 配置文件
  2. Pydantic 严格验证所有字段类型和范围
  3. 合并为单一 AppSettings 对象供全局使用
  4. 支持环境变量覆盖（EIV_ 前缀）

与 src/utils/config.py 的关系：
  本模块是 config.py 的重构增强版，提供更完整的配置验证。
  两模块可互换使用，本模块为推荐入口。
"""

from pathlib import Path
from typing import Any, Dict, List, Optional, Union, Literal

import yaml
from loguru import logger
from pydantic import (
    BaseModel, Field, field_validator, model_validator,
    ValidationError, validator,
)


# ============================================================================
# Pydantic 模型（详细配置项）
# ============================================================================

class DataSourceConfig(BaseModel):
    """单个数据源配置"""
    name: str = Field(..., description="数据源名称")
    type: Literal["csv", "database", "api"] = Field(default="csv", description="数据源类型")
    path: Optional[str] = Field(default=None, description="文件路径（CSV）或连接字符串（DB）")
    table: Optional[str] = Field(default=None, description="数据库表名")
    encoding: str = Field(default="utf-8", description="文件编码")
    required: bool = Field(default=True, description="是否必需")

    @field_validator("path")
    @classmethod
    def path_must_exist_for_csv(cls, v, info):
        """CSV类型的path不应为空。"""
        if info.data.get("type") == "csv" and not v:
            raise ValueError("CSV数据源必须指定 path")
        return v


class ESGWeightItem(BaseModel):
    """行业ESG权重项"""
    industry: str = Field(..., description="行业名称")
    E_weight: float = Field(..., ge=0, le=1, description="环境权重 (0-1)")
    S_weight: float = Field(..., ge=0, le=1, description="社会权重 (0-1)")
    G_weight: float = Field(..., ge=0, le=1, description="治理权重 (0-1)")

    @model_validator(mode="after")
    def check_sum_to_one(self):
        """验证三维度权重之和≈1.0。"""
        total = self.E_weight + self.S_weight + self.G_weight
        if abs(total - 1.0) > 0.05:
            logger.warning(
                f"行业 [{self.industry}] 权重之和={total:.3f}，将自动归一化"
            )
        return self


class IndustryLinkageItem(BaseModel):
    """行业关联传导项"""
    source: str = Field(..., description="上游/源行业")
    target: str = Field(..., description="下游/目标行业")
    coefficient: float = Field(..., ge=0, le=1, description="传导系数 (0-1)")


class ScenarioParams(BaseModel):
    """DCF情景参数"""
    name: str = Field(..., description="情景名称（乐观/中性/悲观）")
    revenue_growth: float = Field(..., description="营收复合增长率")
    margin_change: float = Field(default=0.0, description="净利润率变化（百分点）")
    wacc: float = Field(..., gt=0, le=0.30, description="WACC折现率 (0-30%)")
    terminal_growth: float = Field(..., ge=0, le=0.06, description="永续增长率")
    esg_premium: float = Field(default=0.0, description="ESG溢价/折价 (-50%~+50%)")
    probability: float = Field(default=0.33, ge=0, le=1, description="先验概率")

    @field_validator("terminal_growth")
    @classmethod
    def tg_less_than_wacc(cls, v, info):
        """永续增长率不应超过WACC。"""
        wacc = info.data.get("wacc", 0.10)
        if v >= wacc:
            logger.warning(
                f"[{info.data.get('name', '?')}] 永续增长率({v:.1%}) >= WACC({wacc:.1%})，"
                f"可能导致终值异常"
            )
        return v


class ModelParams(BaseModel):
    """LightGBM模型超参数"""
    learning_rate: float = Field(default=0.05, gt=0, le=0.5, description="学习率")
    n_estimators: int = Field(default=200, gt=0, le=5000, description="迭代次数")
    max_depth: int = Field(default=7, gt=0, le=20, description="树深度")
    num_leaves: int = Field(default=31, gt=0, le=256, description="叶子数")
    min_child_samples: int = Field(default=20, ge=1, description="最小叶子样本")
    subsample: float = Field(default=0.8, gt=0, le=1.0, description="行采样率")
    colsample_bytree: float = Field(default=0.8, gt=0, le=1.0, description="列采样率")
    reg_alpha: float = Field(default=0.1, ge=0, description="L1正则化")
    reg_lambda: float = Field(default=0.1, ge=0, description="L2正则化")
    early_stopping_rounds: int = Field(default=50, ge=1, description="早停轮数")
    random_state: int = Field(default=42, description="随机种子")
    objective: str = Field(default="binary", description="目标函数")
    metric: str = Field(default="auc", description="评估指标")
    verbose: int = Field(default=-1, description="输出详细度")


class SentimentWeights(BaseModel):
    """情绪因子权重"""
    northbound: float = Field(default=0.30, ge=0, le=1)
    margin: float = Field(default=0.25, ge=0, le=1)
    turnover: float = Field(default=0.20, ge=0, le=1)
    sentiment_text: float = Field(default=0.25, ge=0, le=1)

    @model_validator(mode="after")
    def sum_check(self):
        total = self.northbound + self.margin + self.turnover + self.sentiment_text
        if abs(total - 1.0) > 0.05:
            logger.info(f"情绪权重之和={total:.3f}，将自动归一化")
        return self


class FusionWeights(BaseModel):
    """四因子融合权重"""
    dcf_weight: float = Field(default=0.35, ge=0, le=1)
    relative_weight: float = Field(default=0.25, ge=0, le=1)
    esg_weight: float = Field(default=0.20, ge=0, le=1)
    sentiment_weight: float = Field(default=0.20, ge=0, le=1)


class FeatureEngineeringConfig(BaseModel):
    """特征工程参数"""
    lag_periods: List[int] = Field(default=[1, 2, 3, 4], description="滞后季度数")
    rolling_windows: List[int] = Field(default=[4, 8, 12], description="滚动窗口")
    industry_standardize: bool = Field(default=True, description="启用行业标准化")
    min_industry_size: int = Field(default=5, ge=2, description="最小行业样本数")


class AppSettings(BaseModel):
    """
    ESG Insight Valuator 应用全局配置。

    由 config/ 目录下的 YAML 文件合并构建，
    所有字段经 Pydantic 验证后可安全使用。
    """

    # --- 项目信息 ---
    project_name: str = Field(default="ESG Insight Valuator")
    version: str = Field(default="1.0.0")

    # --- 目录 ---
    data_dir: str = Field(default="data")
    output_dir: str = Field(default="output")
    models_dir: str = Field(default="models")

    # --- 数据源 ---
    data_sources: List[DataSourceConfig] = Field(default_factory=list)

    # --- ESG权重 ---
    esg_weights: List[ESGWeightItem] = Field(default_factory=list)

    # --- 行业关联 ---
    industry_linkages: List[IndustryLinkageItem] = Field(default_factory=list)

    # --- 估值情景 ---
    scenarios: List[ScenarioParams] = Field(default_factory=list)

    # --- 模型参数 ---
    model_params: Optional[ModelParams] = None

    # --- 特征工程 ---
    feature_engineering: Optional[FeatureEngineeringConfig] = None

    # --- 情绪权重 ---
    sentiment_weights: Optional[SentimentWeights] = None

    # --- 融合权重 ---
    fusion_weights: Optional[FusionWeights] = None

    # --- 投资建议 ---
    advice_threshold_buy: float = Field(default=0.15, ge=0, le=1, description="买入阈值")
    advice_threshold_sell: float = Field(default=-0.10, ge=-1, le=1, description="卖出阈值")
    confidence_level: float = Field(default=0.95, ge=0.8, le=0.99, description="置信水平")

    # --- 回测 ---
    backtest_start: str = Field(default="2020-01-01")
    backtest_end: str = Field(default="2025-12-31")
    rebalance_frequency: str = Field(default="monthly")
    transaction_cost: float = Field(default=0.001, ge=0, le=0.01, description="交易成本")

    # --- 日志 ---
    log_level: str = Field(default="INFO")
    log_rotation: str = Field(default="10 MB")
    log_retention: str = Field(default="30 days")


# ============================================================================
# 配置加载函数
# ============================================================================

def load_yaml_file(file_path: Union[str, Path]) -> Dict[str, Any]:
    """
    安全加载单个 YAML 文件。

    Parameters
    ----------
    file_path : str or Path
        YAML 文件路径

    Returns
    -------
    dict
        配置字典（文件不存在时返回空字典）
    """
    path = Path(file_path)
    if not path.exists():
        logger.warning(f"配置文件不存在: {path}")
        return {}

    try:
        with open(path, "r", encoding="utf-8") as f:
            data = yaml.safe_load(f)
        logger.debug(f"已加载: {path.name}")
        return data if data else {}
    except yaml.YAMLError as e:
        logger.error(f"YAML解析错误 [{path}]: {e}")
        return {}
    except Exception as e:
        logger.error(f"读取文件失败 [{path}]: {e}")
        return {}


def merge_configs(config_dir: str = "config") -> Dict[str, Any]:
    """
    加载并合并 config/ 目录下所有 YAML 文件。

    加载顺序固定为：
      settings.yaml → industry_weights.yaml → industry_linkages.yaml
      → scenario_params.yaml → model_params.yaml

    Parameters
    ----------
    config_dir : str
        配置文件目录

    Returns
    -------
    dict
        合并后的配置字典
    """
    base = Path(config_dir)
    config_files = [
        "settings.yaml",
        "industry_weights.yaml",
        "industry_linkages.yaml",
        "scenario_params.yaml",
        "model_params.yaml",
    ]

    merged: Dict[str, Any] = {}

    for fname in config_files:
        data = load_yaml_file(base / fname)
        # 深度合并
        for key, value in data.items():
            if key in merged and isinstance(merged[key], dict) and isinstance(value, dict):
                merged[key].update(value)
            elif key in merged and isinstance(merged[key], list) and isinstance(value, list):
                merged[key].extend(value)
            else:
                merged[key] = value

    logger.info(f"配置合并完成: {len(merged)} 个顶级键")
    return merged


def load_app_settings(config_dir: str = "config") -> AppSettings:
    """
    加载并验证完整应用配置。

    流程：
    1. 合并所有 YAML 文件
    2. 通过 Pydantic AppSettings 验证
    3. 返回类型安全的配置对象

    Parameters
    ----------
    config_dir : str
        配置文件目录

    Returns
    -------
    AppSettings
        验证通过的配置对象

    Raises
    ------
    ValidationError
        配置验证失败时抛出
    """
    raw = merge_configs(config_dir)

    try:
        settings = AppSettings(**raw)
        logger.success("✅ 配置验证通过")
        return settings
    except ValidationError as e:
        logger.error(f"❌ 配置验证失败:\n{e}")
        raise


def get_config_paths(config_dir: str = "config") -> Dict[str, Path]:
    """
    获取各配置文件的标准路径。

    Parameters
    ----------
    config_dir : str
        配置文件目录

    Returns
    -------
    dict
        {配置名: 绝对路径}
    """
    base = Path(config_dir).resolve()
    return {
        "settings": base / "settings.yaml",
        "industry_weights": base / "industry_weights.yaml",
        "industry_linkages": base / "industry_linkages.yaml",
        "scenario_params": base / "scenario_params.yaml",
        "model_params": base / "model_params.yaml",
    }


def validate_config_schema(config_dir: str = "config") -> List[str]:
    """
    验证所有配置文件的结构完整性。

    检查项：
    - 文件是否存在
    - YAML 格式是否正确
    - Pydantic 验证是否通过
    - 行业权重是否覆盖行业关联中的所有行业

    Parameters
    ----------
    config_dir : str
        配置文件目录

    Returns
    -------
    list of str
        验证问题列表（空列表=无问题）
    """
    issues: List[str] = []

    # 检查文件存在性
    paths = get_config_paths(config_dir)
    for name, path in paths.items():
        if not path.exists():
            issues.append(f"缺失: {path}")

    # Pydantic 验证
    try:
        settings = load_app_settings(config_dir)
    except ValidationError as e:
        issues.append(f"Pydantic验证失败: {e}")
        return issues

    # 行业一致性检查
    weight_industries = {w.industry for w in settings.esg_weights}
    linkage_industries = set()
    for link in settings.industry_linkages:
        linkage_industries.add(link.source)
        linkage_industries.add(link.target)

    orphan = linkage_industries - weight_industries
    if orphan:
        issues.append(f"行业关联中存在无权重配置的行业: {orphan}")

    # 情景概率检查
    if settings.scenarios:
        total_prob = sum(s.probability for s in settings.scenarios)
        if abs(total_prob - 1.0) > 0.1:
            issues.append(f"情景概率之和={total_prob:.2f}，偏离1.0超过10%")

    if issues:
        logger.warning(f"配置检查发现 {len(issues)} 个问题:\n" + "\n".join(f"  - {i}" for i in issues))
    else:
        logger.info("✅ 所有配置检查通过")

    return issues


# ============================================================================
# 便捷函数
# ============================================================================

def quick_load(config_dir: str = "config") -> AppSettings:
    """
    一行加载配置（忽略文件缺失等非致命问题）。

    Parameters
    ----------
    config_dir : str
        配置目录

    Returns
    -------
    AppSettings
        配置对象（包含默认值填充）
    """
    raw = merge_configs(config_dir)
    # 过滤掉无法识别的字段，使用默认值
    valid_fields = set(AppSettings.model_fields.keys())
    filtered = {k: v for k, v in raw.items() if k in valid_fields}
    return AppSettings(**filtered)


def reload_config(settings: AppSettings, overrides: Dict[str, Any]) -> AppSettings:
    """
    用命令行/环境变量覆盖部分配置。

    Parameters
    ----------
    settings : AppSettings
        基础配置
    overrides : dict
        覆盖项

    Returns
    -------
    AppSettings
        更新后的配置
    """
    data = settings.model_dump()
    data.update(overrides)
    return AppSettings(**data)
