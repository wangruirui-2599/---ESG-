"""
多源数据加载模块
================
支持从 CSV 文件和数据库加载财务、ESG、市场等多源数据，
统一输出标准化的 DataFrame schema，供下游模块使用。
"""

from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import pandas as pd
from loguru import logger
from pydantic import BaseModel, Field, field_validator


# ============================================================================
# 统一 Schema 定义
# ============================================================================

class FinancialSchema(BaseModel):
    """财务数据标准字段"""
    stock_code: str = Field(..., description="股票代码")
    report_date: str = Field(..., description="报告期")
    revenue: float = Field(..., description="营业收入（亿元）")
    net_profit: float = Field(..., description="归母净利润（亿元）")
    total_assets: float = Field(..., description="总资产（亿元）")
    total_equity: float = Field(..., description="股东权益（亿元）")
    operating_cashflow: float = Field(..., description="经营性现金流（亿元）")
    roe: float = Field(..., description="净资产收益率")
    debt_ratio: float = Field(..., description="资产负债率")
    revenue_growth: float = Field(default=0.0, description="营收同比增速")
    gross_margin: float = Field(default=0.0, description="毛利率")


class ESGSchema(BaseModel):
    """ESG评分数据标准字段"""
    stock_code: str = Field(..., description="股票代码")
    rating_date: str = Field(..., description="评级日期")
    E_score: float = Field(..., ge=0, le=100, description="环境评分 (0-100)")
    S_score: float = Field(..., ge=0, le=100, description="社会评分 (0-100)")
    G_score: float = Field(..., ge=0, le=100, description="治理评分 (0-100)")
    ESG_total: float = Field(..., ge=0, le=100, description="ESG综合评分 (0-100)")
    rating_agency: str = Field(default="", description="评级机构")
    industry: str = Field(default="", description="所属行业")


class MarketDataSchema(BaseModel):
    """市场数据标准字段"""
    stock_code: str = Field(..., description="股票代码")
    trade_date: str = Field(..., description="交易日期")
    close_price: float = Field(..., description="收盘价")
    market_cap: float = Field(..., description="总市值（亿元）")
    pe_ttm: float = Field(default=0.0, description="市盈率TTM")
    pb: float = Field(default=0.0, description="市净率")
    turnover_rate: float = Field(default=0.0, description="换手率")
    volume: float = Field(default=0.0, description="成交量（万股）")


# ============================================================================
# 数据加载器
# ============================================================================

class DataLoader:
    """
    多源数据加载器。

    支持从 CSV 文件或数据库连接加载数据，自动进行基础的
    字段映射和类型转换，统一输出标准化 DataFrame。

    Attributes
    ----------
    data_dir : Path
        数据根目录
    encoding : str
        默认文件编码
    """

    # 字段名映射：将原始列名映射到标准 schema 字段
    COLUMN_MAPPING: Dict[str, Dict[str, str]] = {
        "financials": {
            "stock_code": "stock_code",
            "报告期": "report_date",
            "营业总收入": "revenue",
            "净利润": "net_profit",
            "资产总计": "total_assets",
            "股东权益": "total_equity",
            "经营活动现金流": "operating_cashflow",
            "ROE": "roe",
            "资产负债率": "debt_ratio",
            "营收增速": "revenue_growth",
            "毛利率": "gross_margin",
        },
        "esg_ratings": {
            "stock_code": "stock_code",
            "评级日期": "rating_date",
            "E评分": "E_score",
            "S评分": "S_score",
            "G评分": "G_score",
            "ESG总分": "ESG_total",
            "评级机构": "rating_agency",
            "行业": "industry",
        },
        "market_data": {
            "stock_code": "stock_code",
            "交易日期": "trade_date",
            "收盘价": "close_price",
            "总市值": "market_cap",
            "市盈率": "pe_ttm",
            "市净率": "pb",
            "换手率": "turnover_rate",
            "成交量": "volume",
        },
    }

    def __init__(self, data_dir: str = "data", encoding: str = "utf-8") -> None:
        """
        初始化数据加载器。

        Parameters
        ----------
        data_dir : str
            数据根目录路径
        encoding : str
            文件默认编码
        """
        self.data_dir = Path(data_dir)
        self.encoding = encoding
        logger.info(f"DataLoader 初始化完成，数据目录: {self.data_dir}")

    def load_csv(
        self,
        file_path: str,
        schema_name: Optional[str] = None,
        **kwargs: Any,
    ) -> pd.DataFrame:
        """
        加载单个 CSV 文件并进行字段标准化。

        Parameters
        ----------
        file_path : str
            CSV 文件路径（相对于 data_dir 或绝对路径）
        schema_name : str, optional
            schema 名称，用于字段重命名映射（financials/esg_ratings/market_data）
        **kwargs : dict
            传递给 pd.read_csv 的额外参数

        Returns
        -------
        pd.DataFrame
            标准化后的数据表

        Raises
        ------
        FileNotFoundError
            文件不存在时抛出
        """
        path = Path(file_path)
        if not path.is_absolute():
            path = self.data_dir / file_path

        if not path.exists():
            raise FileNotFoundError(f"数据文件不存在: {path}")

        try:
            df = pd.read_csv(path, encoding=self.encoding, **kwargs)
            logger.info(f"成功加载 CSV: {path}, 行数={len(df)}, 列数={len(df.columns)}")
        except UnicodeDecodeError:
            # 尝试其他常见编码
            for enc in ["gbk", "gb2312", "gb18030", "latin-1"]:
                try:
                    df = pd.read_csv(path, encoding=enc, **kwargs)
                    logger.info(f"以编码 {enc} 加载 CSV: {path}")
                    break
                except UnicodeDecodeError:
                    continue
            else:
                raise ValueError(f"无法以任何已知编码读取文件: {path}")

        # 字段名标准化
        if schema_name and schema_name in self.COLUMN_MAPPING:
            df = self._standardize_columns(df, schema_name)

        # 基础数据清洗
        df = self._basic_cleaning(df)

        return df

    def _standardize_columns(self, df: pd.DataFrame, schema_name: str) -> pd.DataFrame:
        """
        将 DataFrame 列名映射到标准 schema 字段。

        Parameters
        ----------
        df : pd.DataFrame
            原始数据
        schema_name : str
            schema 名称

        Returns
        -------
        pd.DataFrame
            列名标准化后的数据
        """
        mapping = self.COLUMN_MAPPING.get(schema_name, {})
        # 只重命名存在的列
        rename_dict = {k: v for k, v in mapping.items() if k in df.columns}
        df = df.rename(columns=rename_dict)
        logger.debug(f"字段标准化完成: {schema_name}, 映射 {len(rename_dict)} 个字段")
        return df

    def _basic_cleaning(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        基础数据清洗：去重、去空值行。

        Parameters
        ----------
        df : pd.DataFrame
            待清洗数据

        Returns
        -------
        pd.DataFrame
            清洗后的数据
        """
        n_before = len(df)
        df = df.drop_duplicates()
        # 去除全空行
        df = df.dropna(how="all")
        n_after = len(df)
        if n_before > n_after:
            logger.debug(f"数据清洗: {n_before} -> {n_after} 行 (去除 {n_before - n_after})")
        return df

    def load_all(
        self, configs: List[Dict[str, str]], schema_name: Optional[str] = None
    ) -> pd.DataFrame:
        """
        批量加载多个数据文件并合并。

        Parameters
        ----------
        configs : list of dict
            数据源配置列表，每项包含 {"name": ..., "path": ..., "type": ...}
        schema_name : str, optional
            schema 名称

        Returns
        -------
        pd.DataFrame
            合并后的数据表
        """
        frames: List[pd.DataFrame] = []
        for cfg in configs:
            source_type = cfg.get("type", "csv")
            source_path = cfg.get("path", "")
            name = cfg.get("name", "unknown")

            if source_type == "csv" and source_path:
                try:
                    df = self.load_csv(source_path, schema_name=schema_name)
                    df["_source"] = name
                    frames.append(df)
                except Exception as e:
                    logger.error(f"加载数据源失败 [{name}]: {e}")

        if not frames:
            logger.warning("没有成功加载任何数据")
            return pd.DataFrame()

        result = pd.concat(frames, ignore_index=True, sort=False)
        logger.info(f"批量加载完成: {len(frames)} 个数据源, 总计 {len(result)} 行")
        return result

    def validate_schema(
        self, df: pd.DataFrame, required_cols: List[str]
    ) -> Tuple[bool, List[str]]:
        """
        校验 DataFrame 是否包含必需的列。

        Parameters
        ----------
        df : pd.DataFrame
            待校验数据
        required_cols : list of str
            必需的列名列表

        Returns
        -------
        tuple
            (是否通过, 缺失列列表)
        """
        missing = [c for c in required_cols if c not in df.columns]
        if missing:
            logger.warning(f"Schema 校验失败，缺失字段: {missing}")
        else:
            logger.info("Schema 校验通过")
        return len(missing) == 0, missing

    def get_data_summary(self, df: pd.DataFrame) -> Dict[str, Any]:
        """
        生成数据摘要统计。

        Parameters
        ----------
        df : pd.DataFrame
            数据表

        Returns
        -------
        dict
            包含行数、列数、缺失率、数据类型等信息的摘要字典
        """
        n_rows, n_cols = df.shape
        missing_rate = df.isnull().mean().to_dict()
        dtypes = {c: str(t) for c, t in df.dtypes.items()}

        # 数值列描述性统计
        numeric_cols = df.select_dtypes(include=["number"]).columns.tolist()
        desc = df[numeric_cols].describe().to_dict() if numeric_cols else {}

        summary = {
            "rows": n_rows,
            "columns": n_cols,
            "column_names": df.columns.tolist(),
            "missing_rate": missing_rate,
            "dtypes": dtypes,
            "numeric_summary": desc,
        }
        return summary


# ============================================================================
# 便捷函数
# ============================================================================

def load_financials(data_dir: str = "data") -> pd.DataFrame:
    """
    便捷函数：加载财务数据。

    Parameters
    ----------
    data_dir : str
        数据目录

    Returns
    -------
    pd.DataFrame
        标准化财务数据
    """
    loader = DataLoader(data_dir)
    return loader.load_csv("raw/financials.csv", schema_name="financials")


def load_esg_ratings(data_dir: str = "data") -> pd.DataFrame:
    """
    便捷函数：加载ESG评级数据。

    Parameters
    ----------
    data_dir : str
        数据目录

    Returns
    -------
    pd.DataFrame
        标准化ESG数据
    """
    loader = DataLoader(data_dir)
    return loader.load_csv("raw/esg_ratings.csv", schema_name="esg_ratings")


def load_market_data(data_dir: str = "data") -> pd.DataFrame:
    """
    便捷函数：加载市场行情数据。

    Parameters
    ----------
    data_dir : str
        数据目录

    Returns
    -------
    pd.DataFrame
        标准化市场数据
    """
    loader = DataLoader(data_dir)
    return loader.load_csv("raw/market_data.csv", schema_name="market_data")
