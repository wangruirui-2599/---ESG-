"""
日志配置模块
============
基于 Loguru 的统一日志管理，支持控制台输出和文件轮转。
"""

import sys
from pathlib import Path
from loguru import logger


def setup_logger(
    log_level: str = "INFO",
    log_dir: str = "logs",
    rotation: str = "10 MB",
    retention: str = "30 days",
) -> None:
    """
    配置全局日志记录器。

    Parameters
    ----------
    log_level : str
        日志级别，可选 DEBUG/INFO/WARNING/ERROR/CRITICAL
    log_dir : str
        日志文件存储目录
    rotation : str
        日志文件轮转大小/时间
    retention : str
        日志文件保留时间
    """
    # 移除默认处理器
    logger.remove()

    # 控制台输出：彩色格式
    logger.add(
        sys.stdout,
        level=log_level,
        format=(
            "<green>{time:YYYY-MM-DD HH:mm:ss.SSS}</green> | "
            "<level>{level: <8}</level> | "
            "<cyan>{name}</cyan>:<cyan>{function}</cyan>:<cyan>{line}</cyan> | "
            "<level>{message}</level>"
        ),
        colorize=True,
    )

    # 确保日志目录存在
    log_path = Path(log_dir)
    log_path.mkdir(parents=True, exist_ok=True)

    # 文件输出：带轮转
    logger.add(
        log_path / "eiv_{time:YYYY-MM-DD}.log",
        level=log_level,
        format=(
            "{time:YYYY-MM-DD HH:mm:ss.SSS} | {level: <8} | "
            "{name}:{function}:{line} | {message}"
        ),
        rotation=rotation,
        retention=retention,
        encoding="utf-8",
    )

    # 错误日志单独输出
    logger.add(
        log_path / "error_{time:YYYY-MM-DD}.log",
        level="ERROR",
        format=(
            "{time:YYYY-MM-DD HH:mm:ss.SSS} | {level: <8} | "
            "{name}:{function}:{line} | {message}"
        ),
        rotation=rotation,
        retention=retention,
        encoding="utf-8",
    )

    logger.info(f"日志系统初始化完成，级别={log_level}，目录={log_dir}")


def get_logger(name: str = __name__):
    """
    获取命名日志记录器。

    Parameters
    ----------
    name : str
        日志记录器名称（通常使用 __name__）

    Returns
    -------
    logger
        Loguru 日志记录器实例
    """
    return logger.bind(name=name)
