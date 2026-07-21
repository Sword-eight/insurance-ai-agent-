"""
Insurance AI Agent - 日志模块
统一日志管理，支持控制台输出和文件写入。
"""

import logging
import os
from datetime import datetime
from pathlib import Path
from logging.handlers import RotatingFileHandler

from config import LOG_CONFIG


class LoggerManager:
    """
    日志管理器
    提供统一的日志记录接口，支持文件轮转。
    """

    _instance: "LoggerManager | None" = None

    def __new__(cls) -> "LoggerManager":
        """单例模式，确保全局只有一个日志管理器。"""
        if cls._instance is None:
            cls._instance = super().__new__(cls)
            cls._instance._initialized = False
        return cls._instance

    def __init__(self) -> None:
        """初始化日志管理器。"""
        if self._initialized:
            return
        self._initialized = True

        # 确保日志目录存在
        log_dir: Path = Path(LOG_CONFIG["log_dir"])
        log_dir.mkdir(parents=True, exist_ok=True)

        # 根 logger
        self._root_logger = logging.getLogger("insurance_ai_agent")
        self._root_logger.setLevel(getattr(logging, LOG_CONFIG["log_level"]))

        # 避免重复添加 handler
        if not self._root_logger.handlers:
            # 控制台 handler
            console_handler = logging.StreamHandler()
            console_handler.setLevel(logging.INFO)
            console_fmt = logging.Formatter(
                LOG_CONFIG["log_format"],
                datefmt=LOG_CONFIG["log_date_format"],
            )
            console_handler.setFormatter(console_fmt)
            self._root_logger.addHandler(console_handler)

            # 文件 handler（按日期命名，自动轮转）
            date_str: str = datetime.now().strftime("%Y%m%d")
            log_file: str = os.path.join(log_dir, f"agent_{date_str}.log")
            file_handler = RotatingFileHandler(
                log_file,
                maxBytes=10 * 1024 * 1024,  # 10MB
                backupCount=7,
                encoding="utf-8",
            )
            file_handler.setLevel(logging.DEBUG)
            file_fmt = logging.Formatter(
                LOG_CONFIG["log_format"],
                datefmt=LOG_CONFIG["log_date_format"],
            )
            file_handler.setFormatter(file_fmt)
            self._root_logger.addHandler(file_handler)

    def get_logger(self, name: str) -> logging.Logger:
        """
        获取指定模块的 logger 实例。

        Args:
            name: 模块名称（如 "graph.agent"、"tools.rag"）

        Returns:
            配置完成的 logger 实例
        """
        return self._root_logger.getChild(name)


# 全局 LoggerManager 实例
_logger_manager = LoggerManager()


def get_logger(name: str) -> logging.Logger:
    """
    便捷方法：获取指定模块的 logger。

    Args:
        name: 模块名称

    Returns:
        logging.Logger 实例
    """
    return _logger_manager.get_logger(name)
