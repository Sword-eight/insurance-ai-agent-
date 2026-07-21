"""
Insurance AI Agent - 工具函数模块
提供项目通用的辅助函数。
"""

import time
import json
from pathlib import Path
from typing import Any, Dict, List, Optional
from datetime import datetime


def load_json(file_path: str) -> dict:
    """
    安全加载 JSON 文件。

    Args:
        file_path: JSON 文件路径

    Returns:
        解析后的字典，文件不存在则返回空字典
    """
    try:
        with open(file_path, "r", encoding="utf-8") as f:
            return json.load(f)
    except (FileNotFoundError, json.JSONDecodeError) as e:
        from utils.logger import get_logger
        get_logger("utils.helpers").warning(f"无法加载 JSON 文件 {file_path}: {e}")
        return {}


def save_json(file_path: str, data: dict) -> None:
    """
    安全保存 JSON 文件。

    Args:
        file_path: 目标文件路径
        data: 待保存的字典
    """
    Path(file_path).parent.mkdir(parents=True, exist_ok=True)
    with open(file_path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)


class Timer:
    """
    上下文管理器：用于测量代码块执行耗时。
    用法:
        with Timer("检索耗时") as t:
            ... 执行操作 ...
        print(t.elapsed)  # 秒
    """

    def __init__(self, label: str = "") -> None:
        """
        Args:
            label: 耗时标签，用于日志输出
        """
        self.label: str = label
        self.start: float = 0.0
        self.end: float = 0.0
        self.elapsed: float = 0.0

    def __enter__(self) -> "Timer":
        self.start = time.perf_counter()
        return self

    def __exit__(self, *args: Any) -> None:
        self.end = time.perf_counter()
        self.elapsed = self.end - self.start
        if self.label:
            from utils.logger import get_logger
            get_logger("utils.timer").info(
                f"{self.label}: {self.elapsed:.4f}s"
            )


def format_timestamp(ts: Optional[float] = None) -> str:
    """
    格式化时间戳为可读字符串。

    Args:
        ts: Unix 时间戳，默认当前时间

    Returns:
        格式化时间字符串，如 "2026-07-18 14:30:00"
    """
    if ts is None:
        ts = time.time()
    return datetime.fromtimestamp(ts).strftime("%Y-%m-%d %H:%M:%S")
