"""
Insurance AI Agent - 保费计算服务模块
封装保费计算业务逻辑，供 Tool 和 Web UI 调用。
"""

from typing import Any, Dict

from config import PREMIUM_RATE_PATH
from utils.logger import get_logger
from utils.helpers import load_json

logger = get_logger("services.premium")


class PremiumService:
    """
    保费计算服务。
    负责加载费率配置、执行保费计算。
    """

    def __init__(self) -> None:
        """初始化服务，加载费率配置。"""
        self._rate_config: Dict[str, Any] = load_json(PREMIUM_RATE_PATH)
        self._validate_config()

    def _validate_config(self) -> None:
        """验证费率配置文件的完整性。"""
        required_keys: list = ["rate_table", "gender_factor", "age_factor", "term_factor"]
        for key in required_keys:
            if key not in self._rate_config:
                raise ValueError(f"费率配置文件缺少必要字段: {key}")
        logger.info("保费费率配置验证通过")

    def calculate(
        self,
        age: int,
        gender: str,
        coverage_amount: float,
        insurance_term: str,
        occupation_class: str,
    ) -> Dict[str, Any]:
        """
        计算保费。

        Args:
            age: 年龄（16-60）
            gender: 性别（男/女）
            coverage_amount: 保额（万元）
            insurance_term: 保险期限
            occupation_class: 职业类别

        Returns:
            包含计算结果和费率明细的字典
        """
        rate_table: Dict = self._rate_config["rate_table"]
        gender_factor: Dict = self._rate_config["gender_factor"]
        age_config: Dict = self._rate_config["age_factor"]
        term_factor: Dict = self._rate_config["term_factor"]

        # 参数校验
        if occupation_class not in rate_table:
            raise ValueError(f"未知职业类别: {occupation_class}")
        if gender not in gender_factor:
            raise ValueError(f"性别参数无效: {gender}")
        if insurance_term not in term_factor:
            raise ValueError(f"保险期限参数无效: {insurance_term}")
        if age < 16 or age > 60:
            raise ValueError(f"年龄超出范围(16-60): {age}")

        occ_info: Dict = rate_table[occupation_class]
        base_rate: float = occ_info["annual_rate_per_10k"]
        g_factor: float = gender_factor[gender]
        base_age: int = age_config.get("base_age", 30)
        per_year: float = age_config.get("per_year_increase", 0.02)
        age_diff: int = max(0, age - base_age)
        a_factor: float = 1.0 + age_diff * per_year
        t_factor: float = term_factor[insurance_term]

        annual_premium: float = (
            coverage_amount * base_rate * g_factor * a_factor * t_factor
        )
        monthly_premium: float = annual_premium / 12.0

        return {
            "success": True,
            "occupation_class": occupation_class,
            "occupation_name": occ_info.get("name", ""),
            "age": age,
            "gender": gender,
            "coverage_amount": coverage_amount,
            "insurance_term": insurance_term,
            "breakdown": {
                "base_rate_per_10k": base_rate,
                "gender_factor": g_factor,
                "age_factor": round(a_factor, 4),
                "term_factor": t_factor,
            },
            "annual_premium": round(annual_premium, 2),
            "monthly_premium": round(monthly_premium, 2),
        }

    def get_rate_info(self) -> Dict[str, Any]:
        """
        获取费率配置摘要。

        Returns:
            费率信息字典
        """
        rate_table: Dict = self._rate_config.get("rate_table", {})
        summary: Dict[str, Any] = {
            "occupation_classes": {},
            "available_terms": list(self._rate_config.get("term_factor", {}).keys()),
        }
        for cls_name, cls_info in rate_table.items():
            summary["occupation_classes"][cls_name] = {
                "name": cls_info.get("name", ""),
                "rate": cls_info.get("annual_rate_per_10k", 0),
            }
        return summary
