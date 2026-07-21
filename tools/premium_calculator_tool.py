"""
Insurance AI Agent - 保费估算工具
根据用户年龄、性别、保额、保险期限、职业类别估算保费。
"""

from typing import Any, Dict, Type

from langchain_core.tools import BaseTool
from pydantic import BaseModel, Field

from config import PREMIUM_RATE_PATH
from utils.logger import get_logger
from utils.helpers import load_json, Timer

logger = get_logger("tools.premium")


class PremiumCalculatorInput(BaseModel):
    """premium_calculator 工具的参数 Schema。"""

    age: int = Field(
        description="被保险人年龄（周岁），范围 16-60",
        ge=16,
        le=60,
    )
    gender: str = Field(
        description="被保险人性别：男 或 女",
    )
    coverage_amount: float = Field(
        description="保额（万元），如 50 表示 50 万元",
        gt=0,
    )
    insurance_term: str = Field(
        description="保险期限：10年、20年、30年 或 终身",
    )
    occupation_class: str = Field(
        description="职业类别：1类（办公室职员）、2类（轻度体力劳动）、"
        "3类（中度体力劳动）、4类（高风险职业）",
    )


class PremiumCalculatorTool(BaseTool):
    """
    保费估算工具。
    基于年龄、性别、保额、保险期限和职业类别估算年度保费。
    费率配置从 config/premium_rate.json 读取，不硬编码。
    """

    name: str = "premium_calculator"
    description: str = (
        "估算保险保费。需要提供：年龄（16-60岁）、性别（男/女）、"
        "保额（万元）、保险期限（10年/20年/30年/终身）、"
        "职业类别（1类办公室职员/2类轻度体力/3类中度体力/4类高风险）。"
        "返回：预估年度保费金额及计算明细。"
    )
    args_schema: Type[BaseModel] = PremiumCalculatorInput

    def __init__(self, **kwargs: Any) -> None:
        """初始化工具，加载费率配置。"""
        super().__init__(**kwargs)
        self._rate_config: Dict[str, Any] = load_json(PREMIUM_RATE_PATH)

    def _run(
        self,
        age: int,
        gender: str,
        coverage_amount: float,
        insurance_term: str,
        occupation_class: str,
    ) -> str:
        """
        执行保费估算（同步）。

        Args:
            age: 被保险人年龄
            gender: 被保险人性别
            coverage_amount: 保额（万元）
            insurance_term: 保险期限
            occupation_class: 职业类别

        Returns:
            格式化的保费估算结果
        """
        logger.info(
            f"[Tool] premium_calculator 被调用: age={age}, gender={gender}, "
            f"coverage={coverage_amount}万, term={insurance_term}, occ={occupation_class}"
        )

        try:
            with Timer("premium_calculator 计算耗时") as timer:
                result: Dict[str, Any] = self._calculate(
                    age, gender, coverage_amount, insurance_term, occupation_class
                )

                # 格式化输出
                formatted: str = self._format_result(result)

                logger.info(
                    f"[Tool] premium_calculator 完成: "
                    f"年度保费={result['annual_premium']:.2f}元, "
                    f"耗时 {timer.elapsed:.4f}s"
                )

            return formatted

        except Exception as e:
            error_msg: str = f"保费估算失败: {e}"
            logger.error(
                f"[Tool] premium_calculator 异常: {e}", exc_info=True
            )
            return error_msg

    def _calculate(
        self,
        age: int,
        gender: str,
        coverage_amount: float,
        insurance_term: str,
        occupation_class: str,
    ) -> Dict[str, Any]:
        """
        核心保费计算逻辑。

        Args:
            age: 年龄
            gender: 性别
            coverage_amount: 保额（万元）
            insurance_term: 保险期限
            occupation_class: 职业类别

        Returns:
            包含各项系数和最终保费的字典
        """
        rate_table: Dict[str, Dict] = self._rate_config.get("rate_table", {})
        gender_factor: Dict[str, float] = self._rate_config.get("gender_factor", {})
        age_factor_config: Dict[str, Any] = self._rate_config.get("age_factor", {})
        term_factor: Dict[str, float] = self._rate_config.get("term_factor", {})

        # 1. 获取职业基础费率（每万元保额年费率）
        if occupation_class not in rate_table:
            raise ValueError(
                f"未知职业类别: {occupation_class}，可选: {list(rate_table.keys())}"
            )

        occupation_info: Dict[str, Any] = rate_table[occupation_class]
        base_rate: float = occupation_info["annual_rate_per_10k"]

        # 2. 性别系数
        if gender not in gender_factor:
            raise ValueError(
                f"性别参数无效: {gender}，可选: {list(gender_factor.keys())}"
            )
        g_factor: float = gender_factor[gender]

        # 3. 年龄系数（每偏离基准年龄1岁增加 per_year_increase）
        base_age: int = age_factor_config.get("base_age", 30)
        per_year: float = age_factor_config.get("per_year_increase", 0.02)
        age_diff: int = max(0, age - base_age)
        a_factor: float = 1.0 + age_diff * per_year

        # 4. 保险期限折扣系数
        if insurance_term not in term_factor:
            raise ValueError(
                f"保险期限参数无效: {insurance_term}，可选: {list(term_factor.keys())}"
            )
        t_factor: float = term_factor[insurance_term]

        # 5. 计算年度保费
        # 公式：保额(万) × 基础费率(元/万/年) × 性别系数 × 年龄系数 × 期限系数
        annual_premium: float = (
            coverage_amount * base_rate * g_factor * a_factor * t_factor
        )

        # 月度保费
        monthly_premium: float = annual_premium / 12.0

        return {
            "occupation_class": occupation_class,
            "occupation_name": occupation_info.get("name", ""),
            "occupation_examples": occupation_info.get("examples", ""),
            "age": age,
            "gender": gender,
            "coverage_amount": coverage_amount,
            "insurance_term": insurance_term,
            "base_rate_per_10k": base_rate,
            "gender_factor": g_factor,
            "age_factor": round(a_factor, 4),
            "term_factor": t_factor,
            "annual_premium": round(annual_premium, 2),
            "monthly_premium": round(monthly_premium, 2),
        }

    def _format_result(self, result: Dict[str, Any]) -> str:
        """
        将计算结果格式化为用户可读文本。

        Args:
            result: _calculate() 返回的计算结果

        Returns:
            格式化后的保费说明文本
        """
        return (
            f"【保费估算结果】\n"
            f"  被保险人信息: {result['age']}岁 {result['gender']}性, "
            f"职业: {result['occupation_class']}({result['occupation_name']})\n"
            f"  保额: {result['coverage_amount']}万元\n"
            f"  保险期限: {result['insurance_term']}\n"
            f"\n"
            f"【费率明细】\n"
            f"  基础费率: {result['base_rate_per_10k']}元/万元/年\n"
            f"  性别系数: {result['gender_factor']}\n"
            f"  年龄系数: {result['age_factor']}\n"
            f"  期限系数: {result['term_factor']}\n"
            f"\n"
            f"【预估保费】\n"
            f"  年度保费: {result['annual_premium']}元/年\n"
            f"  月度保费: {result['monthly_premium']}元/月\n"
            f"\n"
            f"（注：此为简化估算，实际保费以保险公司核保结果为准）"
        )
