"""
Insurance AI Agent - 保费估算工具
根据用户年龄、性别、保额、保险期限、职业类别估算保费。

设计原则：
  - Tool 只负责：接收 LLM 参数 → 委托 Service 计算 → 格式化输出
  - 业务逻辑全部在 PremiumService 中，Tool 不做计算
  - 如果将来改成 REST API，PremiumService 可直接复用
"""

from typing import Any, Dict, Type

from langchain_core.tools import BaseTool
from pydantic import BaseModel, Field

from services.premium_service import PremiumService
from utils.logger import get_logger
from utils.helpers import Timer

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
    委托 PremiumService 执行业务计算，Tool 只负责参数接收和结果格式化。
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
        """初始化工具，创建 PremiumService 实例。"""
        super().__init__(**kwargs)
        self._service = PremiumService()

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
                # 委托 PremiumService 执行计算（Tool 不做业务逻辑）
                result: Dict[str, Any] = self._service.calculate(
                    age=age,
                    gender=gender,
                    coverage_amount=coverage_amount,
                    insurance_term=insurance_term,
                    occupation_class=occupation_class,
                )

                # 格式化为 LLM 可读文本
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

    def _format_result(self, result: Dict[str, Any]) -> str:
        """
        将 PremiumService.calculate() 的计算结果格式化为 LLM 可读文本。

        Args:
            result: PremiumService.calculate() 返回的计算结果

        Returns:
            格式化后的保费说明文本
        """
        breakdown: Dict[str, Any] = result.get("breakdown", {})
        return (
            f"【保费估算结果】\n"
            f"  被保险人信息: {result['age']}岁 {result['gender']}性, "
            f"职业: {result['occupation_class']}({result['occupation_name']})\n"
            f"  保额: {result['coverage_amount']}万元\n"
            f"  保险期限: {result['insurance_term']}\n"
            f"\n"
            f"【费率明细】\n"
            f"  基础费率: {breakdown.get('base_rate_per_10k', 'N/A')}元/万元/年\n"
            f"  性别系数: {breakdown.get('gender_factor', 'N/A')}\n"
            f"  年龄系数: {breakdown.get('age_factor', 'N/A')}\n"
            f"  期限系数: {breakdown.get('term_factor', 'N/A')}\n"
            f"\n"
            f"【预估保费】\n"
            f"  年度保费: {result['annual_premium']}元/年\n"
            f"  月度保费: {result['monthly_premium']}元/月\n"
            f"\n"
            f"（注：此为简化估算，实际保费以保险公司核保结果为准）"
        )
