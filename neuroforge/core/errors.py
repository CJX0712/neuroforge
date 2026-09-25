"""NeuroForge 统一错误码。

错误码约定（与 RAGNext 同源设计）：
    E100  环境（Env）相关
    E200  智能体（Agent）相关
    E300  训练（Train）相关
    E400  评测（Eval）相关
    E500  超参搜索（HPO）相关
    E600  配置（Config）相关
"""

from __future__ import annotations


class NeuroForgeError(Exception):
    """所有 NeuroForge 异常的基类。"""

    code = "E000"

    def __init__(self, message: str, code: str | None = None) -> None:
        super().__init__(message)
        self.message = message
        if code is not None:
            self.code = code

    def __str__(self) -> str:  # pragma: no cover - 调试用
        return f"[{self.code}] {self.message}"


class ConfigError(NeuroForgeError):
    code = "E600"


class EnvError(NeuroForgeError):
    code = "E100"


class AgentError(NeuroForgeError):
    code = "E200"


class TrainError(NeuroForgeError):
    code = "E300"


class EvalError(NeuroForgeError):
    code = "E400"


class HPOError(NeuroForgeError):
    code = "E500"
