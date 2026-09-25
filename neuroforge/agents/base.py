"""智能体基类：提供通用存档/读档（pickle 权重）。"""

from __future__ import annotations

import os
import pickle

from ..core.errors import AgentError


class AgentBase:
    """所有智能体的基类，定义 name 与 save/load 契约。"""

    name = "base"

    def save(self, path: str) -> None:
        """将智能体状态持久化到文件。"""
        os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
        try:
            with open(path, "wb") as fh:
                pickle.dump(self._state_dict(), fh)
        except Exception as exc:  # pragma: no cover - IO 异常
            raise AgentError(f"保存模型失败: {exc}", code="E200")

    def load(self, path: str) -> None:
        """从文件恢复智能体状态。"""
        try:
            with open(path, "rb") as fh:
                self._load_state_dict(pickle.load(fh))
        except Exception as exc:  # pragma: no cover - IO 异常
            raise AgentError(f"读取模型失败: {exc}", code="E200")

    def _state_dict(self) -> dict:
        raise NotImplementedError

    def _load_state_dict(self, sd: dict) -> None:
        raise NotImplementedError
