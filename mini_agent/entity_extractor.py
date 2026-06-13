# -*- coding: utf-8 -*-
"""问题实体分解 —— 检索前将问题拆为独立代码实体，确保每个概念都有候选代表。"""

from __future__ import annotations

import json
from dataclasses import dataclass, field

_EXTRACTION_PROMPT = """\
你是一个代码实体分解器。给定一个关于代码库的问题，识别出需要定位的独立代码实体（函数、类、方法、模块或概念）。

对每个实体输出：
- name: 简短标签（实体名）
- query: 优化的搜索关键词（不是自然语言，使用实体的名称和关键标识符）
- description: 这个实体是什么/做什么（一句话）
- type_hint: FUNCTION, CLASS, METHOD, MODULE, 或 CONCEPT

输出纯 JSON 数组，不要其他文字。

问题: __QUESTION__"""


@dataclass
class Entity:
    """问题中需要定位的代码实体。"""

    name: str
    query: str = ""
    description: str = ""
    type_hint: str = "CONCEPT"

    def __post_init__(self):
        if not self.query:
            self.query = self.name

    def to_dict(self) -> dict:
        return {
            "name": self.name,
            "query": self.query,
            "description": self.description,
            "type_hint": self.type_hint,
        }

    @classmethod
    def from_dict(cls, d: dict) -> Entity:
        return cls(
            name=str(d.get("name", "")),
            query=str(d.get("query", "")),
            description=str(d.get("description", "")),
            type_hint=str(d.get("type_hint", "CONCEPT")),
        )


class EntityExtractor:
    """用轻量 LLM 调用将问题分解为独立实体，检索前使用。"""

    def __init__(self, model=None):
        self._model = model

    def extract(self, question: str) -> list[Entity]:
        if self._model is None:
            return [Entity(name="primary", query=question)]

        prompt = _EXTRACTION_PROMPT.replace("__QUESTION__", question)
        try:
            raw = self._model.generate([{"role": "user", "content": prompt}])
        except Exception:
            return [Entity(name="primary", query=question)]

        entities = self._parse(raw)
        if not entities:
            return [Entity(name="primary", query=question)]
        return entities

    @staticmethod
    def _parse(raw: str) -> list[Entity]:
        try:
            data = json.loads(raw)
        except json.JSONDecodeError:
            # 尝试提取 JSON 数组
            import re
            match = re.search(r"\[.*\]", raw, re.DOTALL)
            if match:
                try:
                    data = json.loads(match.group())
                except json.JSONDecodeError:
                    return []
            else:
                return []

        if not isinstance(data, list) or len(data) == 0:
            return []

        entities = []
        for item in data:
            if not isinstance(item, dict):
                continue
            entities.append(Entity.from_dict(item))
        return entities
