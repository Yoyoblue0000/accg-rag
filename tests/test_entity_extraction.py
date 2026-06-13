# -*- coding: utf-8 -*-
"""实体提取器测试。"""

import json

import pytest

from mini_agent.entity_extractor import Entity, EntityExtractor


class _FakeModel:
    """返回预设 JSON 的模型桩。"""

    def __init__(self, response: str):
        self._response = response
        self.last_messages = None

    def generate(self, messages: list[dict]) -> str:
        self.last_messages = messages
        return self._response


def test_extract_single_entity():
    extractor = EntityExtractor(_FakeModel(json.dumps([
        {"name": "format_header", "query": "format_header",
         "description": "格式化头部的函数", "type_hint": "FUNCTION"},
    ])))
    entities = extractor.extract("What does format_header do?")
    assert len(entities) == 1
    assert entities[0].name == "format_header"
    assert entities[0].type_hint == "FUNCTION"


def test_extract_two_entities_for_comparison():
    extractor = EntityExtractor(_FakeModel(json.dumps([
        {"name": "format_header", "query": "format_header",
         "description": "格式化头部的函数", "type_hint": "FUNCTION"},
        {"name": "OutputStreamFormatter", "query": "OutputStreamFormatter",
         "description": "输出流格式化器类", "type_hint": "CLASS"},
    ])))
    entities = extractor.extract(
        "Compare format_header and OutputStreamFormatter"
    )
    assert len(entities) == 2
    assert entities[0].name == "format_header"
    assert entities[1].name == "OutputStreamFormatter"


def test_extract_relation_question():
    extractor = EntityExtractor(_FakeModel(json.dumps([
        {"name": "parse", "query": "parse",
         "description": "解析函数", "type_hint": "FUNCTION"},
        {"name": "render", "query": "render",
         "description": "渲染函数", "type_hint": "FUNCTION"},
    ])))
    entities = extractor.extract("How does parse invoke render?")
    assert len(entities) == 2
    assert all(e.type_hint == "FUNCTION" for e in entities)


def test_fallback_on_invalid_json():
    extractor = EntityExtractor(_FakeModel("not valid json at all"))
    entities = extractor.extract("What is the relationship?")
    assert len(entities) == 1
    assert entities[0].query == "What is the relationship?"


def test_fallback_on_empty_array():
    extractor = EntityExtractor(_FakeModel("[]"))
    entities = extractor.extract("What is the relationship?")
    assert len(entities) == 1
    assert entities[0].query == "What is the relationship?"


def test_fallback_on_missing_fields():
    extractor = EntityExtractor(_FakeModel(json.dumps([
        {"name": "foo"},
    ])))
    entities = extractor.extract("What is foo?")
    assert len(entities) == 1
    assert entities[0].name == "foo"


def test_entity_to_dict_roundtrip():
    entity = Entity(
        name="test_func",
        query="test_func keyword",
        description="A test function",
        type_hint="FUNCTION",
    )
    d = entity.to_dict()
    restored = Entity.from_dict(d)
    assert restored.name == entity.name
    assert restored.query == entity.query
    assert restored.description == entity.description
    assert restored.type_hint == entity.type_hint


def test_extract_trims_extra_fields():
    """LLM 返回多余字段时不应报错，只提取需要的字段。"""
    extractor = EntityExtractor(_FakeModel(json.dumps([
        {"name": "foo", "query": "foo bar", "description": "desc",
         "type_hint": "FUNCTION", "extra_field": "should be ignored",
         "confidence": 0.9},
    ])))
    entities = extractor.extract("test")
    assert len(entities) == 1
    assert entities[0].name == "foo"


def test_model_receives_question_in_prompt():
    model = _FakeModel(json.dumps([
        {"name": "x", "query": "x", "description": "x", "type_hint": "FUNCTION"},
    ]))
    extractor = EntityExtractor(model)
    extractor.extract("Explain the call chain from A to B")
    user_msg = model.last_messages[0]["content"]
    assert "A to B" in user_msg
    assert "实体" in user_msg
