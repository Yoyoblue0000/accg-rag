# -*- coding: utf-8 -*-
"""工具定义测试 — 验证 tools_schema.py 的公共接口。"""

from mini_agent.tools_schema import TOOLS


def test_tools_importable():
    """TOOLS 可以被导入且是非空列表。"""
    assert isinstance(TOOLS, list)
    assert len(TOOLS) > 0


def test_tools_count():
    """TOOLS 包含 11 个工具（9 个图查询 + 2 个文件工具）。"""
    assert len(TOOLS) == 11


def test_tools_format():
    """每个工具包含 type, function.name, function.description, function.parameters。"""
    for tool in TOOLS:
        assert "type" in tool, f"工具缺少 type 字段: {tool}"
        assert tool["type"] == "function", f"工具 type 应为 function: {tool}"
        assert "function" in tool, f"工具缺少 function 字段: {tool}"

        func = tool["function"]
        assert "name" in func, f"工具缺少 name: {func}"
        assert "description" in func, f"工具缺少 description: {func}"
        assert "parameters" in func, f"工具缺少 parameters: {func}"


def test_tools_required_params():
    """每个工具有必需参数列表。"""
    for tool in TOOLS:
        params = tool["function"]["parameters"]
        assert "required" in params, f"工具 {tool['function']['name']} 缺少 required"
        assert isinstance(params["required"], list), f"required 应为列表: {tool['function']['name']}"
        assert len(params["required"]) > 0, f"required 不能为空: {tool['function']['name']}"


def test_tools_param_types():
    """参数类型使用标准 JSON Schema（string/integer/number/array）。"""
    valid_types = {"string", "integer", "number", "array", "object", "boolean"}
    for tool in TOOLS:
        params = tool["function"]["parameters"]
        assert "properties" in params, f"工具 {tool['function']['name']} 缺少 properties"
        for prop_name, prop_def in params["properties"].items():
            assert "type" in prop_def, f"参数 {prop_name} 缺少 type"
            assert prop_def["type"] in valid_types, f"参数 {prop_name} 类型无效: {prop_def['type']}"
