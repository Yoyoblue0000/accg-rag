# -*- coding: utf-8 -*-
"""ACCG MCP Server —— 将 ACCG 代码图查询能力暴露为 MCP 工具。

供 OpenCode 等 MCP 客户端调用，复用 mini_agent.tools_schema 中的统一工具定义。

使用方式：
  在 opencode.json 中配置:

  {
    "mcp": {
      "accg": {
        "type": "local",
        "command": ["uv", "run", "python", "scripts/accg_mcp_server.py"],
        "enabled": true,
        "environment": {
          "PROJECT_PATH": "/path/to/your/project"
        }
      }
    }
  }

  或直接命令行启动测试:
    PROJECT_PATH=/path/to/project uv run python scripts/accg_mcp_server.py

环境变量：
  PROJECT_PATH           目标项目路径（必填）
  ACCG_ENABLE_EMBEDDINGS 启用向量语义检索，默认 0
"""

import json
import os
import sys
from pathlib import Path

from mcp.server import Server
from mcp.server.stdio import stdio_server
from mcp.types import Tool, TextContent

# 确保项目根目录在 sys.path 中，以便 import mini_agent
_PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from mini_agent.graph_tool import GraphTool  # noqa: E402
from mini_agent.tools_schema import TOOLS  # noqa: E402

# ── 全局状态 ──────────────────────────────────────────────

server = Server("accg")
graph_tool: GraphTool | None = None
_target_project: str = ""


def _get_graph_tool() -> GraphTool:
    global graph_tool, _target_project

    project_path = os.environ.get("PROJECT_PATH", "")
    if not project_path:
        raise RuntimeError(
            "未设置 PROJECT_PATH 环境变量。请在 opencode.json 的 mcp.accg.environment 中设置，"
            "或直接设置环境变量 PROJECT_PATH=你的项目路径"
        )

    enable_embeddings = os.environ.get("ACCG_ENABLE_EMBEDDINGS", "").lower() in {
        "1", "true", "yes", "on",
    }

    if graph_tool is None or project_path != _target_project:
        graph_tool = GraphTool(
            project_path=project_path,
            enable_embeddings=enable_embeddings,
        )
        _target_project = project_path

    if not graph_tool.is_ready:
        result = graph_tool.ensure_built()
        sys.stderr.write(f"[accg-mcp] {result}\n")
        sys.stderr.flush()

    return graph_tool


# ── 工具处理器 ────────────────────────────────────────────

async def _handle_contextualize(args: dict) -> list[TextContent]:
    """定位符号并返回源码 + 调用关系 + 继承/实例化信息"""
    gt = _get_graph_tool()
    result = gt.execute_full(
        "contextualize",
        name=args.get("name", args.get("symbol", "")),
        limit=args.get("limit", 3),
    )
    return [TextContent(type="text", text=result)]


async def _handle_narrow_down(args: dict) -> list[TextContent]:
    """基于线索词精简候选符号"""
    gt = _get_graph_tool()
    result = gt.execute_full(
        "narrow_down",
        clues=args.get("clues", []),
        limit=args.get("limit", 5),
    )
    return [TextContent(type="text", text=result)]


async def _handle_extract_clues(args: dict) -> list[TextContent]:
    """从源码文本中提取可在图上定位的符号"""
    gt = _get_graph_tool()
    result = gt.execute_full(
        "extract_clues",
        source=args.get("source", ""),
    )
    return [TextContent(type="text", text=result)]


async def _handle_transitive_callers(args: dict) -> list[TextContent]:
    """传递调用者：谁调用了该符号（BFS 可达）"""
    gt = _get_graph_tool()
    result = gt.execute_full(
        "transitive_callers",
        symbol=(
            args.get("symbol")
            or args.get("name")
            or args.get("function_id")
            or ""
        ),
        max_depth=args.get("max_depth", 3),
        min_confidence=args.get("min_confidence", 0.45),
    )
    return [TextContent(type="text", text=result)]


async def _handle_transitive_callees(args: dict) -> list[TextContent]:
    """传递被调用者：该符号调用了谁"""
    gt = _get_graph_tool()
    result = gt.execute_full(
        "transitive_callees",
        symbol=(
            args.get("symbol")
            or args.get("name")
            or args.get("function_id")
            or ""
        ),
        max_depth=args.get("max_depth", 3),
        min_confidence=args.get("min_confidence", 0.45),
    )
    return [TextContent(type="text", text=result)]


async def _handle_call_paths(args: dict) -> list[TextContent]:
    """查找两个符号之间的调用路径"""
    gt = _get_graph_tool()
    result = gt.execute_full(
        "call_paths",
        source=args.get("source", ""),
        target=args.get("target", ""),
        max_depth=args.get("max_depth", 5),
        min_confidence=args.get("min_confidence", 0.45),
    )
    return [TextContent(type="text", text=result)]


async def _handle_class_hierarchy(args: dict) -> list[TextContent]:
    """查询类的继承层次（父类 + 子类）"""
    gt = _get_graph_tool()
    result = gt.execute_full(
        "class_hierarchy",
        class_name=(
            args.get("class_name")
            or args.get("symbol")
            or args.get("name")
            or ""
        ),
    )
    return [TextContent(type="text", text=result)]


async def _handle_module_tree(args: dict) -> list[TextContent]:
    """查看项目的目录树结构"""
    gt = _get_graph_tool()
    result = gt.execute_full(
        "module_tree",
        prefix=args.get("prefix", ""),
    )
    return [TextContent(type="text", text=result)]


async def _handle_module_structure(args: dict) -> list[TextContent]:
    """查看模块结构（文件中定义的符号）"""
    gt = _get_graph_tool()
    result = gt.execute_full(
        "module_structure",
        prefix=args.get("prefix", ""),
    )
    return [TextContent(type="text", text=result)]


async def _handle_read_file(args: dict) -> list[TextContent]:
    """读取项目中的文件（带行号和窗口范围）"""
    from mini_agent.environment import EnvConfig, Environment  # noqa: E402

    project_path = os.environ.get("PROJECT_PATH", "")
    env = Environment(EnvConfig(cwd=project_path))

    result = env.read_file(
        path=args.get("path", ""),
        start_line=args.get("start_line", 0),
        end_line=args.get("end_line", 0),
        context=args.get("context", 0),
    )
    return [TextContent(type="text", text=result)]


async def _handle_list_dir(args: dict) -> list[TextContent]:
    """列出项目中的目录内容"""
    from mini_agent.environment import EnvConfig, Environment  # noqa: E402

    project_path = os.environ.get("PROJECT_PATH", "")
    env = Environment(EnvConfig(cwd=project_path))

    result = env.list_dir(path=args.get("path", ""))
    return [TextContent(type="text", text=result)]


# ── 工具处理器映射 ────────────────────────────────────────

_HANDLERS = {
    "contextualize": _handle_contextualize,
    "narrow_down": _handle_narrow_down,
    "extract_clues": _handle_extract_clues,
    "transitive_callers": _handle_transitive_callers,
    "transitive_callees": _handle_transitive_callees,
    "call_paths": _handle_call_paths,
    "class_hierarchy": _handle_class_hierarchy,
    "module_tree": _handle_module_tree,
    "module_structure": _handle_module_structure,
    "read_file": _handle_read_file,
    "list_dir": _handle_list_dir,
}


# ── MCP 协议处理 ──────────────────────────────────────────


@server.list_tools()
async def list_tools() -> list[Tool]:
    """从 tools_schema.py 导入工具定义，转换为 MCP 格式。"""
    tools = []
    for tool_def in TOOLS:
        func = tool_def["function"]
        tools.append(Tool(
            name=func["name"],
            description=func["description"],
            inputSchema=func["parameters"],
        ))
    return tools


@server.call_tool()
async def call_tool(name: str, arguments: dict) -> list[TextContent]:
    handler = _HANDLERS.get(name)
    if handler is None:
        return [TextContent(type="text", text=f"[错误] 未知工具: {name}")]
    try:
        return await handler(arguments)
    except Exception as e:
        return [TextContent(type="text", text=f"[错误] 工具执行失败: {e}")]


# ── 入口 ──────────────────────────────────────────────────


async def main():
    sys.stderr.write(
        f"[accg-mcp] ACCG MCP Server 启动\n"
        f"[accg-mcp] PROJECT_PATH={os.environ.get('PROJECT_PATH', '(未设置)')}\n"
    )
    sys.stderr.flush()

    async with stdio_server() as (read_stream, write_stream):
        await server.run(read_stream, write_stream, server.create_initialization_options())


if __name__ == "__main__":
    import asyncio
    asyncio.run(main())
