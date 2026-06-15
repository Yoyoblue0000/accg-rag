# -*- coding: utf-8 -*-
"""统一工具定义 — 供 Agent（OpenAI function calling）和 MCP 服务器复用。

工具定义格式遵循 OpenAI function calling 规范，与 MCP 的 JSON Schema 兼容。
"""

TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "contextualize",
            "description": "定位代码符号并返回完整上下文：源码、调用关系（calls/called_by）、类继承层次、实例化信息。适用于需要深入理解某个函数或类的场景。",
            "parameters": {
                "type": "object",
                "properties": {
                    "name": {
                        "type": "string",
                        "description": "符号名或完整节点 ID，如 'resolve' 或 'module::Class::method'"
                    },
                    "limit": {
                        "type": "integer",
                        "description": "返回结果上限，默认 3"
                    }
                },
                "required": ["name"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "narrow_down",
            "description": "基于关键词线索在候选符号中精确定位。当初步搜索结果太多时，用此工具缩小范围。",
            "parameters": {
                "type": "object",
                "properties": {
                    "clues": {
                        "type": "array",
                        "items": {"type": "string"},
                        "description": "关键词列表，如 ['resolve', 'url']"
                    },
                    "limit": {
                        "type": "integer",
                        "description": "返回结果上限，默认 5"
                    }
                },
                "required": ["clues"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "extract_clues",
            "description": "从源码文本中提取可在图上定位的函数名/类名/方法名。用于从一段代码中找出可以进一步查询的符号。",
            "parameters": {
                "type": "object",
                "properties": {
                    "source": {
                        "type": "string",
                        "description": "源码文本"
                    }
                },
                "required": ["source"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "transitive_callers",
            "description": "查找传递调用链：哪些函数（直接或间接）调用了目标符号。BFS 遍历，可控制深度和置信度。",
            "parameters": {
                "type": "object",
                "properties": {
                    "symbol": {
                        "type": "string",
                        "description": "目标符号名或完整 ID"
                    },
                    "max_depth": {
                        "type": "integer",
                        "description": "最大搜索深度，默认 3"
                    },
                    "min_confidence": {
                        "type": "number",
                        "description": "调用边最低置信度，默认 0.45"
                    }
                },
                "required": ["symbol"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "transitive_callees",
            "description": "查找传递被调用链：目标符号（直接或间接）调用了哪些函数。BFS 遍历，可控制深度和置信度。",
            "parameters": {
                "type": "object",
                "properties": {
                    "symbol": {
                        "type": "string",
                        "description": "目标符号名或完整 ID"
                    },
                    "max_depth": {
                        "type": "integer",
                        "description": "最大搜索深度，默认 3"
                    },
                    "min_confidence": {
                        "type": "number",
                        "description": "调用边最低置信度，默认 0.45"
                    }
                },
                "required": ["symbol"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "call_paths",
            "description": "查找从源符号到目标符号的调用路径。用于理解两个函数之间是如何通过调用链连接的。",
            "parameters": {
                "type": "object",
                "properties": {
                    "source": {
                        "type": "string",
                        "description": "源符号名或完整 ID"
                    },
                    "target": {
                        "type": "string",
                        "description": "目标符号名或完整 ID"
                    },
                    "max_depth": {
                        "type": "integer",
                        "description": "最大搜索深度，默认 5"
                    },
                    "min_confidence": {
                        "type": "number",
                        "description": "调用边最低置信度，默认 0.45"
                    }
                },
                "required": ["source", "target"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "class_hierarchy",
            "description": "查询类的继承层次，返回父类和子类列表。",
            "parameters": {
                "type": "object",
                "properties": {
                    "class_name": {
                        "type": "string",
                        "description": "类名或完整节点 ID"
                    }
                },
                "required": ["class_name"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "module_tree",
            "description": "查看项目的目录树结构。输入空字符串或目录前缀即可。",
            "parameters": {
                "type": "object",
                "properties": {
                    "prefix": {
                        "type": "string",
                        "description": "目录前缀，空字符串表示根目录"
                    }
                },
                "required": ["prefix"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "module_structure",
            "description": "查看模块（文件）中定义了哪些符号（函数、类、方法等）。",
            "parameters": {
                "type": "object",
                "properties": {
                    "prefix": {
                        "type": "string",
                        "description": "文件路径前缀，如 'src/utils/'"
                    }
                },
                "required": ["prefix"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "read_file",
            "description": "读取项目中的文件内容，带行号和上下文字段。支持指定行号范围和上下文窗口。",
            "parameters": {
                "type": "object",
                "properties": {
                    "path": {
                        "type": "string",
                        "description": "相对于项目根目录的文件路径"
                    },
                    "start_line": {
                        "type": "integer",
                        "description": "起始行号，0 表示从头开始"
                    },
                    "end_line": {
                        "type": "integer",
                        "description": "结束行号，0 表示到末尾"
                    },
                    "context": {
                        "type": "integer",
                        "description": "上下文窗口行数，在 start_line 前额外展示的行数"
                    }
                },
                "required": ["path"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "list_dir",
            "description": "列出项目目录中的文件和子目录。",
            "parameters": {
                "type": "object",
                "properties": {
                    "path": {
                        "type": "string",
                        "description": "相对于项目根目录的路径，空字符串表示根目录"
                    }
                },
                "required": ["path"]
            }
        }
    },
]
