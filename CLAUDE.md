# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## 项目概述

基于 ACCG 代码图的 ReAct Agent，使用 OpenAI function calling 协议调用本地 Ollama 模型进行仓库级代码问答。依赖 [accg-core](https://github.com/Yoyoblue0000/accg-core) 提供图构建与查询能力。

## 常用命令

```bash
# 安装
uv venv && uv pip install -e .

# 本地单题测试
.venv/Scripts/python.exe scripts/run_agent.py "问题描述"

# 运行全部测试
.venv/Scripts/python.exe -m pytest tests/ -v

# 运行单个测试文件
.venv/Scripts/python.exe -m pytest tests/test_sufficiency.py -v

# 服务器：QA 批量测试
ssh amd-jk6kg8k@10.67.8.138 "cd ~/program/accg-rag && .venv/bin/python scripts/run_qa.py \
  --project-path ~/program/sqlfluff_repo \
  --qa-path ~/program/test_repos/sqlfluff_qa.json \
  --model qwen2.5:14b --embedding --id 1 2 7 --output /tmp/qa_p4.json"

# 服务器：仅跑指定题目
ssh ... --id 1 2 7 33 42 --output /tmp/qa_p4_fixed.json

# MCP 服务器启动（供 OpenCode 等客户端调用）
PROJECT_PATH=/path/to/project uv run python scripts/accg_mcp_server.py
```

## 模型默认值

| 组件 | 默认 | 覆盖参数 |
|------|------|---------|
| LLM | `qwen2.5:14b` | `--model` / `OLLAMA_MODEL` |
| Embedding | `mxbai-embed-large`（334M） | `--embedding-model` / `EMBEDDING_MODEL` |
| 重排 | 不启用 | `--reranker-model` / `RERANKER_MODEL` |

Embedding 需显式传 `--embedding` 才会启用，否则仅走确定性检索（exact_id → exact_symbol → lexical → fuzzy）。

**重要**：`qwen2.5-coder` 系列不支持 function calling，请使用 `qwen2.5` 系列。

## 协议

使用 OpenAI function calling 协议。LLM 通过 `tool_calls` 字段调用工具，不再使用纯文本 ACTION 格式。

工具定义统一在 `mini_agent/tools_schema.py`，供 Agent 内部和 MCP 服务器复用。

**兼容性**：当 Ollama 流式响应不支持 `tool_calls` 时，Model 层会自动从 `content` 中解析 JSON 格式的工具调用（回退方案）。

## 核心流程

```
Agent.run(task)
  │
  ├─ 1. 建图 + EmbeddingRanker.build_index（首次慢，磁盘缓存加速）
  ├─ 2. EntityExtractor → 按实体执行 4 阶段级联检索 → 合并候选并选择锚点
  ├─ 3. 验证锚点 → 预取源码 → 构建系统提示
  ├─ 4. ReAct 循环（max 15 步）
  │     └─ model.query(messages, tools=TOOLS) → tool_calls → 工具执行 → 证据收集
  ├─ 5. 完成请求（无 tool_calls 时判定完成）
  ├─ 6. SufficiencyGate 确定性检查
  │     ├─ 通过 → _synthesize()
  │     └─ 未通过 → 最多 2 次有界关系扩展并继续探索
  └─ 7. _synthesize()：问题 + 完整证据 + 最新草稿 + 搜索范围 → 最终答案
```

## 关键设计

- **OpenAI function calling**：使用结构化 tool_calls，不再依赖纯文本解析
- **锚点选择分数驱动**：候选按实际匹配分数排序，类型（FUNCTION/CLASS/METHOD）仅在同分时打破平局
- **SufficiencyGate**：按单实体、比较、调用/数据流、继承、实例化和否定结论执行确定性规则
- **受控关系扩展**：最多 2 次、深度 ≤2、最低置信度 0.45，记录完整审计
- **证据展示 COMPLETE 优先**：探索阶段源码先尝试完整展示，预算不够再降级
- **模型历史压缩**：移除被拒 FINAL，旧工具结果替换为证据 ID 和查询计划摘要
- **EmbeddingRanker 磁盘缓存**：指纹校验，代码不变则直接加载
- **RetrievalConfig / GateConfig**：检索分数/权重/乘数和门控阈值集中为可配置 dataclass

## 4 阶段加权级联检索

1. **RECALL**：lexical 与 embedding 各取 top-200，归一化后合并为召回池
2. **PRECISION**：`exact_id` / `exact_symbol` 仅提升召回池内候选
3. **REFINEMENT**：fuzzy 仅对召回池做文本对齐
4. **RANKING**：按固定权重计算 `0..1` 相关度并稳定排序

```text
final_score =
    0.35 * norm_lexical
  + 0.35 * norm_embedding
  + 0.20 * exact_bonus
  + 0.10 * fuzzy_bonus
```

## MCP 服务器

`scripts/accg_mcp_server.py` 将 ACCG 代码图查询能力暴露为 MCP 工具，供 OpenCode 等客户端调用。

配置方式（在 opencode.json 中）：
```json
{
  "mcp": {
    "accg-rag": {
      "type": "local",
      "command": ["uv", "run", "python", "scripts/accg_mcp_server.py"],
      "enabled": true,
      "environment": {"PROJECT_PATH": "."}
    }
  }
}
```

## 服务器验证

依赖 Ollama，本地 GPU 有限，QA 批量测试在服务器上执行。

### 服务器环境

| 项目 | 详情 |
|------|------|
| 地址 | `ssh amd-jk6kg8k@10.67.8.138`（密钥 `~/.ssh/id_ed25519`） |
| 硬件 | AMD Ryzen AI MAX+ 395，128GB 统一内存，Radeon 8060S（gfx1151） |
| Python | `.venv/bin/python` |
| Ollama API | `http://localhost:11434/v1` |
| GPU 驱动 | ROCm 7.2.3 |

### GPU 模型兼容性

⚠️ **Qwen3 全系（含 MoE）在 gfx1151 上输出为空，不可用。**
⚠️ **Qwen2.5-coder 系列不支持 function calling，请使用 qwen2.5 系列。**

| 模型 | 架构 | GPU | 速度 | Function Calling | 备注 |
|------|------|-----|------|------------------|------|
| qwen2.5:14b | Dense 14.8B | ✅ | ~20 t/s | ✅ | **当前默认** |
| qwen2.5:7b | Dense 7.6B | ✅ | ~30 t/s | ✅ | 轻量级 |
| qwen2.5:72b | Dense 72.7B | ✅ | 4.5 t/s | ✅ | 太慢，不适合批量 |
| qwen2.5-coder:14b-instruct | Dense 14.8B | ✅ | 19 t/s | ❌ | 不支持 function calling |
| qwen2.5-coder:32b | Dense 32.8B | ✅ | ~4.5 t/s | ❌ | 不支持 function calling |
| qwen3:30b | MoE 30.5B | ❌ | — | — | GPU 输出为空 |
| mxbai-embed-large | 334M | ✅ | ~15ms | — | Embedding 推荐 |
| nomic-embed-text | 137M | ✅ | 15ms | — | 备选 embedding |

### 同步流程

```bash
# 正式部署：按 Git SHA 发布。确保本地工作区干净。
ssh amd-jk6kg8k@10.67.8.138 "cd ~/program/accg-rag && git fetch origin && git checkout <sha> && ~/.local/bin/uv pip install -e . && echo DEPLOY_OK"

# QA 全量（含门禁）
ssh amd-jk6kg8k@10.67.8.138 "cd ~/program/accg-rag && ~/.local/bin/uv run python scripts/run_qa.py \
  --project-path ~/program/test_repos/sqlfluff_repo \
  --qa-path ~/program/test_repos/sqlfluff_qa.json \
  --model qwen2.5:14b --limit 20 \
  --judge-model qwen2.5:14b --judge-threshold 0.7 \
  --fail-on-error --prohibit-dirty"

# 查看进度
ssh amd-jk6kg8k@10.67.8.138 "cat /tmp/qa_results.json | python3 -c 'import json; d=json.load(open(\"/tmp/qa_results.json\")); print(len(d), \"done\")'"
```

### 常见问题

| 症状 | 可能原因 | 解决 |
|------|---------|------|
| QA 进程卡住 | 前次残留进程抢 GPU | `pkill -f run_qa.py` |
| 输出全为空 | 用了 qwen3 系列 | 换 `--model qwen2.5:14b` |
| embedding 极慢 | 多模型抢 VRAM | `sudo systemctl restart ollama` |
| tool_calls 为空 | 用了 qwen2.5-coder | 换 `qwen2.5` 系列 |

## 编码规范

- 注释和提交信息使用中文
- BFS 队列必须使用 `collections.deque`
- 集合成员检查使用 `set`，禁止对列表做 `in` 扫描
- 删除代码时同步清理相关的 import 和注释
- 新增可调参数通过 config dataclass（`RetrievalConfig`/`GateConfig`）暴露，不直接写模块级常量
