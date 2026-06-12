# ACCG RAG Agent

基于 ACCG 代码图的 ReAct Agent，使用纯文本协议调用本地 LLM 进行仓库级代码问答。

## 安装

```bash
# 先安装 ACCG 建图核心
git clone https://github.com/Yoyoblue0000/accg-core.git
cd accg-core && uv venv && uv pip install -e .

# 安装 Agent
git clone https://github.com/Yoyoblue0000/accg-rag.git
cd accg-rag && uv venv && uv pip install -e .
```

## 快速开始

```bash
# 单任务
.venv/Scripts/python.exe scripts/run_agent.py "accg/query.py 中 find_symbol 的返回类型？"

# QA 批量评估
.venv/Scripts/python.exe scripts/run_qa.py \
  --project-path /path/to/repo \
  --qa-path /path/to/questions.json \
  --model qwen2.5-coder:14b-instruct \
  --limit 20
```

## 架构

```
mini_agent/
  agent.py        — ReAct 循环、SYSTEM_PROMPT、ANSWER_PROMPT、收敛分析
  model.py        — LLM 接口：流式调用 + THOUGHT/ACTION 解析
  graph_tool.py   — 图查询工具：9 种 action + EmbeddingRanker
  environment.py  — 只读文件工具
scripts/
  run_agent.py    — 单任务入口
  run_qa.py       — QA 批量评估
```

## 协议

纯文本 ReAct 协议，非 OpenAI function calling：

```
THOUGHT: <推理>
ACTION: {"name": "<工具名>", "arguments": {<参数>}}
FINAL: <最终答案>
```

## License

MIT
