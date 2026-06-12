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

# 仅运行确定性候选检索基线
.venv/Scripts/python.exe scripts/run_qa.py \
  --project-path /path/to/repo \
  --qa-path /path/to/questions.json \
  --retrieval-only --limit 20

# 检索基线 + embedding 增强
.venv/Scripts/python.exe scripts/run_qa.py \
  --project-path /path/to/repo \
  --qa-path /path/to/questions.json \
  --retrieval-only --embedding --limit 20
```

QA 结果保留完整参考答案、候选排名、检索阶段和降级原因，并生成
`*.summary.json` 汇总 Recall@1/3/5/10、MRR、NDCG 和 fallback 计数。
当前 gold 由参考答案中的 Python 路径、反引号符号和限定名启发式提取；
NDCG 使用候选文档是否命中任一 gold 的二元相关性，后续可替换为人工标签。
Embedding 默认关闭；可用 `--embedding` 或单任务环境变量
`ACCG_ENABLE_EMBEDDINGS=1` 显式启用。

## 架构

```
mini_agent/
  agent.py        — ReAct 循环、SYSTEM_PROMPT、ANSWER_PROMPT、收敛分析
  model.py        — LLM 接口：流式调用 + THOUGHT/ACTION 解析
  graph_tool.py   — 图查询工具：9 种 action + 可选 EmbeddingRanker
  retrieval.py    — 精确/BM25/embedding/模糊候选检索级联
  retrieval_metrics.py — 临时 gold 抽取与检索指标
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
