# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## 项目概述

基于 ACCG 代码图的 ReAct Agent，使用纯文本协议调用本地 Ollama 模型进行仓库级代码问答。依赖 [accg-core](https://github.com/Yoyoblue0000/accg-core) 提供图构建与查询能力。

## 架构

```
mini_agent/
  agent.py         — ReAct 循环、SYSTEM_PROMPT/ANSWER_PROMPT、锚点预取、扩展执行、消息压缩
  model.py         — LLM 接口：流式调用 + THOUGHT/ACTION/FINAL 解析 + finish_reason 捕获
  graph_tool.py    — 图查询工具：9 种 action + EmbeddingRanker（摘要向量增强 + 磁盘缓存）
  retrieval.py     — 4 阶段加权级联检索 + 候选排序 + 锚点选择
  sufficiency.py   — FinishAction 解析、确定性证据充分性门控、受控扩展计划、GateConfig
  evidence.py      — 证据账本：分层展示（COMPLETE/PREVIEW/SNIPPET/FOLD）、预算控制、合成选择
  environment.py   — 只读文件工具：read_file / list_dir
  query_plan.py    — 查询计划与锚点数据结构
  retrieval_metrics.py — 检索评估：临时 gold 提取、recall/MRR/NDCG、锚点 PR/type coverage
  reranker.py      — 可选在线重排器（小模型二次排序候选人）
scripts/
  run_agent.py          — 单任务入口
  run_qa.py             — QA 批量评估（支持 --id、--embedding、即时写入）
  build_summary_index.py — 离线 7B 摘要索引构建（用于 embedding 增强）
tests/
  test_agent_model.py      — model 层解析测试
  test_agent_graph_tool.py — 候选排序、锚点选择、预取、扩展、合成测试
  test_sufficiency.py      — 门控通过/拒绝、实体过滤、否定语义测试
  test_query_plan.py       — 锚点选择排序、验证、预取预算测试
  test_retrieval_metrics.py — gold 提取、检索指标计算测试
```

## 常用命令

```bash
# 安装
uv venv && uv pip install -e .

# 本地单题测试
.venv/Scripts/python.exe scripts/run_agent.py "问题描述"

# 运行全部测试（234 条）
.venv/Scripts/python.exe -m pytest tests/ -v

# 运行单个测试文件
.venv/Scripts/python.exe -m pytest tests/test_sufficiency.py -v

# 服务器：QA 批量测试（32B + embedding）
ssh amd-jk6kg8k@10.67.8.138 "cd ~/program/accg-rag && .venv/bin/python scripts/run_qa.py \
  --project-path ~/program/sqlfluff_repo \
  --qa-path /home/amd-jk6kg8k/program/sqlfluff_qa.json \
  --model qwen2.5-coder:32b --embedding --id 1 2 7 --output /tmp/qa_p4.json"

# 服务器：仅跑指定题目
ssh ... --id 1 2 7 33 42 --output /tmp/qa_p4_fixed.json
```

## 模型默认值

| 组件 | 默认 | 覆盖参数 |
|------|------|---------|
| LLM | `qwen2.5-coder:14b-instruct` | `--model` / `OLLAMA_MODEL` |
| Embedding | `mxbai-embed-large`（334M） | `--embedding-model` / `EMBEDDING_MODEL` |
| 重排 | 不启用 | `--reranker-model` / `RERANKER_MODEL` |

Embedding 需显式传 `--embedding` 才会启用，否则仅走确定性检索（exact_id → exact_symbol → lexical → fuzzy）。

## 协议

纯文本 ReAct，非 OpenAI function calling。LLM 输出：

```
THOUGHT: <推理>
ACTION: {"name": "<工具名>", "arguments": {<参数>}}
FINAL: <最终答案>
```

无依赖时可并行写多个 ACTION（最多 2 个）。model.py 将图操作自动包装为 `query_graph(action=..., ...)`。

## 核心流程

```
Agent.run(task)
  │
  ├─ 1. 建图 + EmbeddingRanker.build_index（首次慢，磁盘缓存加速）
  ├─ 2. EntityExtractor → 按实体执行 4 阶段级联检索 → 合并候选并选择锚点
  ├─ 3. 验证锚点 → 预取源码 → 构建系统提示
  ├─ 4. ReAct 循环（max 15 步）
  │     └─ model.query() → THOUGHT/ACTION 解析 → 工具执行 → 证据收集
  ├─ 5. 完成请求
  │     ├─ FINAL 文本 → FinishAction(draft)
  │     └─ finish_reason="stop" / 无工具 → 仅表示本轮传输结束
  ├─ 6. SufficiencyGate 确定性检查
  │     ├─ 通过 → _synthesize()
  │     └─ 未通过 → 最多 2 次有界关系扩展并继续探索
  └─ 7. _synthesize()：问题 + 完整证据 + 最新草稿 + 搜索范围 → 最终答案
```

## 关键设计

- **传输结束 ≠ 任务完成**：`finish_reason="stop"` 不绕过证据门控
- **锚点选择分数驱动**：候选按实际匹配分数排序，类型（FUNCTION/CLASS/METHOD）仅在同分时打破平局，不再是硬编码优先级
- **SufficiencyGate**：按单实体、比较、调用/数据流、继承、实例化和否定结论执行确定性规则；Gate 实体过滤排除工具名、代码字面量、表达式片段，`::` 实体需各部分在同一源码中验证
- **受控关系扩展**：最多 2 次、深度 ≤2、最低置信度 0.45，记录完整审计（source_evidence_ids）
- **证据展示 COMPLETE 优先**：探索阶段源码先尝试完整展示（COMPLETE），预算不够再降级到 PREVIEW/SNIPPET/FOLD，避免模型看不到完整内容而重复请求
- **模型历史压缩**：移除被拒 FINAL，旧工具结果替换为证据 ID 和查询计划摘要
- **EmbeddingRanker 磁盘缓存**：指纹校验（含摘要文本），代码不变则直接加载 `.accg/embeddings_*.json`
- **摘要向量增强**：离线 7B 模型生成函数英文一行摘要，作为 embedding 输入文本
- **CJK 分词**：`tokenize()` 提取中文单字参与检索，中文查询不再被丢弃
- **RetrievalConfig / GateConfig**：检索分数/权重/乘数和门控阈值集中为可配置 dataclass
- **重复调用拦截**：最近 5 条 action 去重 + contextualize 符号去重

## 4 阶段加权级联检索

1. **RECALL**：lexical 与 embedding 各取 top-200，分别按本次查询最大正分归一化，合并后截断为 200 条召回池。
2. **PRECISION**：`exact_id` / `exact_symbol` 仅提升召回池内候选，不从池外注入条目。
3. **REFINEMENT**：fuzzy 仅对召回池做文本对齐。
4. **RANKING**：按固定权重计算 `0..1` 相关度并稳定排序。

```text
final_score =
    0.35 * norm_lexical
  + 0.35 * norm_embedding
  + 0.20 * exact_bonus
  + 0.10 * fuzzy_bonus
```

Agent 检索前强制运行 `EntityExtractor`。单实体使用清洗后的 `entity.query`；多实体分别检索后按 ID 合并，并优先保证每个实体都有锚点覆盖。提取失败时回退到原问题全文。

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

| 模型 | 架构 | GPU | 速度 | 备注 |
|------|------|-----|------|------|
| qwen2.5-coder:14b-instruct | Dense 14.8B | ✅ | 19 t/s | 当前默认 |
| qwen2.5-coder:32b (Q4_K_M) | Dense 32.8B | ✅ | ~4.5 t/s | 高质量，慢 |
| qwen2.5:72b | Dense 72.7B | ✅ | 4.5 t/s | 太慢，不适合批量 |
| qwen3:30b | MoE 30.5B | ❌ | — | GPU 输出为空 |
| mxbai-embed-large | 334M | ✅ | ~15ms | Embedding 推荐 |
| nomic-embed-text | 137M | ✅ | 15ms | 备选 embedding |

### 同步流程

```bash
# 正式部署：按 Git SHA 发布。确保本地工作区干净。
ssh amd-jk6kg8k@10.67.8.138 "cd ~/program/accg-rag && git fetch origin && git checkout <sha> && ~/.local/bin/uv pip install -e . && echo DEPLOY_OK"

# QA 全量（含门禁）
ssh amd-jk6kg8k@10.67.8.138 "cd ~/program/accg-rag && ~/.local/bin/uv run python scripts/run_qa.py \
  --project-path ~/program/test_repos/requests_repo \
  --qa-path ~/program/test_repos/sweqa_requests.json \
  --model qwen2.5-coder:14b-instruct --limit 20 \
  --judge-model qwen2.5:14b --judge-threshold 0.7 \
  --fail-on-error --prohibit-dirty"

# 查看进度
ssh amd-jk6kg8k@10.67.8.138 "cat /tmp/qa_results.json | python3 -c 'import json; d=json.load(open(\"/tmp/qa_results.json\")); print(len(d), \"done\")'"
```

### 常见问题

| 症状 | 可能原因 | 解决 |
|------|---------|------|
| QA 进程卡住 | 前次残留进程抢 GPU | `pkill -f run_qa.py` |
| 输出全为空 | 用了 qwen3 系列 | 换 `--model qwen2.5-coder:14b-instruct` |
| embedding 极慢 | 多模型抢 VRAM | `sudo systemctl restart ollama` |
| `finish_reason` 为 None | Ollama 版本太旧 | 升级到 0.24.0+ |

## 编码规范

- 注释和提交信息使用中文
- BFS 队列必须使用 `collections.deque`
- 集合成员检查使用 `set`，禁止对列表做 `in` 扫描
- 删除代码时同步清理相关的 import 和注释
- 新增可调参数通过 config dataclass（`RetrievalConfig`/`GateConfig`）暴露，不直接写模块级常量
