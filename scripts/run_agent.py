# -*- coding: utf-8 -*-
"""入口：运行 mini_agent

选项：
  --verbose       每步打印发给模型的上下文统计（消息数、总字符、估算 token）
  --dump-context  运行结束后将每次模型请求的完整上下文写入 context_dump.json
  --embedding     启用向量语义检索

环境变量：
  OLLAMA_URL       Ollama API 地址，默认 http://localhost:11434/v1
  OLLAMA_MODEL     模型名，默认 qwen2.5-coder:14b-instruct
  PROJECT_PATH     项目路径，默认本仓库根目录

注意：Ollama 默认 num_ctx=2048，远小于合成 prompt 的 ~6000 token。
如果模型答案与证据不符，先检查上下文是否被截断。
"""

import json
import os
import sys
from pathlib import Path
from mini_agent.model import Model, ModelConfig
from mini_agent.environment import Environment, EnvConfig
from mini_agent.agent import Agent


def _estimate_tokens(text: str) -> int:
    """粗略估算 token 数：英文 ~4 chars/token，中文 ~1.5 chars/token。"""
    cjk = sum(1 for c in text if "一" <= c <= "鿿")
    ascii_chars = len(text) - cjk
    return (ascii_chars // 4) + int(cjk / 1.5)


def main():
    base_url = os.environ.get("OLLAMA_URL", "http://localhost:11434/v1")
    model_name = os.environ.get("OLLAMA_MODEL", "qwen2.5-coder:14b-instruct")
    project_path = os.environ.get("PROJECT_PATH", str(Path(__file__).resolve().parent.parent))
    enable_embeddings = os.environ.get(
        "ACCG_ENABLE_EMBEDDINGS",
        "",
    ).lower() in {"1", "true", "yes", "on"}

    args = sys.argv[1:]
    verbose = "--verbose" in args
    dump_context = "--dump-context" in args
    if "--embedding" in args:
        enable_embeddings = True
    task_parts = [a for a in args if not a.startswith("--")]
    task = " ".join(task_parts) if task_parts else (
        "accg/resolver/call_resolver.py 中 CallResolver 类的 resolve 方法的核心逻辑是什么？"
    )

    model = Model(ModelConfig(
        base_url=base_url,
        api_key="ollama",
        model_name=model_name,
    ))

    env = Environment(EnvConfig(cwd=project_path))

    def _on_step(m):
        if m.role == "system":
            return
        elif m.role == "user":
            print(f"\n[任务] {task}")
        elif m.role == "assistant":
            print(f"\n{'─'*40}\n[Step {m.step}] LLM\n{'─'*40}")
            print(m.content)
        elif m.role == "tool":
            tag = " (拦截)" if m.intercepted else ""
            print(f"\n[Step {m.step}] {m.tool_name}{tag}")
            print(m.content)

    # 上下文审计回调
    context_dumps = []
    context_log_path = Path.cwd() / "context_log.txt"
    if verbose:
        context_log_path.write_text("", encoding="utf-8")  # 每次运行清空

    def _on_audit(audit_text: str):
        if verbose:
            import re
            stats_match = re.search(
                r"msgs=(\d+)\s+chars=(\d+)\s+est_tokens=(\d+)",
                audit_text,
            )
            if stats_match:
                msgs = int(stats_match.group(1))
                chars = int(stats_match.group(2))
                tokens = int(stats_match.group(3))
                header = audit_text.split("\n")[0] if audit_text else ""
                stage = header.split("|")[1].strip() if "|" in header else "?"
                stats_line = (
                    f"[上下文] {stage} | "
                    f"{msgs} 条消息, "
                    f"{chars:,} 字符, "
                    f"~{tokens:,} tokens"
                )
                print(f"\n{stats_line}")
                warning = ""
                if tokens > 30000:
                    warning = (
                        f"  ⚠ 估算 {tokens:,} tokens，"
                        f"接近模型上下文上限 32768，可能截断！"
                    )
                    print(warning)
                # 写入日志文件
                with open(context_log_path, "a", encoding="utf-8") as f:
                    f.write(f"{stats_line}\n")
                    if warning:
                        f.write(f"{warning}\n")
                    f.write(f"{audit_text}\n\n")
            else:
                est = _estimate_tokens(audit_text)
                line = f"[上下文] ~{est:,} tokens"
                print(f"\n{line}")
                with open(context_log_path, "a", encoding="utf-8") as f:
                    f.write(f"{line}\n{audit_text}\n\n")

        if dump_context:
            context_dumps.append(audit_text)

    from mini_agent.graph_tool import GraphTool
    from mini_agent.multi_entity import EntityExtractor
    graph_tool = GraphTool(
        project_path,
        enable_embeddings=enable_embeddings,
    )
    entity_extractor = EntityExtractor(model)
    agent = Agent(
        model, env,
        graph_tool=graph_tool, max_steps=15,
        on_step=_on_step,
        on_audit=_on_audit,
        entity_extractor=entity_extractor,
    )

    result = agent.run(task)

    if dump_context and context_dumps:
        dump_path = Path.cwd() / "context_dump.json"
        dump_data = {
            "task": task,
            "model": model_name,
            "requests": [
                {"index": i, "audit_text": text}
                for i, text in enumerate(context_dumps)
            ],
        }
        dump_path.write_text(
            json.dumps(dump_data, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        print(f"\n[上下文导出] {dump_path} ({len(context_dumps)} 次模型请求)")

    if result.error:
        print(f"\n[错误] {result.error}")
        sys.exit(1)
    if result.synthesis:
        print(f"\n{'─'*40}\n[合成答案]\n{'─'*40}")
        print(result.synthesis.answer)
    print(f"\n{'='*60}")
    print(f"结果: {result.answer}")
    print(f"({result.rounds}轮, {result.explorations}探)")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
