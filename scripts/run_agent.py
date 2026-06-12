# -*- coding: utf-8 -*-
"""入口：运行 mini_agent"""

import os
import sys
from pathlib import Path
from mini_agent.model import Model, ModelConfig
from mini_agent.environment import Environment, EnvConfig
from mini_agent.agent import Agent
from mini_agent.graph_tool import GraphTool


def main():
    base_url = os.environ.get("OLLAMA_URL", "http://localhost:11434/v1")
    model_name = os.environ.get("OLLAMA_MODEL", "qwen2.5-coder:14b-instruct")
    project_path = os.environ.get("PROJECT_PATH", str(Path(__file__).resolve().parent.parent))

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

    graph_tool = GraphTool(project_path)
    agent = Agent(model, env, graph_tool=graph_tool, max_steps=15, on_step=_on_step)

    if len(sys.argv) > 1:
        task = " ".join(sys.argv[1:])
    else:
        task = "accg/resolver/call_resolver.py 中 CallResolver 类的 resolve 方法的核心逻辑是什么？"

    result = agent.run(task)
    if result.error:
        print(f"\n[错误] {result.error}")
        return
    if result.synthesis:
        print(f"\n{'─'*40}\n[合成答案]\n{'─'*40}")
        print(result.synthesis.answer)
    print(f"\n{'='*60}")
    print(f"结果: {result.answer}")
    print(f"({result.rounds}轮, {result.explorations}探)")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
