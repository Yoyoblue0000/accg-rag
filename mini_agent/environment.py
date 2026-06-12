# -*- coding: utf-8 -*-
"""环境接口 -- 只读文件操作"""
import logging
from pathlib import Path

logger = logging.getLogger("mini_agent.env")

DENY_PATTERNS = {".git", ".env", ".ssh", "id_rsa", "id_ed25519", "credentials", "secrets"}


class EnvConfig:
    def __init__(self, cwd: str = ""):
        self.cwd = str(Path(cwd).resolve()) if cwd else str(Path.cwd())


class Environment:
    """只读环境，提供 read_file / list_dir"""

    def __init__(self, config: EnvConfig):
        self.config = config
        self._root = Path(self.config.cwd)

    def _resolve_inside_root(self, path: str) -> Path | None:
        """解析路径，若越界或命中敏感路径则返回 None"""
        candidate = (self._root / path).resolve()
        try:
            candidate.relative_to(self._root)
        except ValueError:
            return None
        # 拒绝敏感路径
        for part in candidate.parts:
            if part in DENY_PATTERNS or part.startswith(".env"):
                return None
        return candidate

    def read_file(self, path: str, start_line: int = 0, end_line: int = 0, context: int = 0) -> str:
        """读取文件内容，返回格式化的带行号文本"""
        full_path = self._resolve_inside_root(path)
        if full_path is None:
            return f"[错误] 路径越界或不可访问: {path}"
        if not full_path.is_file():
            return f"[错误] 文件不存在: {path}"
        try:
            lines = full_path.read_text(encoding="utf-8", errors="replace").splitlines()
        except Exception as e:
            return f"[错误] 无法读取: {e}"

        total = len(lines)
        if start_line and end_line:
            begin = max(0, start_line - 1)
            end = min(total, end_line + (context or 0))
            if context:
                begin = max(0, begin - context)
        elif start_line:
            begin = max(0, start_line - 1)
            end = min(total, begin + 30)
        else:
            begin = 0
            end = min(total, 200)

        result = [f"[{path} 行 {begin+1}-{end} / 共 {total} 行]"]
        for i in range(begin, end):
            result.append(f"{i+1:4d}| {lines[i]}")
        return "\n".join(result)

    def list_dir(self, path: str = "") -> str:
        """列出目录内容"""
        target = self._resolve_inside_root(path) if path else self._root
        if target is None:
            return f"[错误] 路径越界: {path}"
        if not target.is_dir():
            return f"[错误] 目录不存在: {path}"

        items = []
        try:
            for p in sorted(target.iterdir(), key=lambda x: (x.is_file(), x.name.lower())):
                name = p.name
                if name.startswith(".") and name not in (".gitignore", ".env.example"):
                    continue
                suffix = "/" if p.is_dir() else ""
                items.append(f"  {name}{suffix}")
        except Exception as e:
            return f"[错误] 列目录失败: {e}"

        return f"{path or '(根目录)'}:\n" + "\n".join(items[:100])
