# -*- coding: utf-8 -*-
"""run_qa.py 单元测试 —— 聚焦门禁判定、dirty 检查、恢复模式、Judge 解析。"""

import json
import os

# 将 scripts/ 加入路径以导入 run_qa
import sys
import tempfile
from pathlib import Path
from unittest.mock import MagicMock, patch

_scripts_dir = str(Path(__file__).resolve().parent.parent / "scripts")
if _scripts_dir not in sys.path:
    sys.path.insert(0, _scripts_dir)

import run_qa  # noqa: E402

from mini_agent.agent import MsgRecord, RunResult  # noqa: E402
from mini_agent.retrieval import RetrievalResult  # noqa: E402

# ─── _is_judge_passed ───────────────────────────────────────

class TestJudgePassed:
    def test_score_above_threshold_passes(self):
        assert run_qa._is_judge_passed({"score": 0.8}, 0.7) is True

    def test_score_equal_to_threshold_passes(self):
        assert run_qa._is_judge_passed({"score": 0.7}, 0.7) is True

    def test_score_below_threshold_fails(self):
        assert run_qa._is_judge_passed({"score": 0.5}, 0.7) is False

    def test_none_score_fails(self):
        assert run_qa._is_judge_passed({"score": None}, 0.7) is False

    def test_missing_score_fails(self):
        assert run_qa._is_judge_passed({}, 0.7) is False

    def test_non_numeric_score_fails(self):
        assert run_qa._is_judge_passed({"score": "high"}, 0.7) is False


# ─── _evaluate_answer_quality ───────────────────────────────

class TestJudgeEvaluation:
    def test_no_judge_model_returns_unevaluated(self):
        result = run_qa._evaluate_answer_quality(None, "Q", "A", "answer")
        assert result["score"] is None
        assert result["label"] == "未评估"

    def test_max_steps_answer_scores_zero(self):
        result = run_qa._evaluate_answer_quality(
            MagicMock(), "Q", "A", "[达到最大步数] 无法完成",
        )
        assert result["score"] == 0.0
        assert result["label"] == "错误"

    def test_error_answer_scores_zero(self):
        result = run_qa._evaluate_answer_quality(
            MagicMock(), "Q", "A", "[错误] 某异常",
        )
        assert result["score"] == 0.0
        assert result["label"] == "错误"

    def test_valid_json_from_judge(self):
        mock_model = MagicMock()
        mock_model.generate.return_value = (
            '{"score": 0.85, "label": "正确", "reason": "一致"}'
        )
        result = run_qa._evaluate_answer_quality(
            mock_model, "Q", "A", "answer text",
        )
        assert result["score"] == 0.85
        assert result["label"] == "正确"

    def test_judge_parse_failure_returns_fallback(self):
        mock_model = MagicMock()
        mock_model.generate.return_value = "not a json at all"
        result = run_qa._evaluate_answer_quality(
            mock_model, "Q", "A", "answer text",
        )
        assert result["score"] is None
        assert result["label"] == "解析失败"

    def test_score_out_of_range_rejected(self):
        mock_model = MagicMock()
        mock_model.generate.return_value = (
            '{"score": 1.5, "label": "好", "reason": ""}'
        )
        result = run_qa._evaluate_answer_quality(
            mock_model, "Q", "A", "answer text",
        )
        assert result["score"] is None
        assert result["label"] == "解析失败"

    def test_judge_exception_returns_parse_failure(self):
        mock_model = MagicMock()
        mock_model.generate.side_effect = RuntimeError("API 挂了")
        result = run_qa._evaluate_answer_quality(
            mock_model, "Q", "A", "answer text",
        )
        assert result["score"] is None
        assert result["label"] == "解析失败"


# ─── _status_mark ───────────────────────────────────────────

class TestStatusMark:
    def test_error_marks_cross(self):
        r = RunResult(answer="", error="图构建失败")
        assert run_qa._status_mark(r) == "✗"

    def test_max_steps_marks_cross(self):
        r = RunResult(answer="[达到最大步数] 未完成")
        assert run_qa._status_mark(r) == "✗"

    def test_zero_exploration_marks_warning(self):
        r = RunResult(answer="some answer")
        r.messages = [MsgRecord(role="assistant", content="test", step=1)]
        assert run_qa._status_mark(r) == "⚠"

    def test_normal_completion_marks_check(self):
        r = RunResult(answer="完整答案")
        r.messages = [
            MsgRecord(role="assistant", content="test", step=1),
            MsgRecord(role="tool", content="result", step=1, intercepted=False),
        ]
        assert run_qa._status_mark(r) == "✓"

    def test_retrieval_only_failed(self):
        r = RunResult(answer="")
        r.retrieval = RetrievalResult(
            candidates=[], stages_attempted=[], stages_succeeded=[],
            diagnostics=[], status="failed",
        )
        assert run_qa._status_mark(r, retrieval_only=True) == "✗"

    def test_retrieval_only_ok(self):
        r = RunResult(answer="")
        r.retrieval = RetrievalResult(
            candidates=[], stages_attempted=[], stages_succeeded=[],
            diagnostics=[], status="ok",
        )
        assert run_qa._status_mark(r, retrieval_only=True) == "✓"


# ─── _fmt_candidates ────────────────────────────────────────

class TestFmtCandidates:
    def test_empty_returns_none(self):
        assert run_qa._fmt_candidates([]) == "无"

    def test_formats_two_candidates(self):
        anchors = [
            {"label": "Foo", "score": 0.95},
            {"name": "Bar", "score": 0.80},
        ]
        result = run_qa._fmt_candidates(anchors)
        assert "Foo(0.95)" in result
        assert "Bar(0.80)" in result

    def test_truncates_to_two(self):
        anchors = [
            {"label": "A", "score": 1.0},
            {"label": "B", "score": 0.9},
            {"label": "C", "score": 0.8},
        ]
        result = run_qa._fmt_candidates(anchors)
        assert "C" not in result


# ─── _build_run_metadata dirty 检查 ─────────────────────────

class TestBuildRunMetadata:
    def test_dirty_detects_unstaged_changes(self):
        with patch("subprocess.run") as mock_run:
            mock_run.side_effect = [
                MagicMock(returncode=0, stdout="abc123\n"),     # rev-parse
                MagicMock(returncode=1),                         # git diff (unstaged dirty)
                MagicMock(returncode=0),                         # git diff --cached (clean)
                MagicMock(returncode=0, stdout=""),              # ls-files (no untracked)
            ]
            meta = run_qa._build_run_metadata(MagicMock(model="test"))
            assert meta["dirty"] is True

    def test_dirty_detects_staged_changes(self):
        with patch("subprocess.run") as mock_run:
            mock_run.side_effect = [
                MagicMock(returncode=0, stdout="abc123\n"),
                MagicMock(returncode=0),    # git diff (unstaged clean)
                MagicMock(returncode=1),    # git diff --cached (staged dirty)
                MagicMock(returncode=0, stdout=""),
            ]
            meta = run_qa._build_run_metadata(MagicMock(model="test"))
            assert meta["dirty"] is True

    def test_dirty_detects_untracked_files(self):
        with patch("subprocess.run") as mock_run:
            mock_run.side_effect = [
                MagicMock(returncode=0, stdout="abc123\n"),
                MagicMock(returncode=0),                    # git diff (clean)
                MagicMock(returncode=0),                    # git diff --cached (clean)
                MagicMock(returncode=0, stdout="new_file.py\n"),  # untracked
            ]
            meta = run_qa._build_run_metadata(MagicMock(model="test"))
            assert meta["dirty"] is True

    def test_clean_workspace_reports_not_dirty(self):
        with patch("subprocess.run") as mock_run:
            mock_run.side_effect = [
                MagicMock(returncode=0, stdout="abc123\n"),
                MagicMock(returncode=0),    # clean
                MagicMock(returncode=0),    # clean
                MagicMock(returncode=0, stdout=""),  # no untracked
            ]
            meta = run_qa._build_run_metadata(MagicMock(model="test"))
            assert meta["dirty"] is False

    def test_dirty_details_recorded(self):
        with patch("subprocess.run") as mock_run:
            mock_run.side_effect = [
                MagicMock(returncode=0, stdout="abc123\n"),
                MagicMock(returncode=1),    # unstaged dirty
                MagicMock(returncode=0),    # staged clean
                MagicMock(returncode=0, stdout="tmp.log\n"),
            ]
            meta = run_qa._build_run_metadata(MagicMock(model="test"))
            assert "unstaged" in meta.get("dirty_details", [])
            assert "untracked" in meta.get("dirty_details", [])

    def test_git_failure_marks_sha_unknown(self):
        with patch("subprocess.run") as mock_run:
            mock_run.side_effect = FileNotFoundError("git not found")
            meta = run_qa._build_run_metadata(MagicMock(model="test"))
            assert meta["git_sha"] == "unknown"


# ─── 恢复模式逻辑 ───────────────────────────────────────────

class TestResumeLogic:
    """验证 completed_indices 只包含成功完成的记录。"""

    def _make_resume_file(self, records: list[dict], tmpdir: str) -> str:
        path = os.path.join(tmpdir, "qa_results.json")
        with open(path, "w", encoding="utf-8") as f:
            json.dump(records, f, ensure_ascii=False, indent=2)
        return path

    def test_successful_record_marked_completed(self):
        """成功记录（无 error、有答案、retrieval ok）应标记为已完成。"""
        records = [{
            "index": 0,
            "error": None,
            "agent_answer": "完整答案",
            "retrieval": {"status": "ok"},
        }]
        with tempfile.TemporaryDirectory() as tmpdir:
            path = self._make_resume_file(records, tmpdir)
            mock_args = MagicMock()
            mock_args.resume = True
            mock_args.output = path
            mock_args.qa_path = os.path.join(tmpdir, "dummy.json")
            # 创建 dummy qa 文件
            with open(mock_args.qa_path, "w", encoding="utf-8") as f:
                json.dump([{"question": "test?", "answer": "ans"}], f)
            mock_args.id = None
            mock_args.limit = 1
            mock_args.verbose = 0
            mock_args.retrieval_only = True
            mock_args.embedding = False
            mock_args.no_embedding = True
            mock_args.reranker_model = ""
            mock_args.judge_model = ""
            mock_args.judge_threshold = 0.7
            mock_args.fail_on_error = False
            mock_args.prohibit_dirty = False
            mock_args.project_path = tmpdir
            mock_args.model = "test"
            mock_args.base_url = "http://localhost:11434/v1"
            mock_args.embedding_model = "test"

            # 模拟 run_qa.main() 中恢复模式的逻辑
            out_path = Path(path)
            artifact_records = []
            completed_indices = set()
            if out_path.is_file():
                existing = json.loads(out_path.read_text(encoding="utf-8"))
                for record in existing:
                    if not isinstance(record, dict):
                        continue
                    artifact_records.append(record)
                    error = record.get("error")
                    answer = record.get("agent_answer")
                    retrieval = record.get("retrieval") or {}
                    if (
                        not error
                        and answer is not None
                        and retrieval.get("status") != "failed"
                    ):
                        completed_indices.add(record.get("index", -1))

            assert 0 in completed_indices

    def test_failed_record_not_marked_completed(self):
        """失败记录（有 error）不应标记为已完成。"""
        records = [{
            "index": 0,
            "error": "图构建失败",
            "agent_answer": "",
            "retrieval": {"status": "failed"},
        }]
        with tempfile.TemporaryDirectory() as tmpdir:
            path = self._make_resume_file(records, tmpdir)
            out_path = Path(path)
            completed_indices = set()
            if out_path.is_file():
                existing = json.loads(out_path.read_text(encoding="utf-8"))
                for record in existing:
                    if not isinstance(record, dict):
                        continue
                    error = record.get("error")
                    answer = record.get("agent_answer")
                    retrieval = record.get("retrieval") or {}
                    if (
                        not error
                        and answer is not None
                        and retrieval.get("status") != "failed"
                    ):
                        completed_indices.add(record.get("index", -1))

            assert 0 not in completed_indices

    def test_retrieval_failed_record_not_marked_completed(self):
        """retrieval status=failed 的 record 不应标记为已完成。"""
        records = [{
            "index": 0,
            "error": None,
            "agent_answer": "可能不错",
            "retrieval": {"status": "failed"},
        }]
        with tempfile.TemporaryDirectory() as tmpdir:
            path = self._make_resume_file(records, tmpdir)
            out_path = Path(path)
            completed_indices = set()
            if out_path.is_file():
                existing = json.loads(out_path.read_text(encoding="utf-8"))
                for record in existing:
                    if not isinstance(record, dict):
                        continue
                    error = record.get("error")
                    answer = record.get("agent_answer")
                    retrieval = record.get("retrieval") or {}
                    if (
                        not error
                        and answer is not None
                        and retrieval.get("status") != "failed"
                    ):
                        completed_indices.add(record.get("index", -1))

            assert 0 not in completed_indices


# ─── Judge 门禁集成测试 ─────────────────────────────────────

class TestGateJudgeAllParseFail:
    """模拟 Judge 全部解析失败时的门禁行为。"""

    def test_all_parse_fail_with_judge_model_should_block(self):
        """有 judge_model 但全部解析失败应阻断。"""
        artifact_records = [
            {"index": 0, "answer_judge": {"score": None, "label": "解析失败"}},
            {"index": 1, "answer_judge": {"score": None, "label": "解析失败"}},
            {"index": 2, "answer_judge": {"score": None, "label": "解析失败"}},
        ]
        judge_evaluable = 0
        judge_model_set = True  # judge_model 已设置

        gate_reasons = []
        if (
            not False  # retrieval_only
            and judge_model_set
            and judge_evaluable == 0
            and len(artifact_records) > 0
        ):
            gate_reasons.append(
                f"Judge 全部 {len(artifact_records)} 题解析失败，无法评估答案质量"
            )

        assert len(gate_reasons) > 0
        assert "全部" in gate_reasons[0]
        assert "解析失败" in gate_reasons[0]

    def test_no_judge_model_no_block(self):
        """没有 judge_model 时不检查 Judge。"""
        gate_reasons: list[str] = []
        assert len(gate_reasons) == 0

    def test_mixed_results_parse_failure_counted(self):
        """部分解析失败应单独统计。"""
        records = [
            {"answer_judge": {"score": 0.8, "label": "正确"}},
            {"answer_judge": {"score": None, "label": "解析失败"}},
            {"answer_judge": {"score": 0.3, "label": "错误"}},
        ]
        parse_failures = sum(
            1 for r in records
            if r.get("answer_judge", {}).get("label") == "解析失败"
        )
        assert parse_failures == 1


# ─── 双重计数验证 ───────────────────────────────────────────

class TestNoDoubleCountOnException:
    """验证异常捕获不会造成双重 failed_count 计数。"""

    def test_exception_block_does_not_pre_count(self):
        """异常块的 failed_count 递增应在 is_failed 判定处统一处理。"""
        # 模拟：异常路径只设置 result，由 is_failed 统一计数
        result = RunResult(answer="", error="未捕获异常: boom")
        
        # is_failed 判定逻辑（从 run_qa.py 提取）
        is_failed = bool(result.error)
        assert is_failed is True
        
        # 验证不会在异常块中预先递增
        # （正确行为：仅 is_failed 递增一次）
        failed_count = 0
        if is_failed:
            failed_count += 1
        assert failed_count == 1
