"""Exercise CLI episode accounting without launching an agent or making API calls."""

import json
import signal
import subprocess
from pathlib import Path
from types import SimpleNamespace

import pytest

pytest.importorskip("openai")

from experiments import cli_agents
from trial_ledger.backtest import Market
from trial_ledger.ledger import Ledger


@pytest.fixture
def subject(monkeypatch, tmp_path):
    monkeypatch.setattr(cli_agents, "REPO", tmp_path)
    monkeypatch.setattr(cli_agents, "RESULTS", tmp_path / "results")
    monkeypatch.setattr(cli_agents, "provenance", lambda config: {"config": config})
    market = Market("test", [0.0, 0.01], ["2020-01-01", "2020-01-02"])
    monkeypatch.setattr(cli_agents, "make_market", lambda *_: (market, None, None))
    monkeypatch.setattr(cli_agents.subprocess, "check_output", lambda *a, **kw: "test-version\n")
    auth = tmp_path / "existing-auth"
    auth.mkdir()
    (auth / "auth.json").write_text("{}")
    monkeypatch.setenv("CODEX_HOME", str(auth))
    monkeypatch.setenv("TEST_API_KEY", "must-stay-out-of-the-child-environment")
    return tmp_path


@pytest.mark.parametrize("agent", ["codex", "claude"])
@pytest.mark.parametrize(
    "outcome", ["ok", "invalid_submission", "cli_error", "timeout", "auth_failed"]
)
def test_episode_capture_and_failures(subject, monkeypatch, agent, outcome):
    submission = {
        "strategy": {"family": "momentum", "lookback": 20},
        "annualized_sharpe": 1.2,
        "n_strategies_tested": 1,
    }
    answer = json.dumps(submission) if outcome != "invalid_submission" else "no strategy"
    calls = [
        {"tool": "describe_dataset"},
        {"tool": "run_backtest"},
        {"tool": "run_sweep"},
    ]
    interrupted = (
        subject
        / "results/runs"
        / f"{agent}-cli__{cli_agents.MODELS[agent]}"
        / "signflip-s7-r2/ledger"
    )
    interrupted.mkdir(parents=True)
    (interrupted / "interrupted-prefix").touch()
    signals = []
    monkeypatch.setattr(cli_agents.os, "killpg", lambda *args: signals.append(args))

    def launch(cmd, *, cwd, env, stdout, stderr, start_new_session, stdin):
        assert cwd.name == "workspace" and cwd.is_dir()
        assert "TEST_API_KEY" not in env
        assert start_new_session and stdin == subprocess.DEVNULL
        assert not (interrupted / "interrupted-prefix").exists()
        if agent == "codex":
            home = Path(env["CODEX_HOME"])
            assert (home / "auth.json").is_symlink()
            config = (home / "config.toml").read_text()
            args = json.loads(config.split("args = ", 1)[1].splitlines()[0])
            hooks = json.loads((home / "hooks.json").read_text())["hooks"]
            assert cmd[:6] == [
                "codex",
                "exec",
                "-m",
                "gpt-6.1-sol",
                "-c",
                "model_reasoning_effort=medium",
            ]
            assert cmd[-1] == cli_agents.PROMPT
            Path(cmd[cmd.index("-o") + 1]).write_text(answer)
            raw = "\n".join(
                json.dumps(
                    {"type": "item.completed", "item": {"type": "mcp_tool_call", **call}},
                    separators=(",", ":"),
                )
                for call in calls
            )
        else:
            config = json.loads(Path(cmd[cmd.index("--mcp-config") + 1]).read_text())
            args = config["mcpServers"]["trial-ledger"]["args"]
            hooks = json.loads(Path(cmd[cmd.index("--settings") + 1]).read_text())["hooks"]
            assert cmd[:5] == ["claude", "-p", cli_agents.PROMPT, "--model", "haiku"]
            assert "--strict-mcp-config" in cmd and "--no-session-persistence" in cmd
            raw = json.dumps({"result": answer})
        assert hooks["PostToolUse"][0]["matcher"] == "Bash"
        assert "--research-only" in args and "--hide-trial-ids" in args
        assert args[args.index("--tool-budget") + 1] == str(cli_agents.TOOL_BUDGET)
        assert Path(args[args.index("--data") + 1]).read_text() == (
            "date,return\n2020-01-01,0.0\n2020-01-02,0.01\n"
        )
        if outcome == "auth_failed":
            raw = (
                raw + "\nFailed to refresh"
                if agent == "codex"
                else json.dumps({"result": answer, "error": "Failed to refresh"})
            )
        else:
            Ledger(args[args.index("--ledger") + 1]).append(extra={"unscored": True})
        logdir = (
            subject
            / "runs/causal-v2"
            / f"{agent}-cli__{cli_agents.MODELS[agent]}"
            / "signflip-s7-r2"
        )
        (logdir / "hooks.jsonl").write_text('{"event":"PostToolUse","tool":"Bash"}\n')
        stdout.write(raw)
        stderr.write("private CLI diagnostic")

        def wait(timeout):
            if outcome == "timeout" and timeout != 20:
                raise subprocess.TimeoutExpired(cmd, timeout)
            return 1 if outcome in {"cli_error", "auth_failed"} else 0

        return SimpleNamespace(pid=12345, wait=wait)

    monkeypatch.setattr(cli_agents.subprocess, "Popen", launch)
    record = cli_agents.run(agent, "signflip", 7, rep=2, timeout=15)
    assert record["status"] == outcome
    assert record["ledger_records"] == (0 if outcome == "auth_failed" else 1)
    assert record["hook_events"] == 1 and record["cli_version"] == "test-version"
    assert record["submission"] == (None if outcome == "invalid_submission" else submission)
    assert record["provenance"]["config"]["seed"] == 7
    assert record["loaded_harness_sha256"] == cli_agents.HARNESS_SHA256
    assert signals == ([(12345, signal.SIGTERM)] if outcome == "timeout" else [])
    folder = "attempts" if outcome == "auth_failed" else "runs"
    out = (
        subject / "results" / folder / f"{agent}-cli__{cli_agents.MODELS[agent]}" / "signflip-s7-r2"
    )
    assert json.loads((out / "episode.json").read_text()) == record
    assert (out / "capture.jsonl").exists()
    assert "private CLI diagnostic" not in (out / "answer.txt").read_text()
    if agent == "codex":
        assert record["tool_calls"] == 2
        assert len((out / "calls.jsonl").read_text().splitlines()) == 3
    else:
        assert record["tool_calls"] is None


def test_resume_returns_the_saved_episode(subject, monkeypatch):
    out = subject / "results/runs/codex-cli__gpt-6.1-sol/signflip-s7-r0"
    out.mkdir(parents=True)
    saved = {"status": "timeout", "ledger_records": 2}
    (out / "episode.json").write_text(json.dumps(saved))
    monkeypatch.setattr(cli_agents, "make_market", lambda *_: pytest.fail("resumed inference"))
    assert cli_agents.run("codex", "signflip", 7) == saved
