import io
import json
import sys

import numpy as np
import pytest

from trial_ledger import cli, hooks
from trial_ledger.ledger import Ledger
from trial_ledger.report import build_report


@pytest.fixture
def noisy_ledger(tmp_path):
    lg = Ledger(tmp_path / "tl")
    rng = np.random.default_rng(0)
    for i in range(40):
        lg.append(strategy={"i": i}, returns=rng.standard_normal(800) * 0.01, periods_per_year=365)
    return lg


def test_report_on_null_trials(noisy_ledger):
    rep = build_report(noisy_ledger, claimed_n=3)
    assert rep.n_trials == 40 and rep.n_unscored == 0
    assert rep.dsr_ledger < rep.psr0
    assert rep.dsr_claimed > rep.dsr_ledger  # under-reporting N inflates the DSR
    assert 0.0 <= rep.pbo <= 1.0
    assert rep.n_effective == pytest.approx(40, rel=0.1)


def test_report_dedups_identical_reruns(tmp_path):
    lg = Ledger(tmp_path / "tl")
    r = np.random.default_rng(1).standard_normal(300)
    for _ in range(3):
        lg.append(strategy={"same": True}, returns=r)
    lg.append(strategy={"other": True}, returns=r[::-1].copy())
    rep = build_report(lg)
    assert rep.n_records == 4 and rep.n_trials == 2


def test_report_counts_unscored_runs(tmp_path):
    lg = Ledger(tmp_path / "tl")
    rng = np.random.default_rng(2)
    lg.append(strategy={"a": 1}, returns=rng.standard_normal(300))
    lg.append(strategy={"b": 1}, returns=rng.standard_normal(300))
    base = build_report(lg).dsr_ledger
    for _ in range(20):
        lg.append(extra={"unscored": True})
    rep = build_report(lg)
    assert rep.n_unscored == 20
    assert rep.dsr_ledger < base


def test_cli_verify_and_report(noisy_ledger, capsys):
    d = str(noisy_ledger.dir)
    assert cli.main(["--dir", d, "verify", "--check-returns"]) == 0
    assert "OK: 40 records" in capsys.readouterr().out
    assert cli.main(["--dir", d, "report", "--claimed-n", "5", "--json"]) == 0
    out = json.loads(capsys.readouterr().out)
    assert out["n_trials"] == 40 and out["claimed_n"] == 5
    assert cli.main(["--dir", d, "report"]) == 0
    assert "DSR with ledger N=40" in capsys.readouterr().out
    lines = noisy_ledger.path.read_text().splitlines(keepends=True)
    noisy_ledger.path.write_text("".join(lines[:10] + lines[11:]))
    assert cli.main(["--dir", d, "verify"]) == 1
    assert "TAMPERED" in capsys.readouterr().out


def test_cli_log_and_head(tmp_path, capsys):
    d = str(tmp_path / "tl")
    f = tmp_path / "r.csv"
    np.savetxt(f, np.random.default_rng(3).standard_normal(100), delimiter=",")
    assert cli.main(["--dir", d, "log", "--strategy", '{"k": 1}', "--returns", str(f)]) == 0
    seq_hash = capsys.readouterr().out.strip()
    cli.main(["--dir", d, "head"])
    assert capsys.readouterr().out.strip() == seq_hash


def test_cli_exec_logs_emitted_trials(tmp_path, capsys):
    d = str(tmp_path / "tl")
    script = tmp_path / "grid.py"
    script.write_text(
        "from trial_ledger.hooks import emit\n"
        "for k in range(7):\n"
        "    emit(0.01 * k, 500, {'k': k})\n"
    )
    rc = cli.main(["--dir", d, "exec", "--", sys.executable, str(script)])
    assert rc == 0
    recs = Ledger(d).records()
    assert len(recs) == 7 and recs[3]["strategy"] == {"k": 3}
    assert all(r["source"] == "exec" for r in recs)


def _claude_event(command, stdout="", cwd=None):
    return {
        "session_id": "s-1",
        "hook_event_name": "PostToolUse",
        "tool_name": "Bash",
        "cwd": cwd,
        "tool_input": {"command": command, "description": "run"},
        "tool_response": {"stdout": stdout, "stderr": "", "interrupted": False},
    }


def test_hook_logs_emitted_trials(tmp_path):
    lg = Ledger(tmp_path / "tl")
    out = "\n".join(
        [
            "loading data",
            'TRIAL_LEDGER {"sharpe": 0.05, "n_obs": 1000, "strategy": {"lb": 10}}',
            'TRIAL_LEDGER {"sharpe": 0.07, "n_obs": 1000, "strategy": {"lb": 20}}',
            "TRIAL_LEDGER {not json",
        ]
    )
    written = hooks.log_shell_event(_claude_event("python sweep.py", out), lg, agent="claude")
    assert len(written) == 2
    assert lg.records()[1]["strategy"] == {"lb": 20}
    assert lg.records()[0]["session_id"] == "s-1"


def test_hook_logs_unscored_backtest_and_ignores_other_commands(tmp_path):
    lg = Ledger(tmp_path / "tl")
    assert hooks.log_shell_event(_claude_event("ls -la"), lg) == []
    assert hooks.log_shell_event(_claude_event("git status"), lg) == []
    w = hooks.log_shell_event(_claude_event("uv run python backtest_ma.py --fast 5"), lg)
    assert len(w) == 1 and w[0]["extra"]["unscored"] and w[0]["sharpe"] is None
    assert hooks.log_shell_event({**_claude_event("python x.py"), "tool_name": "Read"}, lg) == []


def test_hook_accepts_codex_style_argv(tmp_path):
    lg = Ledger(tmp_path / "tl")
    event = {
        "tool_name": "shell",
        "tool_input": {"command": ["bash", "-lc", "python run_backtest.py"]},
        "output": 'TRIAL_LEDGER {"sharpe": 0.02, "n_obs": 250}',
    }
    assert len(hooks.log_shell_event(event, lg, agent="codex")) == 1


def test_hook_main_never_fails(monkeypatch, tmp_path):
    monkeypatch.setattr(sys, "stdin", io.StringIO("not json"))
    assert hooks.main(["--dir", str(tmp_path / "tl")]) == 0
    ev = _claude_event("python backtest.py", 'TRIAL_LEDGER {"sharpe": 0.1, "n_obs": 99}')
    monkeypatch.setattr(sys, "stdin", io.StringIO(json.dumps(ev)))
    assert hooks.main(["--dir", str(tmp_path / "tl")]) == 0
    assert len(Ledger(tmp_path / "tl").records()) == 1
