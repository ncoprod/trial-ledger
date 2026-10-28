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
    assert len(written) == 3  # the unparseable line is still counted, as malformed
    assert lg.records()[1]["strategy"] == {"lb": 20}
    assert lg.records()[2]["extra"]["malformed"] is True
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


MIXED_OUTPUT = "\n".join(
    [
        'TRIAL_LEDGER {"sharpe": 0.05, "n_obs": 1000, "strategy": {"lb": 10}}',
        'TRIAL_LEDGER {"sharpe": "abc", "n_obs": 1000}',
        'TRIAL_LEDGER {"sharpe": 0.03, "n_obs": 12.5}',
        'TRIAL_LEDGER {"sharpe": 0.03, "n_obs": 500, "strategy": "not-an-object"}',
        "TRIAL_LEDGER {broken json",
        "TRIAL_LEDGER [1, 2, 3]",
        'TRIAL_LEDGER {"sharpe": NaN, "n_obs": 500}',
        'TRIAL_LEDGER {"sharpe": 0.04, "n_obs": 1000, "periods_per_year": -1}',
        'TRIAL_LEDGER {"sharpe": 0.06, "n_obs": 1000, "strategy": {"lb": 30}}',
    ]
)
N_ATTEMPTS, N_GOOD = 9, 2


def _check_mixed(records):
    assert len(records) == N_ATTEMPTS  # one record per attempt: nothing is dropped
    good = [r for r in records if not r["extra"].get("malformed")]
    bad = [r for r in records if r["extra"].get("malformed")]
    assert [r["strategy"] for r in good] == [{"lb": 10}, {"lb": 30}]
    assert len(bad) == N_ATTEMPTS - N_GOOD
    assert all(r["sharpe"] is None and r["extra"]["unscored"] for r in bad)
    assert {r["extra"]["error"] for r in bad} == {"ValueError", "TypeError", "JSONDecodeError"}


def test_hook_fails_closed_on_malformed_trials(tmp_path):
    lg = Ledger(tmp_path / "tl")
    written = hooks.log_shell_event(_claude_event("python sweep.py", MIXED_OUTPUT), lg)
    assert len(written) == N_ATTEMPTS
    _check_mixed(lg.records())
    assert lg.verify().ok
    rep = build_report(lg)
    assert rep.n_trials == N_GOOD and rep.n_unscored == N_ATTEMPTS - N_GOOD


def test_exec_fails_closed_on_malformed_trials(tmp_path):
    d = str(tmp_path / "tl")
    script = tmp_path / "mixed.py"
    script.write_text(f"print({MIXED_OUTPUT!r})\n")
    assert cli.main(["--dir", d, "exec", "--", sys.executable, str(script)]) == 0
    _check_mixed(Ledger(d).records())


# Fake, token-shaped strings built at runtime (not real credentials).
FAKE = {
    "nv": "nvapi-" + "Ab3dEf9h" * 6,
    "sk": "sk-proj-" + "Zx81Qw7e" * 5,
    "gh": "gho_" + "Kp4mN7qR" * 5,
    "jwt": "eyJhbGciOiJIUzI1NiJ9." + "eyJzdWIiOiIxIn0" + ".c2lnbmF0dXJlMTIz",
    "hex": "9f86d081884c7d659a2feaa0c55ad015a3bf4f1b2b0b822cd15d6c15b0f00a08",
    "plain": "hunter2-not-token-shaped",
}
SECRET_CMD = (
    f"NVIDIA_API_KEY={FAKE['nv']} export DB_PASS='{FAKE['plain']}'; "
    f'OPENAI_API_KEY="{FAKE["sk"]}" uv run python backtest.py --fast 5 '
    f"--api-key {FAKE['plain']} --token={FAKE['gh']} "
    f"&& curl -H 'Authorization: Bearer {FAKE['jwt']}' "
    f"'https://user:{FAKE['plain']}@example.com/x?token={FAKE['hex']}&n=3' && echo {FAKE['gh']}"
)


def test_default_record_has_hash_and_program_only(tmp_path, monkeypatch):
    monkeypatch.delenv(hooks.CAPTURE_ENV, raising=False)
    lg = Ledger(tmp_path / "tl")
    out = 'TRIAL_LEDGER {"sharpe": 0.05, "n_obs": 1000}'
    hooks.log_shell_event(_claude_event(SECRET_CMD, out), lg)  # scored path
    hooks.log_shell_event(_claude_event(SECRET_CMD), lg)  # unscored path
    raw = lg.path.read_text()
    for value in FAKE.values():
        assert value not in raw, value
    recs = lg.records()
    assert len(recs) == 2
    for r in recs:
        assert "command" not in r["extra"] and "command_preview" not in r["extra"]
        assert r["extra"]["command_sha256"] == hooks.sha256_hex(SECRET_CMD)
        assert r["extra"]["program"] == "uv run python backtest.py"


@pytest.mark.parametrize(
    ("command", "program"),
    [
        ("python3 -u sweep.py --fast 5", "python3 sweep.py"),
        (
            "cd proj && uv run --with pandas python -m experiments.run all",
            "uv run python -m experiments.run",
        ),
        ("FOO=1 node bt.js", "node bt.js"),
        ("/usr/bin/env python /abs/path/backtest.py", "python /abs/path/backtest.py"),
        ("ls -la; git status", "ls"),
        ("python -c 'print(1)'", "python -c"),
        ("echo 'unbalanced", "echo"),
    ],
)
def test_command_program(command, program):
    assert hooks.command_program(command) == program


def test_preview_is_opt_in_and_redacted(tmp_path, monkeypatch):
    monkeypatch.setenv(hooks.CAPTURE_ENV, "1")
    lg = Ledger(tmp_path / "tl")
    hooks.log_shell_event(_claude_event(SECRET_CMD), lg)
    prev = lg.records()[0]["extra"]["command_preview"]
    for value in FAKE.values():
        assert value not in prev, value
    assert "NVIDIA_API_KEY=<redacted>" in prev
    assert "--api-key <redacted>" in prev and "--token=<redacted>" in prev
    assert "backtest.py --fast 5" in prev


def test_exec_does_not_store_raw_argv(tmp_path, monkeypatch):
    monkeypatch.delenv(hooks.CAPTURE_ENV, raising=False)
    d = str(tmp_path / "tl")
    rc = cli.main(["--dir", d, "exec", "--", sys.executable, "-c", "pass", f"--token={FAKE['nv']}"])
    assert rc == 0
    assert FAKE["nv"] not in Ledger(d).path.read_text()


def test_preview_redacts_long_flag_names():
    flag = "--" + "my-very-long-service-name-" * 3 + "api-key"  # > 40 characters
    assert len(flag) > 60
    for cmd in (f"python x.py {flag} hunter2", f"python x.py {flag}=hunter2"):
        prev = hooks.redacted_preview(cmd)
        assert "hunter2" not in prev and "<redacted>" in prev


def test_preview_secret_at_truncation_boundary():
    secret = "Zx81Qw7e" * 4  # 32 chars, letters + digits
    # place the secret so that it straddles the preview length
    pad = "a " * ((hooks.PREVIEW_CHARS - 10) // 2)
    prev = hooks.redacted_preview(f"python x.py {pad}{secret} tail")
    assert len(prev) <= hooks.PREVIEW_CHARS + 4
    assert not any(secret[:k] in prev for k in range(6, len(secret) + 1))
    # ... and at the tokenizer's scan limit, where the token itself gets cut
    filler = "b " * ((hooks.SCAN_CHARS - 12) // 2)
    cmd = f"python x.py {filler}{secret}"
    assert len(cmd) > hooks.SCAN_CHARS
    prev = hooks.redacted_preview(cmd, max_chars=10**6)
    assert not any(secret[:k] in prev for k in range(6, len(secret) + 1))


PATHOLOGICAL = {
    "alnum_run": "a1" * 500_000,
    "letters_run": "a" * 1_000_000,
    "dash_run": "-" * 1_000_000,
    "flag_like": "--x" * 333_334,
    "slashed": "ab12/" * 200_000,
    "assignments": "A=" * 500_000,
    "words": "authorization " * 75_000,
    "quotes": "'a" * 500_000,
}


@pytest.mark.parametrize("name", sorted(PATHOLOGICAL))
def test_1mb_command_is_handled_fast(name, monkeypatch):
    import time

    monkeypatch.setenv(hooks.CAPTURE_ENV, "1")
    text = PATHOLOGICAL[name]
    assert len(text) >= 1_000_000
    t0 = time.perf_counter()
    fields = hooks.command_fields(text)
    hooks.code_hash_for_command(text)
    assert time.perf_counter() - t0 < 0.1
    assert len(fields["command_preview"]) <= hooks.PREVIEW_CHARS + 4
