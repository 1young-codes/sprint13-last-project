"""End-to-end self-check: python .agent-git/test_flow.py  (builds throwaway repos in a temp folder)."""
import json
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

KIT = Path(__file__).resolve().parent
sys.path.insert(0, str(KIT))
import flow  # noqa: E402

HOOK = 'git -c "alias.agent-git-hook=!sh .agent-git/run.sh" agent-git-hook hook {host} {event}'


def sh(cmd, cwd, stdin=None, check=True):
    p = subprocess.run(cmd, cwd=cwd, shell=isinstance(cmd, str), input=stdin, capture_output=True,
                       text=True, encoding="utf-8")
    if check and p.returncode:
        raise AssertionError("%s\n%s%s" % (cmd, p.stdout, p.stderr))
    return p


def hook(cwd, host, event, payload):
    out = sh(HOOK.format(host=host, event=event), cwd, json.dumps(payload, ensure_ascii=False)).stdout
    return json.loads(out) if out.strip() else None


def unit_checks():
    good = ("## 한 줄 요약\n제품 사진의 조명을 바꾸는 기능을 추가했습니다.\n\n"
            "## 생성 조건과 결과\n| 항목 | 값 |\n| --- | --- |\n| 모델 | `IC-Light` |\n\n"
            "## 바뀐 파일\n\n<details><summary>1개</summary>\n\n- `app/relight.py`\n\n</details>")
    assert flow.check_text("[기능/이미지 생성] 조명 프리셋 추가", good) == []
    assert any("English" in p for p in flow.check_text("[기능] 조명 추가", "## 한 줄 요약\n이 PR은 UI를 바꿉니다."))
    assert any("two backtick" in p for p in flow.check_text("[기능] 조명", "## 요약\n`a` `b` `c` 를 바꿨습니다."))
    assert any("first mention" in p for p in flow.check_text("[기능] 조명", "## 요약\n`a` 수정.\n\n## 더\n`a` 다시."))
    assert any("placeholders" in p for p in flow.check_text("[기능] 조명", "## 요약\n{{why}}"))
    assert any("Empty section" in p for p in flow.check_text("[기능] 조명", "## 요약\n\n## 남은 일\n- 없음"))
    assert any("Title" in p for p in flow.check_text("feature: 조명", "## 요약\n내용입니다."))
    assert flow.check_text("[통합] 3주차 반영", "## 한 줄 요약\n반영합니다.", release=True) == []

    dev, work = {"kind": "protected", "branch": "dev"}, {"kind": "work", "branch": "feature/dev-tools"}
    assert "Committing" in flow.check_shell('git commit -m "feature: 추가"', dev)
    assert "type" in flow.check_shell('git commit -m "조명 추가"', work)
    assert flow.check_shell('git add . && git commit -m "fix: 업로드 오류 수정"', work) is None
    assert flow.check_shell("git push -u origin feature/dev-tools", work) is None
    assert flow.check_shell("git push origin main", work)
    assert flow.check_shell("git push --force", work)
    assert flow.check_shell("git push", dev)
    assert flow.check_shell("gh pr create --base dev", work)
    assert flow.check_shell("git reset --hard HEAD~1", work)
    assert flow.check_shell("git agent start feature light-preset", dev) is None


def end_to_end(tmp):
    origin, a = tmp / "origin.git", tmp / "A"
    sh(["git", "init", "-q", "--bare", "-b", "main", str(origin)], tmp)
    sh(["git", "clone", "-q", str(origin), str(a)], tmp)
    for k, v in (("user.name", "tester"), ("user.email", "t@example.com")):
        sh(["git", "config", k, v], a)
    (a / ".claude").mkdir()
    (a / ".claude/settings.json").write_text('{"permissions": {"deny": ["Bash(rm -rf*)"]}}', encoding="utf-8")
    (a / "app.py").write_text("print('hi')\n", encoding="utf-8")
    sh("git add . && git commit -q -m init && git push -q origin main", a)

    shutil.copytree(KIT, a / ".agent-git", ignore=shutil.ignore_patterns("__pycache__", "local", "test_flow.py"))
    out = sh([sys.executable, ".agent-git/flow.py", "install"], a).stdout
    assert "dev" in out and sh("git ls-remote --heads origin dev", a).stdout.strip(), out
    settings = json.loads((a / ".claude/settings.json").read_text(encoding="utf-8"))
    assert settings["permissions"]["deny"] == ["Bash(rm -rf*)"] and len(settings["hooks"]) == 4
    sh([sys.executable, ".agent-git/flow.py", "install"], a)        # idempotent
    settings = json.loads((a / ".claude/settings.json").read_text(encoding="utf-8"))
    assert all(len(groups) == 1 for groups in settings["hooks"].values())
    assert "hooks = true" in (a / ".codex/config.toml").read_text(encoding="utf-8")
    assert "@.agent-git/RULES.md" in (a / "CLAUDE.md").read_text(encoding="utf-8")
    assert sh("git branch --show-current", a).stdout.strip() == "chore/agent-git-setup"
    sh('git add -A && git commit -q -m "chore: 에이전트 git 규칙 패키지 추가" && git push -q -u origin HEAD', a)
    sh("git switch -q dev && git merge -q --ff-only chore/agent-git-setup && git push -q origin dev", a)

    # An Orca-style worktree: Orca's own branch, cut from a dev that has since moved on.
    wt = tmp / "wt"
    sh("git fetch -q origin", a)
    sh(["git", "worktree", "add", "-q", "-b", "orca/abc", str(wt), "origin/dev"], a)
    (a / "later.txt").write_text("teammate\n", encoding="utf-8")
    sh('git add later.txt && git commit -q -m "docs: 팀원 변경" && git push -q origin dev', a)
    session = hook(wt, "codex", "session", {})
    text = session["hookSpecificOutput"]["additionalContext"]
    assert "Git workflow rules" in text and "unstarted" in text
    edit = {"tool_name": "Write", "tool_input": {"file_path": str(wt / "relight.py"), "content": "x"}}
    assert hook(wt, "claude", "pre", edit)["hookSpecificOutput"]["permissionDecision"] == "deny"
    outside = {"tool_name": "Write", "tool_input": {"file_path": str(tmp / "note.txt"), "content": "x"}}
    assert hook(wt, "claude", "pre", outside) is None
    patch = {"tool_name": "apply_patch", "tool_input": {"input": "*** Begin Patch\n*** Add File: relight.py\n+x\n*** End Patch"}}
    assert hook(wt, "codex", "pre", patch)["hookSpecificOutput"]["permissionDecision"] == "deny"

    sh("git agent start feature light-preset", wt)
    assert sh("git branch --show-current", wt).stdout.strip() == "feature/light-preset"
    assert sh("git rev-parse HEAD", wt).stdout == sh("git rev-parse origin/dev", wt).stdout
    assert hook(wt, "claude", "pre", edit) is None
    sh("git agent start fix light-preset", wt)                       # not pushed yet: rename allowed
    assert sh("git branch --show-current", wt).stdout.strip() == "fix/light-preset"
    sh("git agent start feature light-preset", wt)

    (wt / "relight.py").write_text("PRESETS = 4\n", encoding="utf-8")
    sh('git add relight.py && git commit -q -m "feature: 조명 프리셋 4종 추가"', wt)
    post = hook(wt, "codex", "post", {"tool_name": "shell_command",
                                      "tool_input": {"command": 'git commit -m "feature: 조명 프리셋 4종 추가"'}})
    assert "Pushed" in post["hookSpecificOutput"]["additionalContext"], post
    assert sh("git ls-remote --heads origin feature/light-preset", wt).stdout.strip()

    waiting = hook(wt, "claude", "stop", {"last_assistant_message": "요약입니다.\n\nPR을 올릴까요?\n\n1. PR 올리기\n2. 아니오"})
    assert waiting is None
    first = hook(wt, "claude", "stop", {"stop_hook_active": False})
    assert first["decision"] == "block" and "PR을 올릴까요?" in first["reason"]
    assert hook(wt, "claude", "stop", {}) is None                   # same state: asked once

    info = json.loads(sh("git agent classify", wt).stdout)
    assert info["branch_type"] == "feature" and info["suggested_type"] == "code" and info["agrees"]
    draft = sh("git agent template --areas image,eval --jira AD-12", wt).stdout
    assert "[기능/이미지 생성]" in draft and "## 생성 조건과 결과" in draft and "## 평가 결과" in draft
    assert "`AD-12`" in draft and "## 바뀐 파일" in draft and "## 왜 필요한가" in draft
    assert "- `relight.py`" in draft
    (wt / "relight.py").write_text("PRESETS = 5\n", encoding="utf-8")    # unstaged: listed with its full name
    assert "- `relight.py`" in sh("git agent template --areas image", wt).stdout
    sh("git checkout -- relight.py", wt)
    sh("git agent template --areas image,eval --jira AD-12", wt)
    refused = sh("git agent pr", wt, check=False)
    assert refused.returncode == 1 and "placeholders" in refused.stderr, refused.stderr


def main():
    unit_checks()
    tmp = Path(tempfile.mkdtemp(prefix="agent-git-test-"))
    try:
        end_to_end(tmp)
    finally:
        shutil.rmtree(tmp, ignore_errors=True)
    print("agent-git self-check passed")


if __name__ == "__main__":
    main()
