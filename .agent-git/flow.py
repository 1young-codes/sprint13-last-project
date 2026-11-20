"""agent-git: one branch per task, Korean PRs built from typed templates, for Claude Code and Codex.

Standard library only, Python 3.8+ (macOS ships 3.9). Every entry point is reached through
`run.sh`, which git starts from the repository root on Windows, macOS and Linux alike.

  hooks   : flow.py hook <claude|codex> <session|pre|post|stop>   (stdin: the host's JSON)
  agent   : flow.py start | classify | template | pr               (via `git agent ...`)
  person  : flow.py install | doctor
"""
from __future__ import annotations

import argparse
import datetime
import fnmatch
import hashlib
import json
import os
import re
import shutil
import subprocess
import sys
from collections import Counter
from pathlib import Path

KIT = Path(__file__).resolve().parent
CONFIG = json.loads((KIT / "config.json").read_text(encoding="utf-8"))
TYPES = CONFIG["types"]
AREAS = CONFIG["areas"]
REMOTE, BASE, RELEASE = CONFIG["remote"], CONFIG["base"], CONFIG["release"]
PROTECTED = set(CONFIG["protected"])
UPSTREAM = "%s/%s" % (REMOTE, BASE)
WORK = re.compile(r"^(%s)/([a-z0-9]+(?:-[a-z0-9]+)*)$" % "|".join(TYPES))
SLUG = re.compile(r"^[a-z0-9]+(?:-[a-z0-9]+)*$")
COMMIT_MSG = re.compile(r"^(%s): \S" % "|".join(TYPES))
LOCAL = ".agent-git/local/"
ALIAS = "!sh .agent-git/run.sh"
DECISION_TYPES = {"feature", "fix", "refactor"}

SHELL_TOOLS = {"Bash", "PowerShell", "shell", "shell_command", "exec_command", "local_shell", "container.exec"}
EDIT_TOOLS = {"Edit", "Write", "MultiEdit", "NotebookEdit", "apply_patch"}


class Stop(Exception):
    """A message for whoever called; exit code 1."""


# -- git ------------------------------------------------------------------------------------------

def run(args, check=True, timeout=60, errors=False):
    try:
        p = subprocess.run(args, capture_output=True, text=True, encoding="utf-8", errors="replace", timeout=timeout)
    except (OSError, subprocess.TimeoutExpired) as error:
        if check:
            raise Stop("%s failed: %s" % (args[0], error))
        return 1, ""
    if check and p.returncode:
        raise Stop((p.stderr or p.stdout).strip() or "%s exited %d" % (" ".join(args[:3]), p.returncode))
    out = p.stdout + (p.stderr if errors and p.returncode else "")
    return p.returncode, out.rstrip()   # rstrip only: `status --porcelain` starts with a meaningful space


def git(*args, check=True, timeout=60):
    return run(["git", *args], check=check, timeout=timeout)[1]


def git_ok(*args):
    return run(["git", *args], check=False)[0] == 0


def root():
    return Path(git("rev-parse", "--show-toplevel"))


def git_dir():
    return Path(git("rev-parse", "--absolute-git-dir"))


def main_checkout():
    """The first worktree: where personal files survive an Orca worktree being deleted."""
    return Path(git("rev-parse", "--path-format=absolute", "--git-common-dir")).parent


def branch():
    return git("symbolic-ref", "--short", "-q", "HEAD", check=False)


def ahead(ref="HEAD"):
    if not git_ok("rev-parse", "--verify", "-q", UPSTREAM):
        return None
    return int(git("rev-list", "--count", "%s..%s" % (UPSTREAM, ref)) or 0)


def state():
    """kind: protected (main/dev) | unstarted (detached, or Orca's own branch with no commits yet)
    | work (type/slug) | other (a custom branch that already holds commits: left alone)."""
    name = branch()
    if not name:
        return {"kind": "unstarted", "branch": ""}
    if name in PROTECTED:
        return {"kind": "protected", "branch": name}
    if WORK.match(name):
        return {"kind": "work", "branch": name}
    return {"kind": "unstarted" if not ahead() else "other", "branch": name}


def dirty():
    lines = git("status", "--porcelain", "--untracked-files=all").splitlines()
    return [l for l in lines if not l[3:].strip('"').startswith(LOCAL)]


def changed_files():
    files = []
    if git_ok("rev-parse", "--verify", "-q", UPSTREAM):
        files = git("diff", "--name-only", "%s...HEAD" % UPSTREAM).splitlines()
    files += [l[3:].strip('"').split(" -> ")[-1] for l in dirty()]
    return sorted(set(f for f in files if f))


def gh(*args, check=True, timeout=60):
    return run(["gh", *args], check=check, timeout=timeout)[1]


def gh_ready():
    if not shutil.which("gh"):
        return "missing"
    return "ok" if run(["gh", "auth", "status"], check=False, timeout=20)[0] == 0 else "logged-out"


GH_HELP = (
    "GitHub CLI (gh) is {state}. Tell the user, in Korean, to do this once and then ask you again:\n"
    "  1. Install: Windows `winget install --id GitHub.cli` / macOS `brew install gh`\n"
    "  2. Close and reopen the Orca terminal, then run `gh auth login` and choose GitHub.com > HTTPS > browser.\n"
    "In Claude Code they can type `! gh auth login` in the prompt instead."
)


# -- classification -------------------------------------------------------------------------------

def match(path, patterns):
    return any(fnmatch.fnmatch(path, p) for p in patterns)


def classify():
    files = [f for f in changed_files() if not f.startswith(CONFIG["images_dir"] + "/")]
    areas, unmatched = [], []
    for f in files:
        hit = [key for key, area in AREAS.items() if match(f, area.get("paths") or [])]
        areas += [a for a in hit if a not in areas]
        if not hit:
            unmatched.append(f)
    suggested = None
    if files:
        committed = git_ok("rev-parse", "--verify", "-q", UPSTREAM) and not dirty()
        if committed and git_ok("diff", "--quiet", "-w", "--ignore-blank-lines", "%s...HEAD" % UPSTREAM):
            suggested = "style"
        else:
            for kind in ("test", "chore", "docs"):
                if all(match(f, CONFIG["type_paths"][kind]) for f in files):
                    suggested = kind
                    break
            else:
                suggested = "code"
    st = state()
    current = WORK.match(st["branch"]).group(1) if st["kind"] == "work" else None
    agrees = suggested in (None, current) or (suggested == "code" and current in DECISION_TYPES)
    return {"branch": st["branch"], "branch_type": current, "suggested_type": suggested, "agrees": agrees,
            "areas": areas, "files_without_area": unmatched, "files": files}


# -- hooks ----------------------------------------------------------------------------------------

START_HINT = ("Classify the task first (type x area, see .agent-git/RULES.md), then run "
              "`git agent start <type> <slug>`; it branches from the latest origin/%s." % BASE)


def deny(reason):
    return {"hookSpecificOutput": {"hookEventName": "PreToolUse", "permissionDecision": "deny",
                                   "permissionDecisionReason": "[agent-git] " + reason}}


def context(event, text):
    return {"hookSpecificOutput": {"hookEventName": event, "additionalContext": text}}


def shell_command(payload):
    given = payload.get("tool_input") or {}
    command = given.get("command", given.get("cmd", "")) if isinstance(given, dict) else ""
    return " ".join(map(str, command)) if isinstance(command, list) else str(command)


def edit_targets(payload):
    given = payload.get("tool_input") or {}
    if not isinstance(given, dict):
        return []
    if str(payload.get("tool_name", "")).endswith("apply_patch"):
        patch = str(given.get("input") or given.get("patch") or given.get("command") or "")
        return re.findall(r"^\*\*\* (?:Add|Update|Delete) File: (.+)$", patch, re.M)
    target = given.get("file_path") or given.get("notebook_path")
    return [target] if target else []


def inside(path, top):
    full = Path(path) if os.path.isabs(path) else top / path
    try:
        full.resolve().relative_to(top.resolve())
        return True
    except ValueError:
        return False


def git_words(segment):
    """('git', subcommand, args) for one shell segment, skipping `-C dir` / `-c k=v` before the subcommand."""
    words = [w.strip("\"'") for w in segment.split()]
    for i, w in enumerate(words):
        if re.fullmatch(r"(?:.*[/\\])?git(?:\.exe)?", w):
            j = i + 1
            while j < len(words) and words[j].startswith("-"):
                j += 2 if words[j] in ("-C", "-c") else 1
            if j < len(words):
                return words[j], words[j + 1:]
    return None, []


def commit_message(command):
    m = re.search(r"(?:^|\s)(?:-[a-zA-Z]*m|--message)(?:=|\s*)([\"'])(.*?)\1", command, re.S)
    return m.group(2) if m else None


def check_shell(command, st):
    blocked = st["kind"] in ("protected", "unstarted")
    for segment in re.split(r"&&|\|\||[;\n|]", command):
        words = [w.strip("\"'") for w in segment.split()]
        if words[:3] == ["gh", "pr", "create"]:
            return "Do not run `gh pr create` directly. Follow the PR procedure: `git agent template`, fill the draft, `git agent pr`."
        if words[:3] == ["gh", "pr", "merge"]:
            return "Merging is a reviewer's job on GitHub, not the agent's."
        sub, args = git_words(segment)
        if sub == "commit":
            if blocked:
                return "Committing on '%s' is not allowed. %s" % (st["branch"] or "detached HEAD", START_HINT)
        elif sub == "push":
            if any(a in ("-f", "--force", "--force-with-lease", "--mirror") or a.startswith("+") for a in args):
                return "Force push is not allowed."
            refs = [a for a in args if not a.startswith("-")][1:]
            targets = {r.split(":")[-1].replace("refs/heads/", "") for r in refs}
            if targets & PROTECTED or (not refs and st["kind"] == "protected"):
                return "Pushing to %s is not allowed; changes reach them only through PRs." % "/".join(sorted(PROTECTED))
        elif sub == "reset" and "--hard" in args:
            return "`git reset --hard` is not allowed: it destroys uncommitted work."
        elif sub in ("merge", "cherry-pick", "revert", "am", "rebase") and st["kind"] == "protected":
            return "No `git %s` on '%s'. %s" % (sub, st["branch"], START_HINT)
        elif sub == "branch" and any(a in ("-D", "-d", "-m", "-M", "--delete", "--move") for a in args) \
                and set(args) & PROTECTED:
            return "Deleting or renaming %s is not allowed." % "/".join(sorted(PROTECTED))
    if re.search(r"\bgit\b.*\bcommit\b", command):
        message = commit_message(command)
        if message is not None and not message.startswith("$(") and not COMMIT_MSG.match(message):
            return ("Commit message must be `<type>: <Korean description>` with type one of %s, e.g. "
                    "`feature: 조명 보정 화면 추가`." % ", ".join(TYPES))
    return None


def hook_session(host, payload):
    if git("config", "--get", "alias.agent", check=False) != ALIAS:
        git("config", "alias.agent", ALIAS, check=False)
    git("fetch", "--quiet", REMOTE, BASE, check=False, timeout=15)
    st = state()
    lines = ["[agent-git] Current branch: %s (%s)." % (st["branch"] or "detached HEAD", st["kind"])]
    if st["kind"] in ("protected", "unstarted"):
        lines.append("File edits and commits are blocked here until a work branch exists. " + START_HINT)
    elif st["kind"] == "work":
        lines.append("Work branch is active. Commits are pushed automatically.")
    text = "\n".join(lines)
    if host == "codex":   # Codex has no import for AGENTS.md, so the rules arrive here.
        text = (KIT / "RULES.md").read_text(encoding="utf-8") + "\n\n" + text
    return context("SessionStart", text)


def hook_pre(host, payload):
    tool = str(payload.get("tool_name") or "").replace("functions.", "")
    st = state()
    if tool in EDIT_TOOLS:
        top = root()
        if st["kind"] in ("protected", "unstarted") and any(inside(p, top) for p in edit_targets(payload)):
            return deny("No file edits on '%s'. %s" % (st["branch"] or "detached HEAD", START_HINT))
        return None
    if tool in SHELL_TOOLS:
        reason = check_shell(shell_command(payload), st)
        return deny(reason) if reason else None
    return None


def push(name):
    upstream = git("rev-parse", "--abbrev-ref", "%s@{upstream}" % name, check=False)
    if upstream and git_ok("rev-parse", "--verify", "-q", upstream) \
            and git("rev-list", "--count", "%s..%s" % (upstream, name)) == "0":
        return None
    code, _ = run(["git", "push", "-u", REMOTE, name], check=False, timeout=90)
    return ("Pushed %s to %s." % (name, REMOTE)) if code == 0 else (
        "Automatic push of %s failed. Tell the user in Korean and suggest `git push -u %s %s`." % (name, REMOTE, name))


def hook_post(host, payload):
    tool = str(payload.get("tool_name") or "").replace("functions.", "")
    if tool not in SHELL_TOOLS:
        return None
    if not re.search(r"\bgit\b.*\b(commit|merge|cherry-pick|revert)\b", shell_command(payload)):
        return None
    st = state()
    if st["kind"] not in ("work", "other"):
        return None
    note = push(st["branch"])
    return context("PostToolUse", "[agent-git] " + note) if note else None


def open_pr(name):
    if gh_ready() != "ok":
        return None
    code, out = run(["gh", "pr", "view", name, "--json", "url,state"], check=False, timeout=30)
    if code:
        return None
    data = json.loads(out)
    return data["url"] if data.get("state") == "OPEN" else None


ASK = """[agent-git] The work on {branch} looks finished and has no PR yet. Before you stop:
1. {commit}
2. Run `git agent classify` and check the type still matches the work (rules: .agent-git/RULES.md, "Classify").
3. Tell the user in plain Korean what you did: 3-5 short bullets, no jargon.
4. Ask exactly one question: "PR을 올릴까요?" with two choices, "PR 올리기" and "아니오".
   Claude: use AskUserQuestion. Codex: write the two numbered choices and end your turn to wait.
5. On "PR 올리기", follow the PR procedure in .agent-git/RULES.md. On "아니오", stop; you will not be asked again for this state."""

COMMIT_FIRST = ("Commit the remaining changes in meaningful units as `<type>: <Korean description>` "
                "(each commit is pushed automatically).")


def hook_stop(host, payload):
    if payload.get("stop_hook_active"):
        return None
    st = state()
    if st["kind"] not in ("work", "other"):
        return None
    pending, commits = dirty(), ahead() or 0
    if not pending and not commits:
        return None
    # A question in the closing lines (choices may follow it) means the agent is already waiting on the user.
    tail = str(payload.get("last_assistant_message") or "").strip().splitlines()[-6:]
    if any(line.rstrip().endswith(("?", "？")) for line in tail):
        return None
    key = hashlib.sha1((git("rev-parse", "HEAD") + "\n".join(pending)).encode("utf-8")).hexdigest()
    memo = git_dir() / "agent-git-asked"
    if memo.is_file() and memo.read_text(encoding="utf-8").strip() == key:
        return None
    memo.write_text(key, encoding="utf-8")
    url = open_pr(st["branch"])
    if url:
        if not pending:
            return None
        reason = "[agent-git] %s Then tell the user in Korean that the open PR %s now includes it." % (COMMIT_FIRST, url)
    else:
        reason = ASK.format(branch=st["branch"], commit=COMMIT_FIRST if pending else "Nothing left to commit.")
    return {"decision": "block", "reason": reason}


HOOKS = {"session": hook_session, "pre": hook_pre, "post": hook_post, "stop": hook_stop}


def hook(host, event):
    try:
        sys.stdin.reconfigure(encoding="utf-8")
        sys.stdout.reconfigure(encoding="utf-8")
        raw = sys.stdin.read()
        payload = json.loads(raw) if raw.strip() else {}
        answer = HOOKS[event](host, payload if isinstance(payload, dict) else {})
        if answer:
            json.dump(answer, sys.stdout, ensure_ascii=False)
    except Exception as error:  # a broken hook must never stop the work: fail open
        print("agent-git hook %s: %s: %s" % (event, type(error).__name__, error), file=sys.stderr)


# -- agent commands -------------------------------------------------------------------------------

def cmd_start(kind, slug):
    if kind not in TYPES:
        raise Stop("Unknown type '%s'. Use one of: %s" % (kind, ", ".join(TYPES)))
    if not SLUG.match(slug) or len(slug) > CONFIG["slug_max"]:
        raise Stop("Slug must be lowercase English words joined by '-', at most %d characters (e.g. light-preset)."
                   % CONFIG["slug_max"])
    target = "%s/%s" % (kind, slug)
    if not git_ok("fetch", "--quiet", REMOTE, BASE):
        raise Stop("Could not fetch %s. Check the network and that the remote has a '%s' branch." % (UPSTREAM, BASE))
    if git_ok("rev-parse", "--verify", "-q", "refs/heads/" + target) or \
            git_ok("rev-parse", "--verify", "-q", "refs/remotes/%s/%s" % (REMOTE, target)):
        raise Stop("Branch %s already exists. Pick another slug." % target)
    st = state()
    if st["kind"] == "work":
        if git_ok("ls-remote", "--exit-code", "--heads", REMOTE, st["branch"]):
            print("%s is already pushed, so its name stays. Use `--type %s` when building the PR template." % (st["branch"], kind))
            return
        git("branch", "-m", target)
        print("Renamed %s -> %s (not pushed yet)." % (st["branch"], target))
        return
    if st["kind"] == "other":
        raise Stop("%s already holds its own commits. Ask the user before moving that work." % st["branch"])
    if st["kind"] == "unstarted" and st["branch"]:
        # Orca made this worktree with its own branch: rename it and move it onto the latest base.
        git("branch", "-m", target)
        git("branch", "--unset-upstream", check=False)
        if not git_ok("reset", "--keep", UPSTREAM):
            raise Stop("Renamed to %s, but uncommitted changes conflict with %s. Ask the user how to proceed."
                       % (target, UPSTREAM))
    else:
        git("switch", "--no-track", "-c", target, UPSTREAM)
    print("Now on %s, based on the latest %s. Commits are pushed automatically." % (target, UPSTREAM))


def read_template(*parts):
    return (KIT / "templates" / Path(*parts)).read_text(encoding="utf-8").strip()


def repo_slug():
    return gh("repo", "view", "--json", "nameWithOwner", "-q", ".nameWithOwner", check=False) if shutil.which("gh") else ""


def image_block(slug):
    folder = "%s/%s" % (CONFIG["images_dir"], slug)
    tracked = git("ls-files", folder, check=False).splitlines()
    if not tracked:
        return ("{{before/after images: save them as %s/before-*.png and after-*.png, commit, "
                "then run `git agent template` again}}" % folder)
    repo, sha = repo_slug(), git("rev-parse", "HEAD")
    url = lambda p: "https://github.com/%s/blob/%s/%s?raw=true" % (repo, sha, p)  # noqa: E731
    befores = [p for p in tracked if Path(p).name.startswith("before")]
    afters = [p for p in tracked if Path(p).name.startswith("after")]
    rows = ["| 전 | 후 |", "| --- | --- |"]
    for b, a in zip(befores, afters):
        rows.append("| ![전](%s) | ![후](%s) |" % (url(b), url(a)))
    rest = [p for p in tracked if p not in befores[:len(afters)] + afters[:len(befores)]]
    rows += ["", *("![결과](%s)" % url(p) for p in rest)]
    return "\n".join(rows).strip()


def files_block(files):
    items = "\n".join("- `%s`" % f for f in files)
    return "## 바뀐 파일\n\n<details><summary>%d개</summary>\n\n%s\n\n</details>" % (len(files), items)


def draft_path():
    return root() / LOCAL / "pr-draft.md"


def cmd_template(kind=None, areas=None, jira=None, release=False):
    if release:
        title = "[%s] {{title}}" % CONFIG["release_label"]
        body = read_template("release.md").replace("{{included}}", release_list())
        meta = "release"
    else:
        info = classify()
        kind = kind or info["branch_type"]
        if kind not in TYPES:
            raise Stop("Pass --type: this branch has no type prefix. Types: %s" % ", ".join(TYPES))
        areas = areas or info["areas"]
        bad = [a for a in areas if a not in AREAS]
        if bad:
            raise Stop("Unknown area %s. Areas: %s" % (bad, ", ".join(AREAS)))
        label = TYPES[kind]["label"] + ("/" + AREAS[areas[0]]["label"] if areas else "")
        title = "[%s] {{title}}" % label
        slug = (WORK.match(info["branch"]) or re.match(r"(.*)", info["branch"])).groups()[-1]
        sections = [read_template("head.md"), read_template("types", kind + ".md")]
        sections += [read_template("areas", a + ".md").replace("{{images}}", image_block(slug)) for a in areas]
        tail = read_template("tail.md")
        if jira:
            tail = tail.replace("{{Jira key}}", "`%s`" % jira)
        sections += [tail, files_block(info["files"])]
        body = "\n\n".join(sections)
        meta = "type=%s areas=%s" % (kind, ",".join(areas))
        if not info["agrees"]:
            print("NOTE: the diff looks like '%s' but the branch says '%s'. Tell the user before opening the PR."
                  % (info["suggested_type"], info["branch_type"]), file=sys.stderr)
    path = draft_path()
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("<!-- agent-git: %s -->\n%s\n\n%s\n" % (meta, title, body), encoding="utf-8")
    print("Draft written to %s. Replace every {{...}}, delete optional sections you do not need, "
          "then run `git agent pr%s`.\n" % (path.relative_to(root()).as_posix(), " --release" if release else ""))
    print(path.read_text(encoding="utf-8"))


def release_list():
    if gh_ready() != "ok":
        raise Stop(GH_HELP.format(state=gh_ready()))
    git("fetch", "--quiet", REMOTE, BASE, RELEASE)
    shas = set(git("rev-list", "%s/%s..%s" % (REMOTE, RELEASE, UPSTREAM)).splitlines())
    prs = json.loads(gh("pr", "list", "--base", BASE, "--state", "merged", "--limit", "200",
                        "--json", "number,title,mergeCommit"))
    groups = {}
    for pr in prs:
        if (pr.get("mergeCommit") or {}).get("oid") not in shas:
            continue
        m = re.match(r"^\[([^/\]]+)(?:/[^\]]*)?\]\s*(.*)$", pr["title"])
        label, text = (m.group(1), m.group(2)) if m else ("기타", pr["title"])
        groups.setdefault(label, []).append("- #%d %s" % (pr["number"], text))
    order = [t["label"] for t in TYPES.values()] + ["기타"]
    out = ["### %s\n%s" % (label, "\n".join(groups[label])) for label in order if label in groups]
    return "\n\n".join(out) or "{{no merged PRs found since the last release; say so in Korean}}"


# -- Korean writing check -------------------------------------------------------------------------

def strip_exempt(body):
    body = re.sub(r"<!--.*?-->", "", body, flags=re.S)
    body = re.sub(r"^## 바뀐 파일.*?(?=^## |\Z)", "", body, flags=re.S | re.M)
    body = re.sub(r"^```.*?^```", "", body, flags=re.S | re.M)
    body = re.sub(r"\]\([^)]*\)", "]", body)
    body = re.sub(r"https?://\S+", "", body)
    return re.sub(r"</?[a-zA-Z][^>]*>", "", body)


def check_text(title, body, release=False):
    problems = []
    labels = "|".join(re.escape(t["label"]) for t in TYPES.values())
    areas = "|".join(re.escape(a["label"]) for a in AREAS.values())
    pattern = r"^\[%s\] \S" % re.escape(CONFIG["release_label"]) if release else r"^\[(%s)(/(%s))?\] \S" % (labels, areas)
    if not re.match(pattern, title):
        problems.append("Title must look like `[기능/화면] 설명` (type label, optional area label, then Korean text).")
    whole = title + "\n" + body
    if "{{" in whole:
        problems.append("Unfilled placeholders remain: " + ", ".join(sorted(set(re.findall(r"\{\{.*?\}\}", whole)))[:5]))
    for heading in re.findall(r"^(## .+)\n\s*(?=^## |\Z)", re.sub(r"<!--.*?-->", "", body, flags=re.S), re.M):
        problems.append("Empty section %s: fill it or delete it." % heading.strip())
    text = strip_exempt(title + "\n" + body)
    spans = []
    for line in text.splitlines():
        for sentence in re.split(r"(?<=[.!?。])\s+", line):
            found = re.findall(r"`([^`]+)`", sentence)
            spans += found
            if len(found) > 2:
                problems.append("More than two backtick names in one sentence: %r. Move names to a table or "
                                "refer to them in Korean." % sentence.strip()[:80])
            english = re.findall(r"[A-Za-z]{2,}", re.sub(r"`[^`]+`", "", sentence))
            if english:
                problems.append("English outside backticks: %s in %r. Write it in Korean, or backtick a real name once."
                                % (", ".join(sorted(set(english))), sentence.strip()[:80]))
    repeated = [name for name, n in Counter(spans).items() if n > 1]
    if repeated:
        problems.append("Backtick each name only at its first mention, then refer to it in Korean "
                        "(e.g. '이 서버', '조명 모델'): %s" % ", ".join(repeated))
    return problems


# -- PR -------------------------------------------------------------------------------------------

def parse_draft(path):
    raw = path.read_text(encoding="utf-8")
    meta = dict(re.findall(r"(\w+)=([\w,-]*)", (re.search(r"<!-- agent-git: (.*?) -->", raw) or [None, ""])[1] or ""))
    raw = re.sub(r"<!-- agent-git: .*? -->\n?", "", raw)
    lines = raw.strip().splitlines()
    title, body = lines[0].strip(), "\n".join(lines[1:]).strip()
    body = re.sub(r"\n{3,}", "\n\n", re.sub(r"<!--.*?-->\n?", "", body, flags=re.S)).strip()
    return meta, title, body


def ensure_label(name, color):
    run(["gh", "label", "create", name, "--color", color, "--force"], check=False, timeout=30)


def sections(body):
    parts = re.split(r"^## (.+)$", body, flags=re.M)
    return {parts[i].strip(): parts[i + 1].strip() for i in range(1, len(parts) - 1, 2)}


def write_journal(title, body, url, name):
    s = sections(body)
    today = datetime.date.today()
    day = "월화수목금토일"[today.weekday()]
    pick = lambda *names: next((s[n] for n in names if s.get(n)), "")  # noqa: E731
    text = "\n".join([
        "날짜: %s (%s)" % (today.strftime("%Y.%m.%d"), day),
        "", "오늘 목표(Today's Goal) :", pick("한 줄 요약"),
        "", "오늘 한 일(What I Did) :", pick("무엇이 바뀌었나", "고친 방법", "실행 결과"),
        "", "작업 내용 및 완료도(Details) : (직접 작성) %",
        "", "문제 상황 (Problem) :", pick("증상", "왜 바꾸었나"),
        "", "해결 방법 (Solution) :", pick("고친 방법", "고른 방법과 이유"),
        "", "배운 점 (What I Learned) : (직접 작성)",
        "", "내일 할 일 (Tomorrow) :", pick("남은 일"),
        "", "도움이 필요한 부분 : (직접 작성)",
        "", "참고 자료 (Reference) :", "- %s %s" % (title, url),
        "", "블로그 / 포트폴리오 메모 :", pick("고른 방법과 이유", "왜 필요한가"), "",
    ])
    folder = main_checkout() / LOCAL / "journal"
    folder.mkdir(parents=True, exist_ok=True)
    path = folder / ("%s_%s.md" % (today.isoformat(), name.replace("/", "-")))
    path.write_text(text, encoding="utf-8")
    return path


def cmd_pr(release=False):
    path = draft_path()
    if not path.is_file():
        raise Stop("No draft. Run `git agent template%s` first." % (" --release" if release else ""))
    meta, title, body = parse_draft(path)
    problems = check_text(title, body, release)
    if problems:
        raise Stop("The draft needs fixes before it can be opened (edit %s, then run this again):\n- %s"
                   % (path.relative_to(root()).as_posix(), "\n- ".join(problems)))
    ready = gh_ready()
    if ready != "ok":
        raise Stop(GH_HELP.format(state=ready))
    body_file = git_dir() / "agent-git-pr-body.md"
    body_file.write_text(body + "\n", encoding="utf-8")
    me = gh("api", "user", "-q", ".login", check=False)
    if release:
        base, head, labels, reviewers = RELEASE, BASE, [(CONFIG["release_label"], "0e8a16")], []
    else:
        st = state()
        if st["kind"] not in ("work", "other"):
            raise Stop("Not on a work branch. " + START_HINT)
        if dirty():
            raise Stop("Uncommitted changes remain. Commit them first (`<type>: <Korean description>`).")
        note = push(st["branch"])
        if note and note.startswith("Automatic push"):
            raise Stop(note)
        if open_pr(st["branch"]):
            raise Stop("A PR for %s is already open: %s" % (st["branch"], open_pr(st["branch"])))
        kind = meta.get("type") or WORK.match(st["branch"]).group(1)
        areas = [a for a in (meta.get("areas") or "").split(",") if a in AREAS]
        base, head = BASE, st["branch"]
        labels = [(TYPES[kind]["label"], TYPES[kind]["color"])] + [(AREAS[a]["label"], "c5def5") for a in areas]
        reviewers = sorted({r for a in areas for r in AREAS[a].get("reviewers") or [] if r and r != me})
    for name, color in labels:
        ensure_label(name, color)
    args = ["gh", "pr", "create", "--base", base, "--head", head, "--title", title, "--body-file", str(body_file),
            "--assignee", "@me"]
    for name, _ in labels:
        args += ["--label", name]
    code, out = run(args + sum([["--reviewer", r] for r in reviewers], []), check=False, timeout=120, errors=True)
    if code and reviewers:   # an unknown reviewer id fails the whole call: retry without them
        code, out = run(args, check=False, timeout=120, errors=True)
        out += "\n(Reviewers %s could not be requested; check the ids in .agent-git/config.json.)" % ", ".join(reviewers)
    if code:
        raise Stop("gh pr create failed: %s" % out)
    url = out.strip().splitlines()[0]
    journal = write_journal(title, body, url, "release" if release else head)
    path.unlink()
    print("PR opened: %s\nJournal draft (local only): %s\nGive the user the PR link and the journal path in Korean."
          % (out.strip(), journal))


# -- install / doctor (people read these: Korean) -------------------------------------------------

MARK_START, MARK_END = "<!-- agent-git:start -->", "<!-- agent-git:end -->"
CLAUDE_BLOCK = "## Git workflow (agent-git)\n\n@.agent-git/RULES.md"
AGENTS_BLOCK = ("## Git workflow (agent-git)\n\nBefore any branch, commit or PR work, read and follow "
                "`.agent-git/RULES.md`. Its rules override defaults.")


def hook_command(host, event):
    return 'git -c "alias.agent-git-hook=%s" agent-git-hook hook %s %s' % (ALIAS, host, event)


def ours(group):
    return any(".agent-git/run.sh" in str(h.get("command", "")) for h in group.get("hooks", []))


def merge_hooks(path, host):
    data = json.loads(path.read_text(encoding="utf-8")) if path.is_file() else {}
    hooks = data.setdefault("hooks", {})
    plan = {"SessionStart": ("session", None, 30), "PreToolUse": ("pre", None, 20),
            "PostToolUse": ("post", None, 120), "Stop": ("stop", None, 60)}
    if host == "claude":
        plan["PreToolUse"] = ("pre", "Bash|PowerShell|Edit|Write|MultiEdit|NotebookEdit", 20)
        plan["PostToolUse"] = ("post", "Bash|PowerShell", 120)
    for event, (name, matcher, timeout) in plan.items():
        groups = [g for g in hooks.get(event, []) if not ours(g)]
        entry = {"type": "command", "command": hook_command(host, name), "timeout": timeout}
        if host == "codex" and name == "session":
            entry["additionalContextLimit"] = 20000
        group = {"hooks": [entry]}
        if matcher:
            group = {"matcher": matcher, **group}
        hooks[event] = groups + [group]
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def put_block(path, block):
    text = path.read_text(encoding="utf-8") if path.is_file() else ""
    full = "%s\n%s\n%s" % (MARK_START, block, MARK_END)
    if MARK_START in text:
        text = re.sub(re.escape(MARK_START) + ".*?" + re.escape(MARK_END), lambda _: full, text, flags=re.S)
    else:
        text = (text.rstrip() + "\n\n" if text.strip() else "") + full + "\n"
    path.write_text(text, encoding="utf-8")


def codex_feature(path):
    text = path.read_text(encoding="utf-8") if path.is_file() else ""
    if re.search(r"^\[features\]", text, re.M):
        section = re.search(r"^\[features\][^\[]*", text, re.M)
        if re.search(r"^\s*hooks\s*=", section.group(0), re.M):
            new = re.sub(r"^(\s*hooks\s*=\s*).*$", r"\1true", section.group(0), count=1, flags=re.M)
        else:
            new = section.group(0).rstrip("\n") + "\nhooks = true\n\n"
        text = text.replace(section.group(0), new, 1)
    else:
        text = (text.rstrip() + "\n\n" if text.strip() else "") + "[features]\nhooks = true\n"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")


def cmd_install():
    top = root()
    if top / ".agent-git" != KIT:
        raise Stop("이 폴더(.agent-git)를 저장소 맨 위 폴더에 두고 다시 실행하세요: %s" % top)
    done = []
    git("fetch", "--quiet", REMOTE, check=False, timeout=60)
    if not git_ok("rev-parse", "--verify", "-q", UPSTREAM):
        if not git_ok("rev-parse", "--verify", "-q", "%s/%s" % (REMOTE, RELEASE)):
            raise Stop("원격 저장소에 %s도 %s도 없습니다. GitHub 저장소를 먼저 연결하세요." % (BASE, RELEASE))
        git("push", REMOTE, "%s/%s:refs/heads/%s" % (REMOTE, RELEASE, BASE))
        git("fetch", "--quiet", REMOTE, BASE)
        done.append("원격에 %s 브랜치가 없어 %s에서 만들어 올렸습니다." % (BASE, RELEASE))
    if state()["kind"] in ("protected", "unstarted"):
        cmd_start("chore", "agent-git-setup")
        done.append("설치 내용을 담을 작업 브랜치 chore/agent-git-setup을 만들었습니다.")
    merge_hooks(top / ".claude/settings.json", "claude")
    merge_hooks(top / ".codex/hooks.json", "codex")
    codex_feature(top / ".codex/config.toml")
    put_block(top / "CLAUDE.md", CLAUDE_BLOCK)
    put_block(top / "AGENTS.md", AGENTS_BLOCK)
    done.append("훅: .claude/settings.json, .codex/hooks.json, .codex/config.toml (기존 내용은 그대로 두고 합쳤습니다)")
    done.append("지침: CLAUDE.md, AGENTS.md 끝에 .agent-git/RULES.md를 가리키는 블록을 넣었습니다")
    web = top / ".github/pull_request_template.md"
    if web.is_file():
        done.append("웹 PR 템플릿은 이미 있어서 건드리지 않았습니다: .github/pull_request_template.md")
    else:
        web.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(KIT / "templates/web.md", web)
        done.append("웹 PR 템플릿: .github/pull_request_template.md")
    ignore = top / ".gitignore"
    text = ignore.read_text(encoding="utf-8") if ignore.is_file() else ""
    if LOCAL not in text:
        ignore.write_text(text + ("" if text.endswith("\n") or not text else "\n") + LOCAL + "\n", encoding="utf-8")
    git("config", "alias.agent", ALIAS)
    print("설치를 마쳤습니다.\n- " + "\n- ".join(done))
    print("\n다음 순서로 마무리하세요.\n"
          "1. 바뀐 파일을 커밋합니다: chore: 에이전트 git 규칙 패키지 추가\n"
          "2. 에이전트를 새로 열고(훅은 새 세션부터 적용) 'PR 올려줘'라고 하면 dev로 PR이 올라갑니다.\n"
          "3. dev에 머지되면 팀원은 pull만 하면 됩니다.\n"
          "4. 점검: git agent doctor")


def cmd_doctor():
    top, rows = root(), []
    say = lambda ok, text: rows.append(("✅ " if ok else "❌ ") + text)  # noqa: E731
    say(sys.version_info >= (3, 8), "Python %d.%d" % sys.version_info[:2])
    say(git_ok("rev-parse", "--verify", "-q", UPSTREAM), "원격 %s 브랜치" % UPSTREAM)
    ready = gh_ready()
    say(ready == "ok", {"ok": "GitHub CLI 로그인됨", "missing": "GitHub CLI(gh)가 없습니다. 설치: Windows "
        "`winget install --id GitHub.cli`, macOS `brew install gh`",
        "logged-out": "GitHub CLI 로그인이 필요합니다: gh auth login"}[ready])
    for rel in (".claude/settings.json", ".codex/hooks.json"):
        p = top / rel
        have = p.is_file() and ".agent-git/run.sh" in p.read_text(encoding="utf-8")
        say(have, "%s에 훅 연결" % rel)
    cfg = top / ".codex/config.toml"
    say(cfg.is_file() and re.search(r"^\s*hooks\s*=\s*true", cfg.read_text(encoding="utf-8"), re.M),
        ".codex/config.toml의 hooks = true")
    head = re.search(r"refs/heads/(\S+)\s+HEAD", git("ls-remote", "--symref", REMOTE, "HEAD", check=False, timeout=30))
    say(bool(head) and head.group(1) == BASE,
        "GitHub 기본 브랜치가 %s (지금: %s). Orca는 새 워크트리를 기본 브랜치에서 만드는데, 훅은 %s에 먼저 들어가므로 "
        "기본 브랜치를 %s로 바꾸세요: gh repo edit --default-branch %s (또는 Orca에서 기준 브랜치를 %s로 고르기)"
        % (BASE, head.group(1) if head else "알 수 없음", BASE, BASE, BASE, BASE))
    st = state()
    say(True, "현재 브랜치: %s (%s)" % (st["branch"] or "분리된 HEAD", st["kind"]))
    print("\n".join(rows))
    print("\nCodex를 쓴다면: 이 폴더에서 Codex를 처음 열 때 '폴더 신뢰'와 '훅 신뢰'를 묻습니다. 둘 다 승인하세요.\n"
          "Codex는 훅 신뢰를 폴더 경로별로 기억하므로, Orca에서 새 워크트리를 만들면 그 워크트리에서 한 번 더 승인해야 합니다.\n"
          "승인하지 않으면 규칙 안내만 남고 브랜치 차단·자동 push·PR 질문은 동작하지 않습니다.")


# -- entry ----------------------------------------------------------------------------------------

def main(argv=None):
    p = argparse.ArgumentParser(prog="git agent")
    sub = p.add_subparsers(dest="cmd", required=True)
    h = sub.add_parser("hook")
    h.add_argument("host", choices=["claude", "codex"])
    h.add_argument("event", choices=sorted(HOOKS))
    s = sub.add_parser("start", help="create the work branch from the latest base")
    s.add_argument("type")
    s.add_argument("slug")
    sub.add_parser("classify", help="re-check type and areas against the diff")
    t = sub.add_parser("template", help="write the PR draft from the templates")
    t.add_argument("--type")
    t.add_argument("--areas", help="comma separated: " + ",".join(AREAS))
    t.add_argument("--jira")
    t.add_argument("--release", action="store_true")
    r = sub.add_parser("pr", help="check the draft and open the PR")
    r.add_argument("--release", action="store_true")
    sub.add_parser("install")
    sub.add_parser("doctor")
    a = p.parse_args(argv)
    if a.cmd == "hook":
        return hook(a.host, a.event)
    for stream in (sys.stdout, sys.stderr):
        stream.reconfigure(encoding="utf-8")
    try:
        if a.cmd == "start":
            cmd_start(a.type, a.slug)
        elif a.cmd == "classify":
            print(json.dumps(classify(), ensure_ascii=False, indent=2))
        elif a.cmd == "template":
            cmd_template(a.type, [x for x in (a.areas or "").split(",") if x] or None, a.jira, a.release)
        elif a.cmd == "pr":
            cmd_pr(a.release)
        elif a.cmd == "install":
            cmd_install()
        elif a.cmd == "doctor":
            cmd_doctor()
    except Stop as error:
        print(error, file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
