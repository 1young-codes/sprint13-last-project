#!/bin/sh
# Started by git from the repository root (hooks and `git agent ...` both go through git aliases),
# so this works the same under Git Bash, PowerShell, cmd and macOS/Linux shells.
for py in python3 python py; do
  if command -v "$py" >/dev/null 2>&1 && "$py" -c "import sys; sys.exit(sys.version_info < (3, 8))" >/dev/null 2>&1; then
    exec "$py" "$(dirname "$0")/flow.py" "$@"
  fi
done
echo "agent-git: Python 3.8+ not found; hooks are skipped." >&2
exit 0
