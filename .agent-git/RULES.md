# Git workflow rules (agent-git)

The people you work with are new to the command line and work in Korean. Speak to them in plain Korean,
explain git steps in one short sentence when you take them, and never leave them a command to type
when you can run it yourself. These rules are enforced by hooks; when a hook blocks you, follow its message.

## 1. Branch model

- `main`: stable, always runnable. Receives changes only from `dev` through a release PR.
- `dev`: integration. Receives changes only from work branches through PRs.
- Work branches: `<type>/<slug>`, always cut from the latest `origin/dev`.
- Never commit or push to `main` or `dev` directly. No force push, no `git reset --hard`, no merging your own PR.

## 2. Classify the task (type x area)

Every task has exactly one **type** (what kind of change) and zero or more **areas** (where it lands).
The type names the branch, the commit prefix, the PR title and the PR template; the areas add sections
and labels to the PR.

### Type: ask these in order and take the first "yes"

| Type | Korean label | Ask |
| --- | --- | --- |
| `fix` | 수정 | Did something behave differently from what was intended, and does this make it behave as intended? |
| `feature` | 기능 | Can someone do something new afterwards, or does an experiment try a new model, prompt or setting? |
| `refactor` | 구조 개선 | Does the code structure change while behaviour stays exactly the same? |
| `test` | 테스트 | Do only tests or validation code change? |
| `docs` | 문서 | Do only documents or comments change? |
| `style` | 서식 | Does only formatting change (whitespace, line breaks, import order), with no change in meaning? |
| `chore` | 환경 설정 | Do only dependencies, configuration, build or tooling change? |

Rules for mixed work:
- The type is the **purpose the user asked for**. Tests or docs written along the way do not change it;
  they go in the PR section 함께 바뀐 것.
- If the purpose itself is ambiguous, use the priority order above (fix > feature > refactor > test > docs > style > chore).
- For generative-AI work: trying a new model, prompt, preset or parameter set is `feature`; making the output
  match what was already intended (wrong colors, cut-off text, crashes) is `fix`; changing only a value in a
  config file with no new capability is still `feature` if it changes the generated result, otherwise `chore`.

### Areas

| Key | Korean label | What it covers |
| --- | --- | --- |
| `screen` | 화면 | What the shop owner sees and clicks (Streamlit, React, upload and result pages) |
| `server` | 서버 | API endpoints, request/response formats, saving results, error messages |
| `serving` | 서빙 | Loading models once, GPU memory, speed, queueing, Docker, the VM |
| `image` | 이미지 생성 | Image or video generation, relighting, segmentation, compositing |
| `copy` | 문구 생성 | Ad copy and captions: LLM prompts, tone, length, banned expressions |
| `eval` | 평가 | Quality checks and metrics, test photo sets, review reports |

Areas are decided by the file patterns in `.agent-git/config.json` first (`git agent classify` lists them).
Files no pattern covers are yours to judge with the table above. The first area is the main one and goes
into the PR title.

## 3. Start: before the first file edit

1. Classify from the user's request. Pick a slug: 2-4 lowercase English words joined by `-`, at most 40
   characters, naming the work (e.g. `light-preset`, `upload-size-error`).
2. Run `git agent start <type> <slug>`. It fetches `origin/dev` and either renames the branch Orca created for
   this worktree or creates a new branch, so the work always starts from the latest `dev`.
3. Tell the user in one Korean sentence which branch you made and why that type.

If the work turns out to be a different type: before the first commit, run `git agent start <new-type> <slug>`
again to rename the branch. After the branch is pushed, keep the name and pass `--type` when building the PR.

Pure questions, reviews and explanations need no branch. Edits outside this repository are never blocked.

## 4. Commit

- Format: `<type>: <Korean description>`, e.g. `feature: 조명 프리셋 4종 추가`, `fix: 업로드 용량 오류 수정`.
  The type is one of the seven English keys; the description is short, plain Korean.
- Commit in meaningful units: one idea per commit.
- Every commit on a work branch is pushed automatically by a hook (`-u` the first time). Do not push by hand
  unless the hook reports a failure.
- To test code that lives only on another work branch: commit or stash first, `git fetch origin`, then
  `git merge origin/<type>/<slug>`. Tell the user that both branches stay as they are.

## 5. Finish: summarize, then ask

When the work looks done, a Stop hook returns control once with these steps. Do them in order:
1. Commit whatever is left (meaningful units, format above).
2. Run `git agent classify`. If `agrees` is false, tell the user the branch type and the diff disagree and
   which type you will use for the PR.
3. Summarize what you did in 3-5 short Korean bullets, without jargon.
4. Ask exactly one question, "PR을 올릴까요?", with two choices: "PR 올리기" and "아니오".
   Claude Code: use AskUserQuestion. Codex: write the two numbered choices and end the turn.
5. "아니오" ends it. You will not be asked again until something new is committed.

## 6. Open the PR (base: dev)

1. If the change alters an image or a screen, save before/after images as
   `docs/pr-images/<slug>/before-1.png`, `after-1.png`, and commit them (`docs: PR 전후 이미지 추가` is fine).
2. Run `git agent template` (add `--areas image,eval` when classify found no areas, `--type` when the
   branch type is wrong, `--jira AD-12` when the user gave a Jira key). It writes `.agent-git/local/pr-draft.md`
   from `.agent-git/templates/`: a common head, the type's sections, one section per area, a common tail and an
   automatic file list.
3. Fill the draft. The first line is the title. Replace every `{{...}}`. Delete optional sections that do not
   apply; never leave an empty heading. Keep the order.
4. Run `git agent pr`. It checks the writing rules below, then opens the PR with type and area labels,
   area owners as reviewers, and the user as assignee. If it lists problems, fix the draft and run it again.
5. Give the user the PR link and the local journal draft path it prints (a daily-log draft in the team's format).

### Writing rules for the PR title and body (checked by `git agent pr`)

- Title: `[<type label>/<main area label>] <Korean description>`, e.g. `[기능/이미지 생성] 제품 사진 조명 프리셋 추가`.
  Without an area: `[수정] 업로드 용량 오류 수정`.
- Korean only. No English words in sentences: write 변경, 요청, 화면, 그래픽 메모리 instead of PR, request, UI, GPU.
- Real names that cannot be translated (files, functions, models, libraries) go in backticks, at most two
  per sentence, and only at their **first mention**. After that, refer to them in Korean: 이 함수, 조명 모델, 이 서버.
- Prefer a table when several names or numbers must appear; tables and code blocks are where names belong.
- Write the way a teammate would explain it out loud: short sentences, subject and verb, no translationese,
  no restating the heading, no guessing. Numbers come with what they were compared against.
- The 바뀐 파일 section is generated; do not edit it.

## 7. Release PR (dev -> main)

Only when the user asks. Run `git agent template --release`: it lists every PR merged into `dev` since the last
release, grouped by type. Fill 한 줄 요약, 통합 테스트, 알려진 문제 and 배포 전 확인 with what was actually run,
then `git agent pr --release`. Same writing rules; the title is `[통합] <Korean description>`.

## 8. When something fails

- `gh` missing or logged out: `git agent pr` prints install and login steps; relay them in Korean.
- Automatic push failed: tell the user and show the one command that would retry it.
- Anything else odd: run `git agent doctor` and read its report to the user.
