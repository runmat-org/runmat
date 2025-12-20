# Proposal: LLM Agent to Make RunMat’s REPL Match MATLAB Command Window

Repository target: `runmat-org/runmat` (interactive CLI / REPL)

## 1) Goal

Build and maintain a **MATLAB-like interactive REPL experience** in RunMat by:
- defining a **behavioral spec** (commands + interactive key handling),
- implementing **PTY-driven automated tests** to lock the behavior down,
- running **continuous regressions** in CI, and
- iterating on fixes until the REPL behaves consistently across platforms.

This proposal describes an LLM agent that does the bulk of the work: discovery, spec authoring, test harness creation, bug-fixing loops, and PR-ready changes.

---

## 2) Non-goals (to keep scope sane)

- Full MATLAB language completeness (parser/runtime parity) beyond what RunMat already supports.
- GUI features of MATLAB Desktop (Command History pane, Workspace browser, etc.).
- Full parity with every MATLAB preference/setting (e.g., alternative arrow-key modes). We pick **one default** “MATLAB-like” mode.

---

## 3) Deliverables

### D1. REPL Behavior Spec
A stable, versioned spec at:
- `docs/repl-spec.md`

It includes:
- prompts, output formatting conventions,
- `ans` and semicolon suppression rules,
- core shell-like commands (`pwd`, `cd`, `dir/ls`, `clear`, `who/whos`, `help`),
- keybindings (arrows, backspace, delete, ctrl sequences),
- interrupt/exit behavior (Ctrl+C, Ctrl+D),
- known deviations (documented and intentional).

### D2. Automated REPL Test Harness (PTY-based)
- Integration tests under `tests/` (Rust-first), or a hybrid with a small Python runner if Rust PTY support is too slow to implement.
- Tests run in CI across Linux/macOS/Windows where possible.
- Output assertions: golden snapshots where stable; regex matching where environment-specific.

### D3. Regression Workflow
- GitHub Actions job(s) that run REPL tests on every PR/push.
- A “failure triage” playbook in `docs/repl-testing.md`.

### D4. Bugfix PRs
A sequence of PR-ready commits that:
- implement missing features,
- fix terminal-escape parsing,
- stabilize output, and
- ensure tests pass.

---

## 4) Agent Operating Model

### 4.1 Agent Roles (single agent, multiple hats)
1. **Spec Author**: enumerates expected behaviors and writes `docs/repl-spec.md`.
2. **Test Engineer**: writes PTY/expect tests and fixtures.
3. **Debugger**: runs failing tests, bisects behavior, proposes and implements fixes.
4. **Maintainer**: refactors for stability, adds logging flags, updates docs.

### 4.2 Inputs the Agent Uses
- The RunMat source tree (commands, repl implementation, prompt rendering, history).
- Existing tests (if any).
- Issue discussions / PRs in the repo (for intended behavior and constraints).
- Terminal behavior standards (ANSI/VT100 escape sequences) as practical guidance.

### 4.3 Outputs the Agent Produces
- Concrete diffs/patches (files + tests) that can be merged.
- A running changelog in PR descriptions, mapping features to tests.

---

## 5) Proposed Feature Inventory (“MATLAB feel” MVP)

### 5.1 Language/Session Semantics
- `ans`:
  - expression without assignment stores in `ans` and prints `ans = ...`.
- Output suppression:
  - `;` suppresses printing.
- Basic errors:
  - error format is consistent and parseable (don’t over-specify MATLAB’s exact wording; do stabilize RunMat’s).

### 5.2 Workspace & Introspection (minimal)
- `clear` / `clearvars`
- `who`, `whos`
- `exist`, `which`, `type` (even partial/stub implementations are fine if documented)

### 5.3 Filesystem / Current Folder
- `pwd`
- `cd`
- `dir` and `ls` (aliases acceptable)
- path resolution rules documented (relative vs absolute; platform differences allowed)

### 5.4 Interactive Editing & History
- Arrow keys:
  - Up/Down cycle through history
  - Left/Right move cursor
- Editing:
  - backspace/delete
  - home/end (optional early)
- Ctrl sequences:
  - Ctrl+C interrupts current evaluation and returns to prompt
  - Ctrl+D exits (or sends EOF) gracefully

---

## 6) Test Strategy

### 6.1 Why PTY Tests
Arrow keys and in-line editing require a pseudo-terminal. Pipes won’t reproduce interactive behavior reliably.

### 6.2 Test Layers
1. **Pure unit tests** (fast):
   - parser/tokenizer for escape sequences,
   - history ring behavior,
   - line editor transformations (insert/delete/move).
2. **PTY integration tests** (end-to-end):
   - spawn `runmat` REPL in a PTY,
   - send bytes (`\x1b[A` for up arrow, etc.),
   - assert screen output and prompt reappearance.

### 6.3 Assertion Style
- Use **golden snapshots** for stable outputs (prompt, `ans =`, error headers).
- Use **regex/contains** for environment-specific output (paths, directory listing).
- Normalize:
  - `\r\n` vs `\n`,
  - terminal control codes (strip or explicitly match them),
  - timing (wait for prompt).

### 6.4 Cross-platform Considerations
- Linux/macOS: straightforward PTY.
- Windows:
  - Prefer ConPTY (if feasible), otherwise run PTY tests on non-Windows first and keep Windows smoke tests minimal.
- The agent will:
  - detect platform in CI,
  - skip or downgrade tests where PTY isn’t supported yet,
  - track parity milestones.

---

## 7) Implementation Approach (Recommended Architecture)

### 7.1 Treat the REPL as a small state machine
Core components:
- **Input decoder**: bytes → events (char, enter, arrow up, ctrl-c, etc.)
- **Line editor**: (buffer, cursor) + event → (buffer, cursor, redraw plan)
- **History manager**: stores statements, navigates with up/down
- **Evaluator**: runs full line on enter, returns output + status
- **Renderer**: prints prompt + output + redraws current line cleanly

The agent will refactor toward this shape *only if necessary* to make behavior stable and testable.

### 7.2 Add a “test mode” switch (optional but powerful)
Example:
- `RUNMAT_REPL_TEST=1` or `--repl-test`
- disables spinners/ANSI color (or makes them deterministic),
- forces a stable prompt string,
- can enable verbose logging to stderr.

This reduces flaky output and makes PTY tests robust.

---

## 8) Work Plan & Milestones

### Milestone 0 — Discovery (1–2 PRs)
- Map current REPL features.
- Add `docs/repl-spec.md` skeleton with current/target behavior.
- Add minimal “spawn REPL, see prompt” PTY test.

**Acceptance:**
- CI can spawn REPL and detect prompt reliably.

### Milestone 1 — “Feels like MATLAB” MVP (2–5 PRs)
- Implement/lock down:
  - `ans` behavior,
  - semicolon suppression,
  - history recall via up/down,
  - left/right editing,
  - ctrl+c interrupt,
  - ctrl+d exit.

**Acceptance:**
- PTY tests for the above pass on Linux (and ideally macOS).

### Milestone 2 — Core command set (2–6 PRs)
- `pwd`, `cd`, `dir/ls`, `clear`, `who/whos`, `help <name>` minimal.

**Acceptance:**
- Command tests pass and output is stable under `--repl-test`.

### Milestone 3 — Hardening & Cross-platform (ongoing)
- Windows support strategy (ConPTY or partial skips).
- Reduce flakiness, improve redraw fidelity, refine error output.

**Acceptance:**
- CI runs regressions on at least Linux/macOS; Windows has smoke coverage.

---

## 9) How the Agent Works Day-to-Day (Loop)

1. **Pick one behavior** from `docs/repl-spec.md` marked “TODO”.
2. Write a failing **PTY integration test** for it.
3. Implement or fix REPL behavior until test passes.
4. Add/adjust **unit tests** for the underlying component (decoder/editor/history).
5. Push as PR-ready commit with:
   - spec update,
   - tests,
   - implementation.

---

## 10) Quality Guardrails

- No behavior lands without a test (unless explicitly waived in the spec).
- Avoid exact MATLAB wording unless it’s trivial; focus on stable UX.
- Keep terminal output deterministic in test mode.
- Keep diff sizes manageable:
  - Prefer small PRs (1–3 features) to reduce merge friction.

---

## 11) What You (Maintainer) Provide to the Agent

- How to start the REPL (binary name/flags), e.g. `runmat repl` or `runmat`.
- CI constraints (supported OSes, current GitHub Actions matrix).
- Any “must match MATLAB exactly” areas vs “close enough”.

If any of the above is missing, the agent will infer from `README`/`Cargo.toml`/CLI flags and document assumptions in the PR.

---

## 12) Success Criteria (Definition of Done)

RunMat REPL is “MATLAB-like” when:
- a user can type expressions and rely on `ans`,
- history and in-line editing behave predictably with arrow keys,
- Ctrl+C interrupts and returns to prompt,
- common session/file commands work,
- regressions are caught automatically in CI,
- deviations from MATLAB are clearly documented.

---

## 13) Appendix: Initial Test Case Ideas

### A. `ans` and suppression
- `1+1` prints `ans = 2`
- `1+1;` prints nothing

### B. History
- run `a=1`, `b=2`, then UpArrow → recalls `b=2`
- UpArrow again → `a=1`
- DownArrow → `b=2`

### C. Editing
- type `disp(123)` then left x1 then backspace then type `4` → executes `disp(124)`

### D. Interrupt
- `while true, end` then Ctrl+C → returns to prompt

### E. PWD/CD
- `pwd`, `cd ..`, `pwd` shows change (regex/contains assertions)

---

*End of proposal.*
