# RunMat REPL Behavioral Specification

**Version:** 0.2 (In Progress)  
**Last Updated:** 2025-12-20  
**Status:** Milestone 1 (MVP Complete) - History, editing, interrupt, and testing implemented

---

## 1. Overview

This document defines the expected behavior of the RunMat REPL to match MATLAB's Command Window experience. It covers prompts, output conventions, command sets, keybindings, and interrupt behavior.

---

## 2. Current State (Baseline)

### 2.1 Startup
- **Entry Point:** `runmat` binary (from `crates/runmat-repl/src/main.rs`)
- **Banner:**
  ```
  RunMat REPL v0.2.7
  High-performance MATLAB/Octave language runtime
  Type 'help' for help, 'exit' to quit, '.info' for system information
  ```
- **Prompt:** `runmat> ` (fixed string)

### 2.2 Execution & Output
- **Input Processing:**
  - Single-line statements only (no line continuation yet)
  - Semicolon suppresses output display
  - Assignments (`x = 1`) return the assigned value
  - Expressions (`1 + 1`) return their value

- **Output Format:**
  - Successful expression result: `ans = <value>`
  - Assignments with visible output: `ans = <value>`
  - Suppressed (semicolon): no output, but type info _should_ display
  - Errors: `Error: <message>` (to stderr)

- **Variables:**
  - `ans` persists between statements
  - All assignments persist in workspace
  - Displayed on expression output only

### 2.3 Built-in Commands
- `help` — show help text
- `exit` / `quit` — exit REPL
- `.info` — show system information (JIT status, GC stats)
- `.stats` — show execution statistics
- `.gc-info` — show garbage collector stats
- `.gc-collect` — trigger GC
- `.reset-stats` — reset execution stats

### 2.4 Line Editing
- **Current:** stdin-based, no interactive editing (piping works)
- **Not yet implemented:**
  - Up/Down history navigation
  - Left/Right cursor movement
  - Backspace/Delete editing
  - Ctrl+C interrupt (immediate exit)
  - Ctrl+D EOF handling

### 2.5 History
- **Current:** No persistent history
- **Not yet implemented:** History buffer with up/down navigation

---

## 3. Target State (MATLAB-like MVP)

### 3.1 Startup
- Same banner and prompt
- **Optional:** Environment variable flag for test mode (e.g., `RUNMAT_REPL_TEST=1`) to disable colors/spinners

### 3.2 Execution & Output (no change to core behavior, but stabilize)
- Semicolon suppression: still silent, but should **show type info** like MATLAB does
  - Example: `x = [1 2 3];` → no output (suppressed)
  - Example: `y = sin([1 2 3]);` → no output (suppressed) but MATLAB shows `(1x3 vector)` as type info
  - **RunMat currently:** silent with no type info
  - **Target:** silent (or optional type info on request)

### 3.3 Line Editing (Interactive)
- **Implemented via PTY/expect tests:**
  - `\x1b[A` (Up arrow) → cycle through history backward
  - `\x1b[B` (Down arrow) → cycle through history forward
  - `\x1b[C` (Right arrow) → move cursor forward one char
  - `\x1b[D` (Left arrow) → move cursor backward one char
  - Backspace / Delete → remove char at cursor
  - Home / End → move to start/end of line (optional in MVP)

### 3.4 Interrupt & Exit
- **Ctrl+C (`\x03`):** interrupt current evaluation, return to prompt (do not exit)
- **Ctrl+D (`\x04`):** graceful exit (EOF)

### 3.5 History Management
- Store last N statements (configurable, e.g., 100)
- Up/Down navigate history
- Newly typed text (not in history) is not lost when cycling back
- Down arrow past oldest entry clears line

### 3.6 Core Commands (Minimal Implementations)
These are shell-like commands, not MATLAB syntax:

#### 3.6.1 `pwd` — Print Working Directory
- Show current directory path
- Example: `pwd` → `/home/user/projects`

#### 3.6.2 `cd <path>` — Change Directory
- Change working directory
- Support `.`, `..`, absolute, and relative paths
- Example: `cd ..` → changes to parent directory

#### 3.6.3 `dir` / `ls` — List Directory
- List files in current directory
- Both aliases accepted
- Can include optional path argument: `dir /path/to/dir`

#### 3.6.4 `clear` / `clearvars` — Clear Variables
- Delete all workspace variables except `ans` (optional)
- `clear` → clear all
- `clear x y` → clear specific vars

#### 3.6.5 `who` / `whos` — List Variables
- `who` → one-line list of variable names
- `whos` → detailed list with sizes and types

#### 3.6.6 `help <name>` — Show Help
- Display help for a builtin or command
- Stub implementation: "No documentation available"

### 3.7 Keybindings Summary
| Key | Behavior |
|-----|----------|
| Enter | Execute line |
| Up | Previous history item |
| Down | Next history item |
| Left | Move cursor left |
| Right | Move cursor right |
| Backspace | Delete char before cursor |
| Delete | Delete char at cursor |
| Ctrl+C | Interrupt execution, return to prompt |
| Ctrl+D | Exit (EOF) |
| Home | Move to start of line (optional MVP) |
| End | Move to end of line (optional MVP) |

---

## 4. Known Deviations from MATLAB

- **No multi-line statements** (yet): Statements must fit on one line; line continuation (`...`) not implemented
- **Prompt is fixed:** MATLAB shows variable context in prompt; RunMat does not
- **Type info on semicolon:** MATLAB displays type silently; RunMat currently does not
- **No command history GUI:** MATLAB Desktop has a History pane; CLI REPL does not
- **No variable browser:** No Workspace pane equivalent
- **No debugging:** No `dbstop`, `dbcont`, etc.
- **Partial command implementations:** Many commands are stubs or minimal

---

## 5. Test Strategy

### 5.1 Layers
1. **Unit tests** (fast):
   - History buffer append/navigation
   - Line editor (insert, delete, move cursor)
   - Escape sequence decoding
   
2. **PTY integration tests** (end-to-end):
   - Spawn REPL in pseudo-terminal
   - Send bytes for arrow keys, edits, etc.
   - Assert prompt appears, output matches expected

### 5.2 Assertions
- **Golden snapshots:** prompt text, `ans =` format, error messages
- **Regex/contains:** paths (environment-specific), file listings
- **Timing:** wait for prompt before asserting

### 5.3 Platform-Specific
- **Linux/macOS:** Full PTY tests
- **Windows:** ConPTY if available; fallback to minimal smoke tests

---

## 6. Implementation Notes

### 6.1 Architecture Components (Proposed)
If refactoring is needed:
- **Input decoder:** Bytes → events (Char, Arrow, Ctrl, etc.)
- **Line editor:** Buffer + cursor position, apply edits
- **History manager:** Circular buffer, navigate with up/down
- **Renderer:** Print prompt, current line, handle redraws
- **Evaluator:** (already exists) run parsed MATLAB code

### 6.2 Test Mode Flag
Add support for `RUNMAT_REPL_TEST=1` or `--repl-test`:
- Disable spinners and ANSI colors
- Use stable prompt string
- Enable deterministic output

---

## 7. Checklist

### Milestone 0 (Complete)
- [x] Document current REPL state
- [x] Enumerate target features (MVP scope)
- [x] List commands and keybindings
- [x] Define test strategy
- [x] Create minimal PTY test to spawn REPL and detect prompt
- [x] Verify CI can run PTY tests
- [x] Initial spec and testing docs

### Milestone 1 (In Progress/Complete)
- [x] Implement history buffer with rustyline
- [x] Integrate line editor (up/down history navigation)
- [x] Add Ctrl+C interrupt handling (return to prompt, don't exit)
- [x] Add Ctrl+D EOF handling (graceful exit)
- [x] Add test mode flag (`RUNMAT_REPL_TEST=1`) for deterministic output
- [x] Write 7 PTY integration tests covering:
  - Prompt detection
  - Assignment display
  - Semicolon suppression
  - `ans` persistence
  - Exit command
  - Help command
  - Ctrl+D EOF exit
- [x] All tests pass `cargo fmt` and `cargo clippy`
- [x] All existing tests still pass

---

## 8. Appendix: Example REPL Sessions

### Session 1: Basic Expressions and Assignments
```
>> 1 + 1
ans = 2

>> x = [1, 2, 3]
ans = 1 2 3

>> y = x .* 2;

>> y
ans = 2 4 6

>> ans
ans = 2 4 6
```

### Session 2: History and Editing
```
>> a = 5
ans = 5

>> b = a + 1
ans = 6

>> (up arrow) → recalls "b = a + 1"
(left x3, backspace, type "0") → edits to "b = a + 10"
b = a + 10
ans = 15

>> (up x2) → recalls "a = 5"
```

### Session 3: Interrupt
```
>> while true, end
(Ctrl+C)
>> (prompt returns, loop interrupted)
```

---

*End of specification. Next updates will track progress through Milestones 1–3.*
