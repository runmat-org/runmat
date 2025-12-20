# MATLAB-like REPL with Shell Commands (Milestones 0-2)

## Summary

Complete implementation of MATLAB-like REPL features: behavioral specification, automated testing framework, rustyline integration (history, line editing), and core shell commands (pwd, cd, dir/ls, clear, who/whos).

## Milestones Completed

### Milestone 0: Discovery & Specification
- [x] `docs/repl-spec.md` — Comprehensive behavioral spec (350+ lines)
- [x] `docs/repl-testing.md` — Testing guide with PTY infrastructure notes
- [x] PTY test foundation (`tests/pty_basic.rs`)

### Milestone 1: Interactive MVP (Code Complete)
- [x] Rustyline integration in `main.rs`
  - History buffer with up/down navigation (code)
  - Line editing support (code)
  - Ctrl+C handler → returns to prompt (code)
  - Ctrl+D handler → graceful exit (code + tested)
- [x] Test mode flag `RUNMAT_REPL_TEST=1` for deterministic output
- [x] 7 piped I/O tests (basic execution validation)
- ⚠️ Note: Interactive features implemented but NOT tested via PTY (requires real PTY library in future M3)

### Milestone 2: Shell Commands
- [x] New `commands.rs` module with 6 commands:
  - `pwd` — print working directory
  - `cd <path>` — change directory
  - `dir` / `ls` — list files
  - `clear` / `clearvars` — clear variables
  - `who` — list variable names
  - `whos` — detailed variable info
- [x] 6 integration tests + 3 unit tests for commands
- [x] Integrated into REPL loop via command dispatcher

## Testing

**94 tests passing:**
- 3 command unit tests
- 6 command integration tests
- 7 PTY basic tests
- 9 REPL tokenization tests
- 9 semicolon suppression tests
- 10 variable persistence tests
- 41+ other existing tests

**Checks:**
- ✅ `cargo fmt --check`
- ✅ `cargo clippy -- -D warnings`
- ✅ All existing tests still pass

## Changes

### Files Added
- `docs/repl-spec.md` — REPL behavioral specification
- `docs/repl-testing.md` — Testing strategy & guide
- `docs/REPL_PROGRESS.md` — Development progress tracker
- `crates/runmat-repl/src/commands.rs` — Shell command implementation
- `crates/runmat-repl/tests/commands.rs` — Command integration tests

### Files Modified
- `crates/runmat-repl/src/lib.rs` — Add commands module
- `crates/runmat-repl/src/main.rs` — Replace stdin loop with rustyline + command dispatch
- `crates/runmat-repl/tests/pty_basic.rs` — Enable 7 tests, use `RUNMAT_REPL_TEST=1`

## Commits

1. **docs(repl): add spec and testing guide for MATLAB-like REPL (Milestone 0)**
   - Behavioral specification, testing guide, PTY harness foundation

2. **feat(repl): implement rustyline integration with history, interrupt, and EOF handling (Milestone 1)**
   - Replace stdin loop with rustyline
   - Ctrl+C and Ctrl+D handling
   - Test mode flag for deterministic output
   - 7 piped I/O tests

3. **docs: clarify that M1 tests are piped I/O only, not interactive PTY tests**
   - Honest assessment of what's tested vs. what's coded

4. **feat(repl): add shell commands pwd, cd, dir/ls, clear, who/whos (Milestone 2)**
   - Commands module with 6 shell-like commands
   - 6 integration tests + 3 unit tests

5. **docs: add REPL progress summary (M0-M2 complete, 94 tests)**
   - Development tracker with timeline and recommendations

## Known Limitations

### Implemented but Not Tested
- Up/Down history navigation (rustyline code exists, needs PTY test)
- Left/Right cursor movement (rustyline code exists, needs PTY test)
- Backspace/Delete editing (rustyline code exists, needs PTY test)
- Ctrl+C interrupt behavior (code exists, needs PTY test)

**Why:** Current tests use piped stdin (non-TTY). Real PTY library integration required to test escape sequences and signals. Deferred to Milestone 3.

### Not Implemented (Out of Scope M0-2)
- Multi-line statements / line continuation
- Tab completion
- Persistent history across sessions
- Command-specific help (help <cmd> is stub)
- Variable browser / workspace pane

## Next Steps (Milestone 3)

1. Integrate PTY library (Python `pexpect` recommended)
2. Write arrow key navigation tests
3. Write Ctrl+C interrupt tests
4. Windows ConPTY support or skip strategy
5. CI configuration for PTY tests

See `docs/REPL_PROGRESS.md` for detailed timeline.

## Code Quality

- ✅ No unsafe code added
- ✅ Full documentation (spec, testing guide, progress tracker)
- ✅ Clear separation of concerns (commands module, REPL engine, tests)
- ✅ Easy to extend (new commands = add case to `parse_and_execute`)
- ✅ Follows AGENTS.md (one logical change per PR, all checks pass)

## How to Test

```bash
# Run all REPL tests
cargo test -p runmat-repl

# Run specific test suites
cargo test -p runmat-repl --test commands    # Shell commands
cargo test -p runmat-repl --test pty_basic   # PTY basics

# Run REPL with test mode
RUNMAT_REPL_TEST=1 cargo run -p runmat-repl --bin runmat-repl

# Try commands interactively
cargo run -p runmat-repl --bin runmat-repl
# Then: pwd, cd .., dir, x=42, who, clear
```

## Acceptance Criteria

- [x] Spec & testing docs complete
- [x] Rustyline integrated with handlers for Ctrl+C/Ctrl+D
- [x] 6 shell commands implemented and tested
- [x] 94 tests passing (including all existing tests)
- [x] `cargo fmt` and `cargo clippy` clean
- [x] One logical change per commit
- [x] Ready for review and merge
