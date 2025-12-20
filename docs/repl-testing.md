# RunMat REPL Testing Guide

This document describes the testing strategy for the RunMat REPL, including how to run tests, interpret failures, and add new tests.

---

## 1. Test Layers

### 1.1 Unit Tests (Fast)
Located in: `crates/runmat-repl/tests/repl.rs`

These test isolated components:
- Tokenization/lexing
- Individual line editor operations
- History buffer logic
- Escape sequence parsing (to be added)

Run with:
```bash
cargo test -p runmat-repl --lib
cargo test -p runmat-repl tokenize_simple_input
```

### 1.2 Integration Tests (Medium)
Located in: `crates/runmat-repl/tests/pty_basic.rs`

These spawn the REPL binary and interact via stdin/stdout:
- Prompt detection
- Expression execution
- Output formatting
- Command handling

Run with:
```bash
cargo test -p runmat-repl pty_spawn_and_detect_prompt -- --test-threads=1
```

**Note:** Most PTY tests are currently marked `#[ignore]` pending full test harness setup.

### 1.3 PTY Interactive Tests (End-to-End)
These send escape sequences (arrow keys, Ctrl sequences) and validate screen output.

**Status:** Not yet implemented. Requires:
- Rust PTY library (e.g., `ptyprocess` or `vte` crate)
- Or: Python helper script using `pexpect`

Example (future):
```bash
cargo test -p runmat-repl pty_history_up_arrow -- --test-threads=1 --nocapture
```

---

## 2. Running Tests Locally

### 2.1 Run All Unit Tests
```bash
cargo test -p runmat-repl --lib
```

### 2.2 Run Only REPL Tests
```bash
cargo test -p runmat-repl
```

### 2.3 Run a Specific Test
```bash
cargo test -p runmat-repl repl_binary_processes_single_line -- --nocapture
```

### 2.4 Enable Logging During Tests
```bash
RUST_LOG=debug cargo test -p runmat-repl pty_spawn_and_detect_prompt -- --nocapture
```

---

## 3. CI/CD Integration

### 3.1 Current CI Status
The RunMat CI (GitHub Actions) runs:
```bash
cargo test --all
```

This includes REPL tests. PTY tests are currently marked `#[ignore]` so they don't block CI.

### 3.2 Enabling PTY Tests in CI
Once the PTY test infrastructure is in place, add a CI job:

```yaml
# .github/workflows/ci.yml (example)
name: REPL Integration Tests

on: [push, pull_request]

jobs:
  repl-tests:
    runs-on: ${{ matrix.os }}
    strategy:
      matrix:
        os: [ubuntu-latest, macos-latest]
    steps:
      - uses: actions/checkout@v3
      - uses: dtolnay/rust-toolchain@stable
      - run: cargo test -p runmat-repl pty_
```

---

## 4. Test Failure Triage

### 4.1 Common Failures

#### **Failure: "REPL exited with non-zero status"**
- **Likely cause:** REPL panicked or initialization failed
- **Debug:** Run manually:
  ```bash
  cargo run -p runmat-repl --bin runmat-repl < /dev/null
  ```
- **Check logs:** Set `RUST_LOG=debug` and run the test with `--nocapture`

#### **Failure: "Expected 'RunMat REPL' in stdout"**
- **Likely cause:** Binary didn't run, or banner text changed
- **Debug:** Check `stdout` and `stderr` in test output
- **Fix:** Update spec or verify banner generation in `main.rs`

#### **Failure: "Expected 'ans = 2' in stdout"**
- **Likely cause:** Expression execution failed or output format changed
- **Debug:** Run the REPL manually and test the expression:
  ```bash
  echo "1 + 1" | cargo run -p runmat-repl --bin runmat-repl
  ```
- **Fix:** Check `lib.rs` expression handling or output formatting

#### **Failure: PTY test timeout**
- **Likely cause:** REPL waiting for input, or stuck evaluation
- **Debug:** Add longer timeouts or insert debug prints
- **Fix:** Check for infinite loops or unhandled edge cases

### 4.2 Adding Debug Output
In a test, capture stderr:
```rust
let stderr = String::from_utf8_lossy(&output.stderr);
eprintln!("Stderr output:\n{stderr}");
```

Or set environment variables:
```rust
.env("RUST_LOG", "debug")
```

---

## 5. Writing New Tests

### 5.1 Template: Unit Test (Tokenization)
```rust
#[test]
fn test_tokenize_example() {
    let result = format_tokens("your_input_here");
    assert_eq!(result, "expected token sequence");
}
```

### 5.2 Template: Integration Test (REPL I/O)
```rust
#[test]
#[ignore] // Remove when PTY harness is ready
fn pty_test_something() -> Result<(), Box<dyn std::error::Error>> {
    let mut child = Command::new(env!("CARGO_BIN_EXE_runmat-repl"))
        .stdin(Stdio::piped())
        .stdout(Stdio::piped())
        .stderr(Stdio::piped())
        .spawn()?;

    let mut stdin = child.stdin.take()?;
    stdin.write_all(b"your_command\n")?;
    stdin.write_all(b"exit\n")?;
    drop(stdin);

    let output = child.wait_with_output()?;
    assert!(output.status.success());
    
    let stdout = String::from_utf8_lossy(&output.stdout);
    assert!(stdout.contains("expected output"));

    Ok(())
}
```

### 5.3 Checklist for New Tests
- [ ] Test name is descriptive (e.g., `pty_history_up_arrow_recalls_previous`)
- [ ] Test has a comment explaining what it validates
- [ ] Test includes `#[ignore]` if it requires PTY setup
- [ ] Assertions have clear error messages (use `assert!(condition, "message with {variable}")`
- [ ] Test cleans up (drops stdin, handles errors)
- [ ] Test runs in isolation (not dependent on test execution order)

---

## 6. Snapshot Testing (for Stable Output)

When REPL output is complex or environment-specific, use snapshots:

### 6.1 Example: Golden Snapshot
```rust
#[test]
fn pty_help_output_matches_snapshot() -> Result<(), Box<dyn std::error::Error>> {
    // ... run REPL, capture help output
    let help_text = stdout; // captured from REPL
    
    // Normalize: remove ANSI codes, trailing spaces
    let normalized = help_text
        .lines()
        .map(|l| l.trim_end())
        .collect::<Vec<_>>()
        .join("\n");
    
    insta::assert_snapshot!(normalized);
    Ok(())
}
```

**Note:** Snapshot testing requires the `insta` crate. To be added if needed.

---

## 7. Platform-Specific Considerations

### 7.1 Linux
- PTY tests work natively
- ANSI escape codes handled correctly

### 7.2 macOS
- PTY tests work natively
- May need to disable `TERM` color codes in CI

### 7.3 Windows
- ConPTY not yet tested; skipped for now
- Smoke tests (stdin/stdout only) can run

**To skip PTY tests on Windows:**
```rust
#[cfg(not(target_os = "windows"))]
#[test]
fn pty_example() { /* ... */ }
```

---

## 8. Continuous Improvement

### 8.1 TODO (To Be Added)
- [ ] Implement full PTY harness (Rust or Python-based)
- [ ] Add escape sequence tests (Up/Down arrows, Ctrl+C)
- [ ] Add history navigation tests
- [ ] Add line editing tests (insert, delete, move)
- [ ] Add cross-platform smoke tests
- [ ] Set up snapshot testing for stable outputs
- [ ] Add performance regression tests

### 8.2 Tracking Issues
File issues for:
- Flaky tests (intermittent failures)
- Tests that hang or timeout
- Platform-specific failures
- Output format changes that break snapshots

Use labels: `[testing]`, `[repl]`, `[pty]`, `[ci]`

---

## 9. Quick Reference

| Command | Purpose |
|---------|---------|
| `cargo test -p runmat-repl` | Run all REPL tests |
| `cargo test -p runmat-repl --lib` | Run unit tests only |
| `RUST_LOG=debug cargo test -p runmat-repl -- --nocapture` | Run with debug logging |
| `cargo test -p runmat-repl -- --test-threads=1` | Run tests serially (useful for PTY) |
| `cargo test -p runmat-repl -- --ignored` | Run only ignored tests (after enabling) |

---

*Last updated: 2025-12-20*
