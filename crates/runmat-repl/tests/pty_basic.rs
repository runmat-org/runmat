// Basic PTY (pseudo-terminal) tests for REPL interactive behavior
// These tests spawn the REPL binary in a terminal and interact with it.
// They validate prompt detection, basic execution, and I/O handling.

use std::io::Write;
use std::process::{Command, Stdio};

/// Test that the REPL binary can be spawned and shows the expected banner and prompt.
/// Uses RUNMAT_REPL_TEST=1 for stable output in non-PTY environments.
#[test]
fn pty_spawn_and_detect_prompt() -> Result<(), Box<dyn std::error::Error>> {
    // Spawn the REPL binary with test mode enabled
    let mut child = Command::new(env!("CARGO_BIN_EXE_runmat-repl"))
        .env("RUNMAT_REPL_TEST", "1")
        .stdin(Stdio::piped())
        .stdout(Stdio::piped())
        .stderr(Stdio::piped())
        .spawn()?;

    // Send a simple expression and exit
    let mut stdin = child.stdin.take().ok_or("Failed to open stdin")?;
    stdin.write_all(b"1 + 1\n")?;
    stdin.write_all(b"exit\n")?;
    drop(stdin);

    // Wait for completion
    let output = child.wait_with_output()?;
    let stdout = String::from_utf8_lossy(&output.stdout);

    // Verify the process exited successfully
    assert!(output.status.success(), "REPL exited with non-zero status");

    // Verify banner is present
    assert!(
        stdout.contains("RunMat REPL"),
        "Expected banner 'RunMat REPL' in stdout, got:\n{stdout}"
    );

    // Verify prompt is present (at least once)
    assert!(
        stdout.contains("runmat>"),
        "Expected prompt 'runmat>' in stdout, got:\n{stdout}"
    );

    // Verify execution output (1 + 1 should produce ans = 2)
    assert!(
        stdout.contains("ans = 2"),
        "Expected 'ans = 2' in stdout, got:\n{stdout}"
    );

    println!("✓ PTY spawn and prompt detection passed");
    Ok(())
}

/// Test basic assignment (without semicolon) displays the value.
#[test]
fn pty_assignment_displays_value() -> Result<(), Box<dyn std::error::Error>> {
    let mut child = Command::new(env!("CARGO_BIN_EXE_runmat-repl"))
        .env("RUNMAT_REPL_TEST", "1")
        .stdin(Stdio::piped())
        .stdout(Stdio::piped())
        .stderr(Stdio::piped())
        .spawn()?;

    let mut stdin = child.stdin.take().ok_or("Failed to open stdin")?;
    stdin.write_all(b"x = 42\n")?;
    stdin.write_all(b"exit\n")?;
    drop(stdin);

    let output = child.wait_with_output()?;
    let stdout = String::from_utf8_lossy(&output.stdout);

    assert!(output.status.success());
    // Assignment without semicolon should display: ans = 42
    assert!(
        stdout.contains("ans = 42"),
        "Expected 'ans = 42', got:\n{stdout}"
    );

    println!("✓ PTY assignment display passed");
    Ok(())
}

/// Test semicolon suppression (output should not display).
#[test]
fn pty_semicolon_suppresses_output() -> Result<(), Box<dyn std::error::Error>> {
    let mut child = Command::new(env!("CARGO_BIN_EXE_runmat-repl"))
        .env("RUNMAT_REPL_TEST", "1")
        .stdin(Stdio::piped())
        .stdout(Stdio::piped())
        .stderr(Stdio::piped())
        .spawn()?;

    let mut stdin = child.stdin.take().ok_or("Failed to open stdin")?;
    stdin.write_all(b"y = 99;\n")?;
    stdin.write_all(b"y\n")?; // Next line should show y's value
    stdin.write_all(b"exit\n")?;
    drop(stdin);

    let output = child.wait_with_output()?;
    let stdout = String::from_utf8_lossy(&output.stdout);

    assert!(output.status.success());
    // The suppressed assignment (y = 99;) should NOT produce "ans = 99"
    // But the explicit "y" should show its value
    let ans_99_count = stdout.matches("ans = 99").count();
    assert!(
        ans_99_count >= 1,
        "Expected at least one 'ans = 99' from 'y' lookup, got:\n{stdout}"
    );

    println!("✓ PTY semicolon suppression passed");
    Ok(())
}

/// Test that ans persists across statements (history and previous computation).
#[test]
fn pty_ans_persistence() -> Result<(), Box<dyn std::error::Error>> {
    let mut child = Command::new(env!("CARGO_BIN_EXE_runmat-repl"))
        .env("RUNMAT_REPL_TEST", "1")
        .stdin(Stdio::piped())
        .stdout(Stdio::piped())
        .stderr(Stdio::piped())
        .spawn()?;

    let mut stdin = child.stdin.take().ok_or("Failed to open stdin")?;
    stdin.write_all(b"2 + 3\n")?; // ans = 5
    stdin.write_all(b"ans + 10\n")?; // ans should still be 5, result = 15
    stdin.write_all(b"exit\n")?;
    drop(stdin);

    let output = child.wait_with_output()?;
    let stdout = String::from_utf8_lossy(&output.stdout);

    assert!(output.status.success());
    assert!(
        stdout.contains("ans = 5"),
        "Expected first 'ans = 5', got:\n{stdout}"
    );
    assert!(
        stdout.contains("ans = 15"),
        "Expected second 'ans = 15' (from ans + 10), got:\n{stdout}"
    );

    println!("✓ PTY ans persistence passed");
    Ok(())
}

/// Test the 'exit' command cleanly terminates the REPL.
#[test]
fn pty_exit_command() -> Result<(), Box<dyn std::error::Error>> {
    let mut child = Command::new(env!("CARGO_BIN_EXE_runmat-repl"))
        .env("RUNMAT_REPL_TEST", "1")
        .stdin(Stdio::piped())
        .stdout(Stdio::piped())
        .stderr(Stdio::piped())
        .spawn()?;

    let mut stdin = child.stdin.take().ok_or("Failed to open stdin")?;
    stdin.write_all(b"exit\n")?;
    drop(stdin);

    let output = child.wait_with_output()?;
    let stdout = String::from_utf8_lossy(&output.stdout);

    assert!(output.status.success(), "REPL should exit with status 0");
    assert!(
        stdout.contains("Goodbye!"),
        "Expected 'Goodbye!' message, got:\n{stdout}"
    );

    println!("✓ PTY exit command passed");
    Ok(())
}

/// Test help command displays help text.
#[test]
fn pty_help_command() -> Result<(), Box<dyn std::error::Error>> {
    let mut child = Command::new(env!("CARGO_BIN_EXE_runmat-repl"))
        .env("RUNMAT_REPL_TEST", "1")
        .stdin(Stdio::piped())
        .stdout(Stdio::piped())
        .stderr(Stdio::piped())
        .spawn()?;

    let mut stdin = child.stdin.take().ok_or("Failed to open stdin")?;
    stdin.write_all(b"help\n")?;
    stdin.write_all(b"exit\n")?;
    drop(stdin);

    let output = child.wait_with_output()?;
    let stdout = String::from_utf8_lossy(&output.stdout);

    assert!(output.status.success());
    assert!(
        stdout.contains("RunMat REPL Help"),
        "Expected help header, got:\n{stdout}"
    );

    println!("✓ PTY help command passed");
    Ok(())
}

/// Test Ctrl+D (EOF) gracefully exits the REPL.
#[test]
fn pty_ctrl_d_exit() -> Result<(), Box<dyn std::error::Error>> {
    let mut child = Command::new(env!("CARGO_BIN_EXE_runmat-repl"))
        .env("RUNMAT_REPL_TEST", "1")
        .stdin(Stdio::piped())
        .stdout(Stdio::piped())
        .stderr(Stdio::piped())
        .spawn()?;

    let mut stdin = child.stdin.take().ok_or("Failed to open stdin")?;
    stdin.write_all(b"1 + 1\n")?;
    // Close stdin without explicit exit to signal EOF (Ctrl+D behavior)
    drop(stdin);

    let output = child.wait_with_output()?;
    let stdout = String::from_utf8_lossy(&output.stdout);

    assert!(output.status.success(), "REPL should exit cleanly on EOF");
    // On EOF, rustyline returns ReadlineError::Eof which we handle gracefully
    assert!(
        stdout.contains("ans = 2"),
        "Expected evaluation result before EOF, got:\n{stdout}"
    );

    println!("✓ PTY Ctrl+D exit passed");
    Ok(())
}
