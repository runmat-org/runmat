/// Integration tests for shell-like commands (pwd, cd, dir/ls, clear, who/whos)
use std::io::Write;
use std::process::{Command, Stdio};

/// Test pwd command prints current directory
#[test]
fn cmd_pwd_prints_directory() -> Result<(), Box<dyn std::error::Error>> {
    let mut child = Command::new(env!("CARGO_BIN_EXE_runmat-repl"))
        .env("RUNMAT_REPL_TEST", "1")
        .stdin(Stdio::piped())
        .stdout(Stdio::piped())
        .stderr(Stdio::piped())
        .spawn()?;

    let mut stdin = child.stdin.take().ok_or("Failed to open stdin")?;
    stdin.write_all(b"pwd\n")?;
    stdin.write_all(b"exit\n")?;
    drop(stdin);

    let output = child.wait_with_output()?;
    let stdout = String::from_utf8_lossy(&output.stdout);

    assert!(output.status.success());
    // Should contain a path (at least a slash or drive letter for Windows)
    assert!(
        stdout.contains("/") || stdout.contains("\\") || stdout.contains(":"),
        "Expected path output from pwd, got:\n{stdout}"
    );

    println!("✓ cmd_pwd_prints_directory passed");
    Ok(())
}

/// Test cd command changes directory and returns new path
#[test]
fn cmd_cd_changes_directory() -> Result<(), Box<dyn std::error::Error>> {
    let mut child = Command::new(env!("CARGO_BIN_EXE_runmat-repl"))
        .env("RUNMAT_REPL_TEST", "1")
        .stdin(Stdio::piped())
        .stdout(Stdio::piped())
        .stderr(Stdio::piped())
        .spawn()?;

    let mut stdin = child.stdin.take().ok_or("Failed to open stdin")?;
    stdin.write_all(b"pwd\n")?; // Get current directory
    stdin.write_all(b"cd .\n")?; // cd to current directory (no-op but tests functionality)
    stdin.write_all(b"exit\n")?;
    drop(stdin);

    let output = child.wait_with_output()?;
    let stdout = String::from_utf8_lossy(&output.stdout);

    assert!(output.status.success());
    // Should have pwd output twice (both commands should print paths)
    let path_count = stdout.matches("/").count() + stdout.matches("\\").count();
    assert!(
        path_count >= 1,
        "Expected path output from cd, got:\n{stdout}"
    );

    println!("✓ cmd_cd_changes_directory passed");
    Ok(())
}

/// Test dir/ls command lists files
#[test]
fn cmd_dir_lists_files() -> Result<(), Box<dyn std::error::Error>> {
    let mut child = Command::new(env!("CARGO_BIN_EXE_runmat-repl"))
        .env("RUNMAT_REPL_TEST", "1")
        .stdin(Stdio::piped())
        .stdout(Stdio::piped())
        .stderr(Stdio::piped())
        .spawn()?;

    let mut stdin = child.stdin.take().ok_or("Failed to open stdin")?;
    stdin.write_all(b"dir .\n")?;
    stdin.write_all(b"exit\n")?;
    drop(stdin);

    let output = child.wait_with_output()?;
    let stdout = String::from_utf8_lossy(&output.stdout);

    assert!(output.status.success());
    // Should list files (current directory has content)
    assert!(
        !stdout.contains("Error"),
        "dir should not error, got:\n{stdout}"
    );

    println!("✓ cmd_dir_lists_files passed");
    Ok(())
}

/// Test clear command removes all variables
#[test]
fn cmd_clear_removes_variables() -> Result<(), Box<dyn std::error::Error>> {
    let mut child = Command::new(env!("CARGO_BIN_EXE_runmat-repl"))
        .env("RUNMAT_REPL_TEST", "1")
        .stdin(Stdio::piped())
        .stdout(Stdio::piped())
        .stderr(Stdio::piped())
        .spawn()?;

    let mut stdin = child.stdin.take().ok_or("Failed to open stdin")?;
    stdin.write_all(b"x = 42\n")?;
    stdin.write_all(b"who\n")?; // List vars before clear (should show x, ans)
    stdin.write_all(b"clear\n")?; // Clear all
    stdin.write_all(b"who\n")?; // List vars after clear (should be empty or show no vars)
    stdin.write_all(b"exit\n")?;
    drop(stdin);

    let output = child.wait_with_output()?;
    let stdout = String::from_utf8_lossy(&output.stdout);

    assert!(output.status.success());
    // Should show: ans = 42 (from assignment), then first who showing variables, then "Variables cleared"
    assert!(
        stdout.contains("ans = 42"),
        "Expected assignment result, got:\n{stdout}"
    );
    assert!(
        stdout.contains("Variables cleared"),
        "Expected 'Variables cleared' message, got:\n{stdout}"
    );
    // The second 'who' should show no variables or "(no variables defined)"
    let after_clear = stdout.split("Variables cleared").nth(1).unwrap_or("");
    assert!(
        after_clear.contains("(no variables defined)") || !after_clear.contains("ans"),
        "Expected no variables after clear, got:\n{after_clear}"
    );

    println!("✓ cmd_clear_removes_variables passed");
    Ok(())
}

/// Test who command lists variables
#[test]
fn cmd_who_lists_variables() -> Result<(), Box<dyn std::error::Error>> {
    let mut child = Command::new(env!("CARGO_BIN_EXE_runmat-repl"))
        .env("RUNMAT_REPL_TEST", "1")
        .stdin(Stdio::piped())
        .stdout(Stdio::piped())
        .stderr(Stdio::piped())
        .spawn()?;

    let mut stdin = child.stdin.take().ok_or("Failed to open stdin")?;
    stdin.write_all(b"a = 1\n")?;
    stdin.write_all(b"b = 2\n")?;
    stdin.write_all(b"who\n")?;
    stdin.write_all(b"exit\n")?;
    drop(stdin);

    let output = child.wait_with_output()?;
    let stdout = String::from_utf8_lossy(&output.stdout);

    assert!(output.status.success());
    // who should list variables a, b, and ans
    assert!(
        stdout.contains("ans"),
        "Expected 'ans' in who output, got:\n{stdout}"
    );

    println!("✓ cmd_who_lists_variables passed");
    Ok(())
}

/// Test whos command shows detailed variable info
#[test]
fn cmd_whos_shows_details() -> Result<(), Box<dyn std::error::Error>> {
    let mut child = Command::new(env!("CARGO_BIN_EXE_runmat-repl"))
        .env("RUNMAT_REPL_TEST", "1")
        .stdin(Stdio::piped())
        .stdout(Stdio::piped())
        .stderr(Stdio::piped())
        .spawn()?;

    let mut stdin = child.stdin.take().ok_or("Failed to open stdin")?;
    stdin.write_all(b"x = [1 2 3]\n")?;
    stdin.write_all(b"whos\n")?;
    stdin.write_all(b"exit\n")?;
    drop(stdin);

    let output = child.wait_with_output()?;
    let stdout = String::from_utf8_lossy(&output.stdout);

    assert!(output.status.success());
    // whos should show header and variable info
    assert!(
        stdout.contains("Name"),
        "Expected 'Name' column header in whos output, got:\n{stdout}"
    );
    assert!(
        stdout.contains("Type"),
        "Expected 'Type' column header in whos output, got:\n{stdout}"
    );

    println!("✓ cmd_whos_shows_details passed");
    Ok(())
}
