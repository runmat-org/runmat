use runmat_repl::format_tokens;
use std::io::Write;
use std::process::{Command, Stdio};

#[test]
fn tokenize_simple_input() {
    let result = format_tokens("x = 1 + 2;");
    assert_eq!(result, "Ident Assign Integer Plus Integer Semicolon");
}

#[test]
fn handles_whitespace_and_comments() {
    let result = format_tokens("foo % comment\n+");
    assert_eq!(result, "Ident Plus");
}

#[test]
fn empty_input_yields_empty_string() {
    let result = format_tokens("");
    assert!(result.is_empty());
}

#[test]
fn unknown_char_produces_error() {
    let result = format_tokens("$");
    assert_eq!(result, "Error");
}

#[test]
fn unterminated_string_is_error_token() {
    let result = format_tokens("'oops");
    assert!(result.contains("Error"));
}

#[test]
fn keywords_are_case_sensitive() {
    let result = format_tokens("IF ELSE");
    assert_eq!(result, "Ident Ident");
}

#[test]
fn complex_expression_tokens() {
    let result = format_tokens("1 * (2 + 3)");
    assert_eq!(result, "Integer Star LParen Integer Plus Integer RParen");
}

#[test]
fn multiple_statements() {
    let result = format_tokens("x=1; y=2;");
    assert_eq!(
        result,
        "Ident Assign Integer Semicolon Ident Assign Integer Semicolon"
    );
}

#[test]
fn repl_binary_processes_single_line() -> Result<(), Box<dyn std::error::Error>> {
    let mut child = Command::new(env!("CARGO_BIN_EXE_runmat-repl"))
        .stdin(Stdio::piped())
        .stdout(Stdio::piped())
        .spawn()?;
    child.stdin.as_mut().unwrap().write_all(b"x=1\n")?;
    let output = child.wait_with_output()?;
    assert!(output.status.success());
    let stdout = String::from_utf8_lossy(&output.stdout);
    // Should contain REPL banner and show result of x=1 execution
    assert!(stdout.contains("RunMat REPL"));
    assert!(stdout.contains("runmat>"));
    assert!(stdout.contains("ans = 1")); // Result of x=1
    Ok(())
}
