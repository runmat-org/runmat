// None of these tests use #[wasm_bindgen_test], so they cannot run in the
// browser via wasm-pack. Excluding them from wasm32 avoids compiling a full
// runmat-runtime wasm binary per test file with zero executable tests.
#![cfg(not(target_arch = "wasm32"))]

fn format_tokens(input: &str) -> String {
    runmat_lexer::tokenize_detailed(input)
        .into_iter()
        .map(|t| format!("{:?}", t.token))
        .collect::<Vec<_>>()
        .join(" ")
}

#[test]
fn tokenize_simple_input() {
    let result = format_tokens("x = 1 + 2;");
    assert_eq!(result, "Ident Assign Integer Plus Integer Semicolon");
}

#[test]
fn handles_whitespace_and_comments() {
    let result = format_tokens("foo % comment\n+");
    assert_eq!(result, "Ident LineComment Newline Plus");
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
    assert_eq!(result.split_whitespace().next(), Some("Error"));
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
