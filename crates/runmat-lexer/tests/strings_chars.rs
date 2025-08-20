use runmat_lexer::{tokenize, Token};

#[test]
fn single_and_double_quoted_strings() {
    // Single-quoted char array with escaped quote, and a simple double-quoted string scalar
    let src = r#"'a''b' "hello""#;
    assert_eq!(tokenize(src), vec![Token::Str, Token::Str]);
}

#[test]
fn transpose_then_single_quoted_string_after_semicolon() {
    // After a value, ' is transpose; after a semicolon, a single-quoted string should be recognized
    let src = "A'; 'text'";
    assert_eq!(
        tokenize(src),
        vec![Token::Ident, Token::Transpose, Token::Semicolon, Token::Str]
    );
}
