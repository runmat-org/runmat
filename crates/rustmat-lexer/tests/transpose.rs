use rustmat_lexer::{tokenize, Token};

#[test]
fn transpose_after_ident_then_semicolon() {
    let input = "B = A';";
    let tokens = tokenize(input);
    // Expect: Ident(B), Assign, Ident(A), Transpose, Semicolon
    assert_eq!(tokens, vec![
        Token::Ident,
        Token::Assign,
        Token::Ident,
        Token::Transpose,
        Token::Semicolon,
    ]);
}

#[test]
fn simple_string_literal() {
    let input = "fprintf('done');";
    let tokens = tokenize(input);
    // Expect: Ident, LParen, Str, RParen, Semicolon
    assert_eq!(tokens, vec![
        Token::Ident,
        Token::LParen,
        Token::Str,
        Token::RParen,
        Token::Semicolon,
    ]);
}

#[test]
fn debug_print_tokens_for_apostrophe_case() {
    let input = "B = A';";
    let tokens = tokenize(input);
    println!("TOKENS: {:?}", tokens);
}