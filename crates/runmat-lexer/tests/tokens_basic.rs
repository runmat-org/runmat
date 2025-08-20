use runmat_lexer::{tokenize, Token};

#[test]
fn identifiers_and_numbers() {
    let src = "x 123 4.56";
    assert_eq!(
        tokenize(src),
        vec![Token::Ident, Token::Integer, Token::Float]
    );
}

#[test]
fn true_false_keywords() {
    let src = "true false";
    assert_eq!(tokenize(src), vec![Token::True, Token::False]);
}

#[test]
fn global_and_persistent_keywords() {
    let src = "global persistent";
    assert_eq!(tokenize(src), vec![Token::Global, Token::Persistent]);
}

#[test]
fn switch_case_otherwise_try_catch_keywords() {
    let src = "switch case otherwise try catch end";
    assert_eq!(
        tokenize(src),
        vec![
            Token::Switch,
            Token::Case,
            Token::Otherwise,
            Token::Try,
            Token::Catch,
            Token::End,
        ]
    );
}

#[test]
fn semicolon_and_comma_tokens() {
    let src = "a=1; b=2, c=3";
    assert_eq!(
        tokenize(src),
        vec![
            Token::Ident,
            Token::Assign,
            Token::Integer,
            Token::Semicolon,
            Token::Ident,
            Token::Assign,
            Token::Integer,
            Token::Comma,
            Token::Ident,
            Token::Assign,
            Token::Integer,
        ]
    );
}

#[test]
fn ans_is_identifier() {
    let src = "ans = 1";
    assert_eq!(
        tokenize(src),
        vec![Token::Ident, Token::Assign, Token::Integer]
    );
}
