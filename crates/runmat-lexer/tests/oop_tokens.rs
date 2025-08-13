use runmat_lexer::{tokenize, Token};

#[test]
fn classdef_and_oop_keywords() {
    let src = "classdef A < handle properties end methods end events end enumeration end";
    assert_eq!(
        tokenize(src),
        vec![
            Token::ClassDef,
            Token::Ident,
            Token::Less,
            Token::Ident,
            Token::Properties,
            Token::End,
            Token::Methods,
            Token::End,
            Token::Events,
            Token::End,
            Token::Enumeration,
            Token::End,
        ]
    );
}

#[test]
fn function_handle_tokens() {
    let src = "@(x) x^2 @sin";
    assert_eq!(
        tokenize(src),
        vec![
            Token::At,
            Token::LParen,
            Token::Ident,
            Token::RParen,
            Token::Ident,
            Token::Caret,
            Token::Integer,
            Token::At,
            Token::Ident,
        ]
    );
}


