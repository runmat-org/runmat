use runmat_lexer::{tokenize, Token};

#[test]
fn logical_and_elementwise_operators() {
    let src = "a && b || c & d | e ~f";
    assert_eq!(
        tokenize(src),
        vec![
            Token::Ident,
            Token::AndAnd,
            Token::Ident,
            Token::OrOr,
            Token::Ident,
            Token::And,
            Token::Ident,
            Token::Or,
            Token::Ident,
            Token::Tilde,
            Token::Ident,
        ]
    );
}

#[test]
fn dot_plus_and_dot_minus_split() {
    let src = "1 .+ 2 .- 3";
    assert_eq!(
        tokenize(src),
        vec![
            Token::Integer,
            Token::Dot,
            Token::Plus,
            Token::Integer,
            Token::Dot,
            Token::Minus,
            Token::Integer,
        ]
    );
}

#[test]
fn elementwise_mul_div_pow_tokens() {
    let src = ".* ./ .\\ .^";
    assert_eq!(
        tokenize(src),
        vec![Token::DotStar, Token::DotSlash, Token::DotBackslash, Token::DotCaret]
    );
}


