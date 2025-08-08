use runmat_lexer::{tokenize, Token};

#[test]
fn keywords() {
    let src = "function if else elseif for while break continue return end";
    let tokens = tokenize(src);
    assert_eq!(
        tokens,
        vec![
            Token::Function,
            Token::If,
            Token::Else,
            Token::ElseIf,
            Token::For,
            Token::While,
            Token::Break,
            Token::Continue,
            Token::Return,
            Token::End,
        ]
    );
}

#[test]
fn identifiers_and_numbers() {
    let src = "foo bar123 42 3.14";
    let tokens = tokenize(src);
    assert_eq!(
        tokens,
        vec![Token::Ident, Token::Ident, Token::Integer, Token::Float,]
    );
}

#[test]
fn string_literal() {
    let src = "'hello world'";
    let tokens = tokenize(src);
    assert_eq!(tokens, vec![Token::Str]);
}

#[test]
fn operators() {
    let src = "+ - * / \\ ^ .* ./ .\\ .^ && || & | ~ == ~= <= >= < > = . ; , : ( ) [ ] { } ...";
    let tokens = tokenize(src);
    assert_eq!(
        tokens,
        vec![
            Token::Plus,
            Token::Minus,
            Token::Star,
            Token::Slash,
            Token::Backslash,
            Token::Caret,
            Token::DotStar,
            Token::DotSlash,
            Token::DotBackslash,
            Token::DotCaret,
            Token::AndAnd,
            Token::OrOr,
            Token::And,
            Token::Or,
            Token::Tilde,
            Token::Equal,
            Token::NotEqual,
            Token::LessEqual,
            Token::GreaterEqual,
            Token::Less,
            Token::Greater,
            Token::Assign,
            Token::Dot,
            Token::Semicolon,
            Token::Comma,
            Token::Colon,
            Token::LParen,
            Token::RParen,
            Token::LBracket,
            Token::RBracket,
            Token::LBrace,
            Token::RBrace,
            Token::Ellipsis,
        ]
    );
}

#[test]
fn comments_and_whitespace() {
    let src = "foo % comment\n+";
    let tokens = tokenize(src);
    assert_eq!(tokens, vec![Token::Ident, Token::Plus]);
}

#[test]
fn uppercase_keywords_are_identifiers() {
    let src = "IF ELSE";
    let tokens = tokenize(src);
    assert_eq!(tokens, vec![Token::Ident, Token::Ident]);
}

#[test]
fn unknown_character_yields_error() {
    let src = "$";
    let tokens = tokenize(src);
    assert_eq!(tokens, vec![Token::Error]);
}

#[test]
fn unterminated_string_produces_error() {
    let src = "'oops";
    let tokens = tokenize(src);
    assert!(tokens.contains(&Token::Error));
}
