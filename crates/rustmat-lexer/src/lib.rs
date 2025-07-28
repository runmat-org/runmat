use logos::Logos;

#[derive(Logos, Debug, PartialEq)]
#[logos(skip r"[ \t\n\r]+")]
#[logos(skip r"%[^\n]*")]
pub enum Token {
    // Keywords
    #[token("function")]
    Function,
    #[token("if")]
    If,
    #[token("else")]
    Else,
    #[token("elseif")]
    ElseIf,
    #[token("for")]
    For,
    #[token("while")]
    While,
    #[token("break")]
    Break,
    #[token("continue")]
    Continue,
    #[token("return")]
    Return,
    #[token("end")]
    End,

    // Identifiers and literals
    #[regex(r"[a-zA-Z_][a-zA-Z0-9_]*")]
    Ident,
    #[regex(r"\d+\.\d+([eE][+-]?\d+)?")]
    Float,
    #[regex(r"\d+([eE][+-]?\d+)?")]
    Integer,
    #[regex(r#"'([^']|'')*'"#)]
    Str,

    // Operators and punctuation (longest tokens first)
    #[token("...")]
    Ellipsis,
    #[token(".*")]
    DotStar,
    #[token("./")]
    DotSlash,
    #[token(".\\")]
    DotBackslash,
    #[token(".^")]
    DotCaret,
    #[token("&&")]
    AndAnd,
    #[token("||")]
    OrOr,
    #[token("==")]
    Equal,
    #[token("~=")]
    NotEqual,
    #[token("<=")]
    LessEqual,
    #[token(">=")]
    GreaterEqual,
    #[token("+")]
    Plus,
    #[token("-")]
    Minus,
    #[token("*")]
    Star,
    #[token("/")]
    Slash,
    #[token("\\")]
    Backslash,
    #[token("^")]
    Caret,
    #[token("&")]
    And,
    #[token("|")]
    Or,
    #[token("~")]
    Tilde,
    #[token("<")]
    Less,
    #[token(">")]
    Greater,
    #[token("=")]
    Assign,
    #[token(".")]
    Dot,
    #[token(";")]
    Semicolon,
    #[token(",")]
    Comma,
    #[token(":")]
    Colon,
    #[token("(")]
    LParen,
    #[token(")")]
    RParen,
    #[token("[")]
    LBracket,
    #[token("]")]
    RBracket,
    #[token("{")]
    LBrace,
    #[token("}")]
    RBrace,

    Error,
}

pub fn tokenize(input: &str) -> Vec<Token> {
    Token::lexer(input)
        .map(|res| res.unwrap_or(Token::Error))
        .collect()
}
