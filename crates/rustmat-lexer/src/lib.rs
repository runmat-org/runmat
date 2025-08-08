use logos::{Filter, Lexer, Logos};

#[derive(Default, Clone, Copy)]
pub struct LexerExtras {
    pub last_was_value: bool,
}

#[derive(Logos, Debug, PartialEq, Clone)]
#[logos(skip r"[ \t\n\r]+")]
#[logos(skip r"%[^\n]*")]
#[logos(extras = LexerExtras)]
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
    #[regex(r"[a-zA-Z_][a-zA-Z0-9_]*", |lex| { lex.extras.last_was_value = true; })]
    Ident,
    #[regex(r"\d+\.\d+([eE][+-]?\d+)?", |lex| { lex.extras.last_was_value = true; })]
    Float,
    #[regex(r"\d+([eE][+-]?\d+)?", |lex| { lex.extras.last_was_value = true; })]
    Integer,
    // Transpose first with higher priority to avoid greedy partial string scans swallowing it
    #[token("'", transpose_filter, priority = 10)]
    Transpose,
    #[regex(r#"'([^'\n\r]|'')*'"#, string_or_skip, priority = 1)]
    Str,
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
    #[token(";", |lex| { lex.extras.last_was_value = true; })]
    Semicolon,
    #[token(",")]
    Comma,
    #[token(":")]
    Colon,
    #[token("(", |lex| { lex.extras.last_was_value = false; })]
    LParen,
    #[token(")", |lex| { lex.extras.last_was_value = true; })]
    RParen,
    #[token("[", |lex| { lex.extras.last_was_value = false; })]
    LBracket,
    #[token("]", |lex| { lex.extras.last_was_value = true; })]
    RBracket,
    #[token("{", |lex| { lex.extras.last_was_value = false; })]
    LBrace,
    #[token("}", |lex| { lex.extras.last_was_value = true; })]
    RBrace,

    Error,
}

#[derive(Debug, Clone, PartialEq)]
pub struct SpannedToken {
    pub token: Token,
    pub lexeme: String,
    pub start: usize,
    pub end: usize,
}

pub fn tokenize(input: &str) -> Vec<Token> {
    tokenize_detailed(input)
        .into_iter()
        .map(|t| t.token)
        .collect()
}

pub fn tokenize_detailed(input: &str) -> Vec<SpannedToken> {
    let mut lex = Token::lexer(input);
    let mut out: Vec<SpannedToken> = Vec::new();
    while let Some(res) = lex.next() {
        match res {
            Ok(tok) => {
                let s = lex.slice().to_string();
                let span = lex.span();
                out.push(SpannedToken {
                    token: tok,
                    lexeme: s,
                    start: span.start,
                    end: span.end,
                });
            }
            Err(_) => {
                let s = lex.slice();
                let span = lex.span();
                for (off, ch) in s.char_indices() {
                    let token = match ch {
                        '\'' => Token::Transpose,
                        ';' => Token::Semicolon,
                        ')' => Token::RParen,
                        '(' => Token::LParen,
                        ',' => Token::Comma,
                        ']' => Token::RBracket,
                        '[' => Token::LBracket,
                        '}' => Token::RBrace,
                        '{' => Token::LBrace,
                        ':' => Token::Colon,
                        '.' => Token::Dot,
                        '+' => Token::Plus,
                        '-' => Token::Minus,
                        '*' => Token::Star,
                        '/' => Token::Slash,
                        '\\' => Token::Backslash,
                        '^' => Token::Caret,
                        '&' => Token::And,
                        '|' => Token::Or,
                        '~' => Token::Tilde,
                        '<' => Token::Less,
                        '>' => Token::Greater,
                        '=' => Token::Assign,
                        _ => Token::Error,
                    };
                    let start = span.start + off;
                    let end = start + ch.len_utf8();
                    out.push(SpannedToken {
                        token,
                        lexeme: ch.to_string(),
                        start,
                        end,
                    });
                }
            }
        }
    }
    out
}

// If regex matched but it's actually an unterminated quote (no closing '),
// tell Logos to Skip so the single-quote token can be picked up as Transpose.
fn string_or_skip(_lexer: &mut Lexer<Token>) -> Filter<()> {
    // Regex ensures we only get here for complete strings. Emit unit.
    Filter::Emit(())
}

fn transpose_filter(_lex: &mut Lexer<Token>) -> Filter<()> {
    // Always emit transpose for a single apostrophe.
    // Full strings like 'text' are matched by the longer Str regex and will take precedence.
    Filter::Emit(())
}
