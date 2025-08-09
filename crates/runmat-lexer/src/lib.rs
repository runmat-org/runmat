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
    // Prefer transpose over string at the same position; filter will skip when not after a value
    #[token("'", transpose_filter, priority = 0)]
    Transpose,
    // Full string literal; evaluated after transpose filter chooses to skip
    #[regex(r#"'([^'\n\r]|'')*'"#, string_or_emit, priority = 1)]
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
    #[token("=", |lex| { lex.extras.last_was_value = false; })]
    Assign,
    #[token(".")]
    Dot,
    // Semicolon ends a statement; next token should not be treated as a value.
    // This helps disambiguate that a following apostrophe starts a string, not a transpose.
    #[token(";", |lex| { lex.extras.last_was_value = false; })]
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
                // Robust error recovery: scan the remaining slice and emit best-effort tokens
                // so that downstream parsers can continue (e.g., identifiers, whitespace, parens).
                let s = lex.slice();
                let span = lex.span();

                let mut byte_index = 0usize; // offset within s

                while byte_index < s.len() {
                    // Helper to read next char and its byte length
                    let ch = s[byte_index..].chars().next().unwrap();
                    let ch_len = ch.len_utf8();

                    // Skip whitespace entirely (would normally be skipped by Logos attributes)
                    if ch.is_whitespace() {
                        byte_index += ch_len;
                        continue;
                    }

                    // Coalesce identifiers: [a-zA-Z_][a-zA-Z0-9_]*
                    if ch == '_' || ch.is_ascii_alphabetic() {
                        let start_off = byte_index;
                        byte_index += ch_len;
                        while byte_index < s.len() {
                            let nxt = s[byte_index..].chars().next().unwrap();
                            if nxt == '_' || nxt.is_ascii_alphanumeric() {
                                byte_index += nxt.len_utf8();
                            } else {
                                break;
                            }
                        }
                        let start = span.start + start_off;
                        let end = span.start + byte_index;
                        out.push(SpannedToken {
                            token: Token::Ident,
                            lexeme: s[start_off..byte_index].to_string(),
                            start,
                            end,
                        });
                        continue;
                    }

                    // Numbers: simplistic integer/float scan to avoid splitting
                    if ch.is_ascii_digit() {
                        let start_off = byte_index;
                        byte_index += ch_len;
                        while byte_index < s.len() {
                            let nxt = s[byte_index..].chars().next().unwrap();
                            if nxt.is_ascii_digit() {
                                byte_index += nxt.len_utf8();
                            } else if nxt == '.' {
                                // include one dot and continue scanning digits/exponent
                                byte_index += 1;
                            } else if nxt == 'e' || nxt == 'E' || nxt == '+' || nxt == '-' {
                                byte_index += 1;
                            } else {
                                break;
                            }
                        }
                        let start = span.start + start_off;
                        let end = span.start + byte_index;
                        out.push(SpannedToken {
                            token: Token::Integer, // good enough for recovery; detailed kind not required
                            lexeme: s[start_off..byte_index].to_string(),
                            start,
                            end,
                        });
                        continue;
                    }

                    // Single-character punctuation/operators
                    let token = match ch {
                        '\'' => {
                            // In recovery, only treat apostrophe as transpose when the previous token
                            // was a value; otherwise it's likely a broken string start -> mark as error.
                            if lex.extras.last_was_value {
                                Token::Transpose
                            } else {
                                Token::Error
                            }
                        }
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

                    let start = span.start + byte_index;
                    let end = start + ch_len;
                    out.push(SpannedToken {
                        token,
                        lexeme: ch.to_string(),
                        start,
                        end,
                    });
                    byte_index += ch_len;
                }
            }
        }
    }
    out
}

// If regex matched but it's actually an unterminated quote (no closing '),
// tell Logos to Skip so the single-quote token can be picked up as Transpose.
fn string_or_emit(lexer: &mut Lexer<Token>) -> Filter<()> {
    // Full strings are unambiguous; mark as value and emit.
    lexer.extras.last_was_value = true;
    Filter::Emit(())
}

fn transpose_filter(lex: &mut Lexer<Token>) -> Filter<()> {
    // Emit transpose only when the previous token formed a value
    // (e.g., after identifiers, numbers, closing parens/brackets/braces, etc.).
    // Otherwise, skip so that the Str token (full quoted string) can match.
    if lex.extras.last_was_value {
        lex.extras.last_was_value = true;
        Filter::Emit(())
    } else {
        Filter::Skip
    }
}
