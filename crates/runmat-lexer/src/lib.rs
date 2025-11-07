use logos::{Filter, Lexer, Logos};

#[derive(Default, Clone, Copy)]
pub struct LexerExtras {
    pub last_was_value: bool,
    pub line_start: bool,
}

#[derive(Logos, Debug, PartialEq, Clone)]
// Skip spaces, tabs and carriage returns, but NOT newlines; we need newlines to detect '%%' at line start
#[logos(skip r"[ \t\r]+")]
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

    // Object-oriented and function syntax keywords
    #[token("classdef")]
    ClassDef,
    #[token("properties")]
    Properties,
    #[token("methods")]
    Methods,
    #[token("events")]
    Events,
    #[token("enumeration")]
    Enumeration,
    #[token("arguments")]
    Arguments,

    // Importing packages/classes
    #[token("import")]
    Import,

    // Additional keywords (recognized by lexer; parser may treat as identifiers for now)
    #[token("switch")]
    Switch,
    #[token("case")]
    Case,
    #[token("otherwise")]
    Otherwise,
    #[token("try")]
    Try,
    #[token("catch")]
    Catch,
    #[token("global")]
    Global,
    #[token("persistent")]
    Persistent,
    #[token("true", |lex| { lex.extras.last_was_value = true; })]
    True,
    #[token("false", |lex| { lex.extras.last_was_value = true; })]
    False,

    // Identifiers and literals
    #[regex(r"[a-zA-Z_][a-zA-Z0-9_]*", |lex| { lex.extras.last_was_value = true; })]
    Ident,
    // Float with optional underscores as digit separators (strip later)
    #[regex(r"\d(?:_?\d)*\.(?:\d(?:_?\d)*)?(?:[eE][+-]?\d(?:_?\d)*)?", |lex| {
        lex.extras.last_was_value = true;
    })]
    #[regex(r"\d(?:_?\d)*[eE][+-]?\d(?:_?\d)*", |lex| {
        lex.extras.last_was_value = true;
    })]
    Float,
    // Integer with optional underscores as digit separators (strip later)
    #[regex(r"\d(?:_?\d)*", |lex| {
        lex.extras.last_was_value = true;
    })]
    Integer,
    // Apostrophe is handled contextually in tokenize_detailed: either Transpose or a single-quoted string
    #[token("'")]
    Apostrophe,
    // Double-quoted string scalar (treated as Str at lexer level). Always emit.
    #[regex(r#""([^"\n\r]|"")*""#, double_quoted_string_emit, priority = 1)]
    Str,
    #[token("...", ellipsis_emit_and_skip_to_eol)]
    Ellipsis,
    // Section marker: must be at start of line (after optional whitespace). We match until EOL and emit a single token.
    #[regex(r"%%[^\n]*", section_marker, priority = 3)]
    Section,
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
    #[token("@")]
    At,
    // Meta-class (type) query operator: ?ClassName
    #[token("?")]
    Question,
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

    // Newlines are skipped but set line_start for '%%' detection
    #[regex(r"\n+", newline_skip)]
    Newline,

    // Block comments: '%{' ... '%}' (non-nesting). Skipped entirely.
    #[regex(r"%\{", block_comment_skip, priority = 2)]
    BlockComment,

    // Line comments: single '%' handled here; '%%' and '%{' are matched by other rules first
    #[token("%", line_comment_start, priority = 0)]
    LineComment,

    Error,
    // Synthetic tokens (not produced by Logos directly)
    Transpose,
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
    // We begin at the start of a (virtual) line
    lex.extras.line_start = true;
    let mut out: Vec<SpannedToken> = Vec::new();
    while let Some(res) = lex.next() {
        match res {
            Ok(tok) => {
                let mut s = lex.slice().to_string();
                // Normalize numeric literals: remove underscores in integers/floats
                if matches!(tok, Token::Float | Token::Integer) {
                    s.retain(|c| c != '_');
                }
                let span = lex.span();

                // Handle contextual apostrophe before normal push logic
                if matches!(tok, Token::Apostrophe) {
                    // Decide using adjacency + previous token category.
                    // Transpose only when there is no whitespace between and previous token is a value or dot.
                    let (is_adjacent, prev_token_opt) = out
                        .last()
                        .map(|t| (t.end == span.start, Some(&t.token)))
                        .unwrap_or((false, None));
                    let prev_is_value_or_dot = prev_token_opt
                        .map(|t| matches!(t, Token::Dot) || last_is_value_token(t))
                        .unwrap_or(false);
                    if is_adjacent && prev_is_value_or_dot {
                        out.push(SpannedToken {
                            token: Token::Transpose,
                            lexeme: "'".into(),
                            start: span.start,
                            end: span.end,
                        });
                        continue;
                    }
                    // Otherwise, parse a full single-quoted string starting at this apostrophe
                    let rem = lex.remainder();
                    let mut j = 0usize;
                    let bytes = rem.as_bytes();
                    let mut ok = false;
                    while j < rem.len() {
                        let c = bytes[j] as char;
                        if c == '\'' {
                            if j + 1 < rem.len() && bytes[j + 1] as char == '\'' {
                                j += 2; // escaped quote
                            } else {
                                ok = true; // closing quote
                                j += 1; // include closing
                                break;
                            }
                        } else if c == '\n' || c == '\r' {
                            break;
                        } else {
                            j += 1;
                        }
                    }
                    if ok {
                        // Consume what we scanned and emit Str for the entire single-quoted literal
                        let abs_start = span.start;
                        let abs_end = span.end + j;
                        let lexeme = format!("'{}", &rem[..j]);
                        lex.bump(j); // advance past the content following the leading apostrophe
                        lex.extras.last_was_value = true;
                        out.push(SpannedToken {
                            token: Token::Str,
                            lexeme,
                            start: abs_start,
                            end: abs_end,
                        });
                        lex.extras.line_start = false;
                        continue;
                    } else {
                        // Unterminated; treat as Error
                        out.push(SpannedToken {
                            token: Token::Error,
                            lexeme: "'".into(),
                            start: span.start,
                            end: span.end,
                        });
                        continue;
                    }
                }
                // On any emitted token that is not Newline or comment/skip, we are no longer at line start
                match tok {
                    Token::Newline | Token::LineComment | Token::BlockComment => {}
                    _ => {
                        lex.extras.line_start = false;
                    }
                }
                out.push(SpannedToken {
                    token: tok,
                    lexeme: s,
                    start: span.start,
                    end: span.end,
                });

                // Special-case: immediately after a semicolon, allow a single-quoted string literal
                // to be parsed eagerly to avoid apostrophe/transpose ambiguity.
                if matches!(out.last().map(|t| &t.token), Some(Token::Semicolon)) {
                    // Peek the remainder for optional whitespace + a single-quoted string
                    let rem = lex.remainder();
                    let mut offset = 0usize;
                    for ch in rem.chars() {
                        if ch == ' ' || ch == '\t' || ch == '\r' {
                            offset += ch.len_utf8();
                        } else {
                            break;
                        }
                    }
                    if rem[offset..].starts_with('\'') {
                        // Try to scan a valid single-quoted string with doubled '' escapes
                        let mut j = offset + 1;
                        let bytes = rem.as_bytes();
                        let mut ok = false;
                        while j < rem.len() {
                            let c = bytes[j] as char;
                            if c == '\'' {
                                if j + 1 < rem.len() && bytes[j + 1] as char == '\'' {
                                    j += 2; // escaped quote
                                } else {
                                    ok = true; // closing quote at j
                                    j += 1;
                                    break;
                                }
                            } else if c == '\n' {
                                break;
                            } else {
                                j += 1;
                            }
                        }
                        if ok {
                            // Consume the scanned slice and emit a Str token
                            let abs_start = span.end + offset;
                            let abs_end = span.end + j;
                            let lexeme = &rem[offset..j];
                            lex.bump(j); // advance lexer past the string
                            lex.extras.last_was_value = true;
                            out.push(SpannedToken {
                                token: Token::Str,
                                lexeme: lexeme.to_string(),
                                start: abs_start,
                                end: abs_end,
                            });
                        }
                    }
                }
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

                    // Double-quoted string recovery: "..." with doubled "" escapes
                    if ch == '"' {
                        let start_off = byte_index;
                        byte_index += ch_len; // consume opening quote
                        while byte_index < s.len() {
                            let nxt = s[byte_index..].chars().next().unwrap();
                            if nxt == '"' {
                                // Check for doubled quote escape
                                let next_two = &s[byte_index..];
                                if next_two.starts_with("\"\"") {
                                    // consume both quotes as escaped quote
                                    byte_index += 2;
                                    continue;
                                } else {
                                    // closing quote
                                    byte_index += 1;
                                    break;
                                }
                            } else if nxt == '\n' || nxt == '\r' {
                                // Unterminated; emit error for opening quote and break to resume normal scan
                                let start = span.start + start_off;
                                out.push(SpannedToken {
                                    token: Token::Error,
                                    lexeme: s[start_off..start_off + 1].to_string(),
                                    start,
                                    end: start + 1,
                                });
                                // do not advance byte_index beyond the opening quote; let normal flow handle following chars
                                break;
                            } else {
                                byte_index += nxt.len_utf8();
                            }
                        }
                        // If we ended on a closing quote, emit Str token
                        if byte_index > start_off + 1
                            && &s[start_off..start_off + 1] == "\""
                            && s[start_off..byte_index].ends_with('"')
                        {
                            let start = span.start + start_off;
                            let end = span.start + byte_index;
                            // Mark as value for downstream transpose logic
                            lex.extras.last_was_value = true;
                            out.push(SpannedToken {
                                token: Token::Str,
                                lexeme: s[start_off..byte_index].to_string(),
                                start,
                                end,
                            });
                            continue;
                        } else {
                            // If not properly closed, fall through; single-char error was already emitted
                            byte_index += ch_len;
                            continue;
                        }
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

#[allow(dead_code)]
fn last_is_value_token(tok: &Token) -> bool {
    matches!(
        tok,
        Token::Ident
            | Token::Integer
            | Token::Float
            | Token::True
            | Token::False
            | Token::RParen
            | Token::RBracket
            | Token::RBrace
            | Token::Str
    )
}

fn double_quoted_string_emit(lexer: &mut Lexer<Token>) -> Filter<()> {
    // Always emit and mark as value
    lexer.extras.last_was_value = true;
    Filter::Emit(())
}

#[allow(dead_code)]
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

fn ellipsis_emit_and_skip_to_eol(lex: &mut Lexer<Token>) -> Filter<()> {
    // After an ellipsis, ignore the remainder of the physical line (including comments)
    let rest = lex.remainder();
    if let Some(pos) = rest.find('\n') {
        lex.bump(pos); // position to before the newline; newline token will consume it
    } else {
        lex.bump(rest.len());
    }
    // Ellipsis itself is a token and is considered to be in an expression context
    lex.extras.last_was_value = true; // e.g., '1 + ...\n 2' the ellipsis does not reset value-ness
    Filter::Emit(())
}

fn newline_skip(lex: &mut Lexer<Token>) -> Filter<()> {
    lex.extras.line_start = true;
    lex.extras.last_was_value = false;
    Filter::Emit(())
}

fn section_marker(lex: &mut Lexer<Token>) -> Filter<()> {
    // Only emit a Section token when at start of line; otherwise, treat as a comment and skip
    if lex.extras.line_start {
        lex.extras.line_start = false;
        lex.extras.last_was_value = false;
        Filter::Emit(())
    } else {
        // Skip to end of line (already consumed by regex except for the newline char)
        Filter::Skip
    }
}

// Removed: replaced by explicit line_comment_start which consumes from single '%'

fn block_comment_skip(lex: &mut Lexer<Token>) -> Filter<()> {
    // We matched '%{'. Skip until the first '%}' or end of input.
    let rest = lex.remainder();
    if let Some(end) = rest.find("%}") {
        lex.bump(end + 2); // consume up to and including '%}'
    } else {
        lex.bump(rest.len()); // consume to end if no terminator
    }
    Filter::Skip
}

fn line_comment_start(lex: &mut Lexer<Token>) -> Filter<()> {
    // We just consumed a single '%'. Skip to the end of the line.
    let rest = lex.remainder();
    if let Some(pos) = rest.find('\n') {
        lex.bump(pos);
    } else {
        lex.bump(rest.len());
    }
    Filter::Skip
}
