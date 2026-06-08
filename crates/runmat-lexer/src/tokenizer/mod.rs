mod recovery;

use crate::scan::{last_is_value_token, scan_leading_dot_float_suffix};
use crate::token::{SpannedToken, Token};
use logos::Logos;
use recovery::recover_from_error;

pub fn tokenize(input: &str) -> Vec<Token> {
    tokenize_detailed(input)
        .into_iter()
        .map(|t| t.token)
        .filter(|tok| !matches!(tok, Token::Newline))
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

                if matches!(tok, Token::Dot) {
                    let prev_is_adjacent_value = out
                        .last()
                        .map(|t| t.end == span.start && last_is_value_token(&t.token))
                        .unwrap_or(false);
                    if !prev_is_adjacent_value {
                        let rem = lex.remainder();
                        if let Some(len) = scan_leading_dot_float_suffix(rem) {
                            let mut lexeme = format!(".{}", &rem[..len]);
                            lexeme.retain(|c| c != '_');
                            lex.bump(len);
                            lex.extras.last_was_value = true;
                            lex.extras.line_start = false;
                            out.push(SpannedToken {
                                token: Token::Float,
                                lexeme,
                                start: span.start,
                                end: span.end + len,
                            });
                            continue;
                        }
                    }
                }

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
            Err(_) => recover_from_error(&mut lex, &mut out),
        }
    }
    out
}
