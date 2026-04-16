use crate::token::{SpannedToken, Token};
use logos::Lexer;

pub(super) fn recover_from_error(lex: &mut Lexer<Token>, out: &mut Vec<SpannedToken>) {
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
