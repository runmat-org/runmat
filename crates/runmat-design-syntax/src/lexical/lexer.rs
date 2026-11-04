use super::{
    FeaKeyword, FeaLexDiagnostic, FeaLexDiagnosticKind, FeaToken, FeaTokenKind, FeaTokenStream,
    FeaTriviaKind,
};
use runmat_types::Span;

pub fn tokenize_fea(source: impl Into<String>) -> FeaTokenStream {
    FeaLexer::new(source.into(), false).tokenize()
}

/// Tokenizes a design-dialect source (`.part`, `.assembly`, `.drawing`, `.circuit`,
/// `.circuit.layout`). It adds the design value punctuation to the `.fea` tokens: `±` and
/// `+/-` (a tolerance), `@` (a pin) and `%` (the percent unit). In `.fea` sources these
/// characters stay invalid.
pub fn tokenize_design(source: impl Into<String>) -> FeaTokenStream {
    FeaLexer::new(source.into(), true).tokenize()
}

struct FeaLexer {
    design: bool,
    source: String,
    cursor: usize,
    tokens: Vec<FeaToken>,
    diagnostics: Vec<FeaLexDiagnostic>,
}

impl FeaLexer {
    fn new(source: String, design: bool) -> Self {
        Self {
            design,
            source,
            cursor: 0,
            tokens: Vec::new(),
            diagnostics: Vec::new(),
        }
    }

    fn tokenize(mut self) -> FeaTokenStream {
        while self.cursor < self.source.len() {
            self.scan_token();
        }
        self.tokens.push(FeaToken {
            kind: FeaTokenKind::EndOfFile,
            span: Span {
                start: self.source.len(),
                end: self.source.len(),
            },
        });
        FeaTokenStream {
            source: self.source,
            tokens: self.tokens,
            diagnostics: self.diagnostics,
        }
    }

    fn scan_token(&mut self) {
        let start = self.cursor;
        let Some(current) = self.peek() else {
            return;
        };
        match current {
            ' ' | '\t' | '\r' => {
                self.advance();
                while matches!(self.peek(), Some(' ' | '\t' | '\r')) {
                    self.advance();
                }
                self.push(FeaTokenKind::Trivia(FeaTriviaKind::Whitespace), start);
            }
            '\n' => {
                self.advance();
                self.push(FeaTokenKind::Trivia(FeaTriviaKind::Newline), start);
            }
            '/' if self.peek_second() == Some('/') => {
                self.advance();
                self.advance();
                while !matches!(self.peek(), None | Some('\n')) {
                    self.advance();
                }
                self.push(FeaTokenKind::Trivia(FeaTriviaKind::Comment), start);
            }
            '"' | '\'' => self.scan_string(current, start),
            value if value.is_ascii_digit() => self.scan_number(start),
            '.' if self
                .peek_second()
                .is_some_and(|value| value.is_ascii_digit()) =>
            {
                self.scan_number(start)
            }
            value if is_identifier_start(value) => self.scan_identifier(start),
            '{' => self.single(FeaTokenKind::LeftBrace, start),
            '}' => self.single(FeaTokenKind::RightBrace, start),
            '[' => self.single(FeaTokenKind::LeftBracket, start),
            ']' => self.single(FeaTokenKind::RightBracket, start),
            '(' => self.single(FeaTokenKind::LeftParen, start),
            ')' => self.single(FeaTokenKind::RightParen, start),
            ':' => self.single(FeaTokenKind::Colon, start),
            ',' => self.single(FeaTokenKind::Comma, start),
            ';' => self.single(FeaTokenKind::Semicolon, start),
            '+' if self.design && self.source[self.cursor..].starts_with("+/-") => {
                self.cursor += 3;
                self.push(FeaTokenKind::PlusMinus, start);
            }
            '+' => self.single(FeaTokenKind::Plus, start),
            '-' => self.single(FeaTokenKind::Minus, start),
            '*' => self.single(FeaTokenKind::Star, start),
            '/' => self.single(FeaTokenKind::Slash, start),
            '^' => self.single(FeaTokenKind::Caret, start),
            '±' if self.design => self.single(FeaTokenKind::PlusMinus, start),
            '@' if self.design => self.single(FeaTokenKind::At, start),
            '%' if self.design => self.single(FeaTokenKind::Percent, start),
            '.' if self.peek_second() == Some('.') => {
                self.advance();
                self.advance();
                self.push(FeaTokenKind::Range, start);
            }
            '.' => self.single(FeaTokenKind::Dot, start),
            '=' => self.one_or_two(start, '=', FeaTokenKind::EqualEqual, FeaTokenKind::Equal),
            '!' if self.peek_second() == Some('=') => {
                self.advance();
                self.advance();
                self.push(FeaTokenKind::BangEqual, start);
            }
            '<' => self.one_or_two(start, '=', FeaTokenKind::LessEqual, FeaTokenKind::Less),
            '>' => self.one_or_two(
                start,
                '=',
                FeaTokenKind::GreaterEqual,
                FeaTokenKind::Greater,
            ),
            _ => {
                self.advance();
                self.push(FeaTokenKind::Invalid, start);
                self.diagnostics.push(FeaLexDiagnostic {
                    kind: FeaLexDiagnosticKind::InvalidCharacter,
                    span: Span {
                        start,
                        end: self.cursor,
                    },
                    message: if self.design {
                        format!("invalid character `{current}` in design source")
                    } else {
                        format!("invalid character `{current}` in FEA source")
                    },
                });
            }
        }
    }

    fn scan_string(&mut self, quote: char, start: usize) {
        self.advance();
        let mut terminated = false;
        while let Some(current) = self.peek() {
            if current == quote {
                self.advance();
                terminated = true;
                break;
            }
            if current == '\n' {
                break;
            }
            if current == '\\' {
                let escape_start = self.cursor;
                self.advance();
                match self.peek() {
                    Some('\\' | '"' | '\'' | 'n' | 'r' | 't') => {
                        self.advance();
                    }
                    Some(_) => {
                        self.advance();
                        self.diagnostics.push(FeaLexDiagnostic {
                            kind: FeaLexDiagnosticKind::InvalidEscape,
                            span: Span {
                                start: escape_start,
                                end: self.cursor,
                            },
                            message: "unsupported escape in FEA string".to_string(),
                        });
                    }
                    None => break,
                }
            } else {
                self.advance();
            }
        }
        self.push(FeaTokenKind::String, start);
        if !terminated {
            self.diagnostics.push(FeaLexDiagnostic {
                kind: FeaLexDiagnosticKind::UnterminatedString,
                span: Span {
                    start,
                    end: self.cursor,
                },
                message: "unterminated FEA string literal".to_string(),
            });
        }
    }

    fn scan_number(&mut self, start: usize) {
        let mut saw_digits = false;
        while self.peek().is_some_and(|value| value.is_ascii_digit()) {
            saw_digits = true;
            self.advance();
        }
        if self.peek() == Some('.') && self.peek_second() != Some('.') {
            self.advance();
            while self.peek().is_some_and(|value| value.is_ascii_digit()) {
                saw_digits = true;
                self.advance();
            }
        }
        if matches!(self.peek(), Some('e' | 'E')) {
            let exponent_start = self.cursor;
            self.advance();
            if matches!(self.peek(), Some('+' | '-')) {
                self.advance();
            }
            let digits_start = self.cursor;
            while self.peek().is_some_and(|value| value.is_ascii_digit()) {
                self.advance();
            }
            if self.cursor == digits_start {
                self.diagnostics.push(FeaLexDiagnostic {
                    kind: FeaLexDiagnosticKind::InvalidNumber,
                    span: Span {
                        start: exponent_start,
                        end: self.cursor,
                    },
                    message: "numeric exponent requires at least one digit".to_string(),
                });
            }
        }
        self.push(FeaTokenKind::Number, start);
        if !saw_digits {
            self.diagnostics.push(FeaLexDiagnostic {
                kind: FeaLexDiagnosticKind::InvalidNumber,
                span: Span {
                    start,
                    end: self.cursor,
                },
                message: "numeric literal requires at least one digit".to_string(),
            });
        }
    }

    fn scan_identifier(&mut self, start: usize) {
        self.advance();
        while self.peek().is_some_and(is_identifier_continue) {
            self.advance();
        }
        let source = &self.source[start..self.cursor];
        let kind = FeaKeyword::from_source(source)
            .map(FeaTokenKind::Keyword)
            .unwrap_or(FeaTokenKind::Identifier);
        self.push(kind, start);
    }

    fn one_or_two(&mut self, start: usize, second: char, pair: FeaTokenKind, single: FeaTokenKind) {
        self.advance();
        if self.peek() == Some(second) {
            self.advance();
            self.push(pair, start);
        } else {
            self.push(single, start);
        }
    }

    fn single(&mut self, kind: FeaTokenKind, start: usize) {
        self.advance();
        self.push(kind, start);
    }

    fn push(&mut self, kind: FeaTokenKind, start: usize) {
        self.tokens.push(FeaToken {
            kind,
            span: Span {
                start,
                end: self.cursor,
            },
        });
    }

    fn peek(&self) -> Option<char> {
        self.source[self.cursor..].chars().next()
    }

    fn peek_second(&self) -> Option<char> {
        let mut chars = self.source[self.cursor..].chars();
        chars.next()?;
        chars.next()
    }

    fn advance(&mut self) -> Option<char> {
        let current = self.peek()?;
        self.cursor += current.len_utf8();
        Some(current)
    }
}

fn is_identifier_start(value: char) -> bool {
    value == '_' || value.is_alphabetic()
}

fn is_identifier_continue(value: char) -> bool {
    value == '_' || value.is_alphanumeric()
}
