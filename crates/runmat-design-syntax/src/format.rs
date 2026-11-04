use super::{parse_fea, FeaSyntaxDiagnostic};
use crate::lexical::{FeaToken, FeaTokenKind, FeaTokenStream, FeaTriviaKind};

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct FeaFormatOptions {
    pub indent_width: usize,
}

impl Default for FeaFormatOptions {
    fn default() -> Self {
        Self { indent_width: 4 }
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct FeaFormatResult {
    pub source: String,
    pub diagnostics: Vec<FeaSyntaxDiagnostic>,
}

pub fn format_fea(source: &str, options: FeaFormatOptions) -> FeaFormatResult {
    let module = parse_fea(source);
    format_tokens(&module.tokens, module.diagnostics, options)
}

/// Formats a source of any dialect with the same token formatter as `.fea`: indentation from
/// braces, canonical spacing, comments and blank lines kept.
pub fn format_source(
    dialect: crate::Dialect,
    source: &str,
    options: FeaFormatOptions,
) -> FeaFormatResult {
    let syntax = crate::parse_source(dialect, source);
    format_tokens(&syntax.tokens, syntax.diagnostics, options)
}

fn format_tokens(
    tokens: &FeaTokenStream,
    diagnostics: Vec<FeaSyntaxDiagnostic>,
    options: FeaFormatOptions,
) -> FeaFormatResult {
    let mut formatter = Formatter::new(&tokens.source, options);
    for (index, token) in tokens.tokens.iter().copied().enumerate() {
        let next = tokens.tokens.get(index + 1).copied();
        formatter.token(token, next);
    }
    formatter.finish();
    FeaFormatResult {
        source: formatter.output,
        diagnostics,
    }
}

struct Formatter<'a> {
    source: &'a str,
    options: FeaFormatOptions,
    output: String,
    indent: usize,
    line_start: bool,
    pending_newlines: usize,
    previous: Option<FeaTokenKind>,
    delimiters: Vec<Delimiter>,
    statement_has_relation: bool,
    unit_expression: bool,
    type_expression: bool,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum Delimiter {
    Block,
    Record,
    Parenthesis,
    Bracket,
}

impl<'a> Formatter<'a> {
    fn new(source: &'a str, options: FeaFormatOptions) -> Self {
        Self {
            source,
            options,
            output: String::with_capacity(source.len()),
            indent: 0,
            line_start: true,
            pending_newlines: 0,
            previous: None,
            delimiters: Vec::new(),
            statement_has_relation: false,
            unit_expression: false,
            type_expression: false,
        }
    }

    fn token(&mut self, token: FeaToken, next: Option<FeaToken>) {
        match token.kind {
            FeaTokenKind::Trivia(FeaTriviaKind::Whitespace) => {}
            FeaTokenKind::Trivia(FeaTriviaKind::Newline) => {
                let already_terminated = self.pending_newlines > 0
                    && matches!(
                        self.previous,
                        Some(
                            FeaTokenKind::LeftBrace
                                | FeaTokenKind::RightBrace
                                | FeaTokenKind::Comma
                                | FeaTokenKind::Semicolon
                        )
                    );
                if !already_terminated {
                    self.pending_newlines = (self.pending_newlines + 1).min(2);
                }
            }
            FeaTokenKind::Trivia(FeaTriviaKind::Comment) => {
                self.flush_pending_newlines();
                if !self.line_start && !self.output.ends_with(' ') {
                    self.output.push(' ');
                }
                self.write_indent();
                self.output
                    .push_str(&self.source[token.span.start..token.span.end]);
            }
            FeaTokenKind::EndOfFile => {}
            FeaTokenKind::LeftBrace => self.left_brace(token),
            FeaTokenKind::RightBrace => self.right_brace(token),
            FeaTokenKind::LeftParen => {
                self.flush_pending_inside_delimiter();
                self.trim_spaces();
                self.write_indent();
                self.output.push('(');
                self.delimiters.push(Delimiter::Parenthesis);
                self.previous = Some(token.kind);
            }
            FeaTokenKind::RightParen => {
                let starts_line = self.pending_newlines > 0;
                self.pop_delimiter(Delimiter::Parenthesis);
                self.flush_pending_newlines();
                self.trim_spaces();
                if starts_line {
                    self.write_indent();
                }
                self.output.push(')');
                self.previous = Some(token.kind);
            }
            FeaTokenKind::LeftBracket => {
                self.flush_pending_inside_delimiter();
                self.space_if_needed(token.kind);
                self.write_indent();
                self.output.push('[');
                self.delimiters.push(Delimiter::Bracket);
                self.previous = Some(token.kind);
            }
            FeaTokenKind::RightBracket => {
                let starts_line = self.pending_newlines > 0;
                self.pop_delimiter(Delimiter::Bracket);
                self.flush_pending_newlines();
                self.trim_spaces();
                if starts_line {
                    self.write_indent();
                }
                self.output.push(']');
                self.previous = Some(token.kind);
            }
            FeaTokenKind::Comma => {
                self.flush_pending_inside_delimiter();
                self.trim_spaces();
                self.output.push(',');
                if next.is_some_and(|next| {
                    matches!(next.kind, FeaTokenKind::Trivia(FeaTriviaKind::Newline))
                }) {
                    self.newline(1);
                } else {
                    self.output.push(' ');
                }
                self.previous = Some(token.kind);
                self.unit_expression = false;
                self.type_expression = false;
            }
            FeaTokenKind::Dot | FeaTokenKind::Range => {
                self.flush_pending_inside_delimiter();
                self.trim_spaces();
                self.output
                    .push_str(&self.source[token.span.start..token.span.end]);
                self.previous = Some(token.kind);
            }
            FeaTokenKind::Colon => {
                self.flush_pending_inside_delimiter();
                self.trim_spaces();
                self.output.push_str(" : ");
                self.previous = Some(token.kind);
                self.type_expression = true;
                self.unit_expression = false;
            }
            FeaTokenKind::Semicolon => {
                self.newline(1);
                self.previous = None;
                self.statement_has_relation = false;
                self.unit_expression = false;
                self.type_expression = false;
            }
            kind if is_spaced_operator(kind) => {
                self.flush_pending_inside_delimiter();
                if matches!(kind, FeaTokenKind::Plus | FeaTokenKind::Minus)
                    && self.previous.is_none_or(is_unary_predecessor)
                {
                    self.write_indent();
                    self.output
                        .push_str(&self.source[token.span.start..token.span.end]);
                    self.previous = Some(kind);
                    return;
                }
                if (self.unit_expression
                    && matches!(
                        kind,
                        FeaTokenKind::Slash | FeaTokenKind::Star | FeaTokenKind::Caret
                    ))
                    || (self.type_expression
                        && matches!(kind, FeaTokenKind::Less | FeaTokenKind::Greater))
                {
                    self.trim_spaces();
                    self.output
                        .push_str(&self.source[token.span.start..token.span.end]);
                    self.previous = Some(kind);
                    return;
                }
                self.trim_spaces();
                if !self.line_start {
                    self.output.push(' ');
                }
                self.output
                    .push_str(&self.source[token.span.start..token.span.end]);
                self.output.push(' ');
                self.previous = Some(kind);
                if matches!(
                    kind,
                    FeaTokenKind::Equal | FeaTokenKind::LessEqual | FeaTokenKind::GreaterEqual
                ) {
                    self.statement_has_relation = true;
                }
                self.unit_expression = false;
                if kind == FeaTokenKind::Equal {
                    self.type_expression = false;
                }
            }
            _ => {
                self.flush_pending_newlines();
                if is_name(token.kind)
                    && matches!(
                        self.previous,
                        Some(
                            FeaTokenKind::Number
                                | FeaTokenKind::RightBracket
                                | FeaTokenKind::RightParen
                        )
                    )
                {
                    self.unit_expression = true;
                }
                self.space_if_needed(token.kind);
                self.write_indent();
                self.output
                    .push_str(&self.source[token.span.start..token.span.end]);
                self.previous = Some(token.kind);
            }
        }
    }

    fn left_brace(&mut self, token: FeaToken) {
        self.flush_pending_inside_delimiter();
        let record = self.statement_has_relation;
        if record {
            self.space_if_needed(token.kind);
            self.write_indent();
            self.output.push('{');
            self.output.push(' ');
            self.delimiters.push(Delimiter::Record);
        } else {
            self.trim_spaces();
            if !self.line_start {
                self.output.push(' ');
            }
            self.write_indent();
            self.output.push('{');
            self.delimiters.push(Delimiter::Block);
            self.indent += 1;
            self.newline(1);
            self.statement_has_relation = false;
        }
        self.previous = Some(token.kind);
    }

    fn right_brace(&mut self, token: FeaToken) {
        let delimiter = self.delimiters.last().copied();
        if delimiter == Some(Delimiter::Record) {
            self.flush_pending_inside_delimiter();
            self.trim_spaces();
            if !self.output.ends_with('{') {
                self.output.push(' ');
            }
            self.output.push('}');
            self.delimiters.pop();
        } else {
            self.indent = self.indent.saturating_sub(1);
            self.newline(1);
            self.flush_pending_newlines();
            self.write_indent();
            self.output.push('}');
            self.delimiters.pop();
            self.newline(1);
            self.statement_has_relation = false;
            self.unit_expression = false;
            self.type_expression = false;
        }
        self.previous = Some(token.kind);
    }

    fn space_if_needed(&mut self, current: FeaTokenKind) {
        let Some(previous) = self.previous else {
            return;
        };
        let needs_space = (is_word(previous) && is_word(current))
            || (matches!(
                previous,
                FeaTokenKind::RightParen | FeaTokenKind::RightBracket
            ) && is_word(current))
            || (is_word(previous) && current == FeaTokenKind::LeftBracket);
        if needs_space && !self.line_start && !self.output.ends_with(' ') {
            self.output.push(' ');
        }
    }

    fn flush_pending_inside_delimiter(&mut self) {
        if self.pending_newlines == 0 {
            return;
        }
        if self
            .delimiters
            .iter()
            .rev()
            .any(|delimiter| matches!(delimiter, Delimiter::Parenthesis | Delimiter::Bracket))
        {
            self.newline(1);
        }
        self.flush_pending_newlines();
    }

    fn flush_pending_newlines(&mut self) {
        if self.pending_newlines == 0 {
            return;
        }
        let count = if self.line_start {
            self.pending_newlines.saturating_sub(1)
        } else {
            self.pending_newlines
        };
        self.trim_spaces();
        for _ in 0..count {
            if !self.output.ends_with('\n') || count > 1 {
                self.output.push('\n');
            }
        }
        self.line_start = self.output.is_empty() || self.output.ends_with('\n');
        self.pending_newlines = 0;
        if self.line_start {
            self.previous = None;
            self.statement_has_relation = false;
            self.unit_expression = false;
            self.type_expression = false;
        }
    }

    fn newline(&mut self, count: usize) {
        self.pending_newlines = self.pending_newlines.max(count).min(2);
    }

    fn write_indent(&mut self) {
        if self.line_start {
            let continuation = self
                .delimiters
                .iter()
                .filter(|delimiter| {
                    matches!(
                        delimiter,
                        Delimiter::Record | Delimiter::Parenthesis | Delimiter::Bracket
                    )
                })
                .count();
            self.output.extend(std::iter::repeat_n(
                ' ',
                (self.indent + continuation) * self.options.indent_width,
            ));
            self.line_start = false;
        }
    }

    fn trim_spaces(&mut self) {
        while self.output.ends_with(' ') || self.output.ends_with('\t') {
            self.output.pop();
        }
    }

    fn pop_delimiter(&mut self, expected: Delimiter) {
        if self.delimiters.last() == Some(&expected) {
            self.delimiters.pop();
        }
    }

    fn finish(&mut self) {
        self.pending_newlines = 1;
        self.flush_pending_newlines();
        while self.output.ends_with("\n\n") {
            self.output.pop();
        }
        if !self.output.is_empty() && !self.output.ends_with('\n') {
            self.output.push('\n');
        }
    }
}

fn is_word(kind: FeaTokenKind) -> bool {
    matches!(
        kind,
        FeaTokenKind::Keyword(_)
            | FeaTokenKind::Identifier
            | FeaTokenKind::Number
            | FeaTokenKind::String
            | FeaTokenKind::Percent
    )
}

fn is_name(kind: FeaTokenKind) -> bool {
    matches!(
        kind,
        FeaTokenKind::Keyword(_) | FeaTokenKind::Identifier | FeaTokenKind::Percent
    )
}

fn is_unary_predecessor(kind: FeaTokenKind) -> bool {
    matches!(
        kind,
        FeaTokenKind::LeftParen
            | FeaTokenKind::LeftBracket
            | FeaTokenKind::LeftBrace
            | FeaTokenKind::Comma
            | FeaTokenKind::Equal
            | FeaTokenKind::EqualEqual
            | FeaTokenKind::BangEqual
            | FeaTokenKind::Less
            | FeaTokenKind::LessEqual
            | FeaTokenKind::Greater
            | FeaTokenKind::GreaterEqual
            | FeaTokenKind::Plus
            | FeaTokenKind::Minus
            | FeaTokenKind::Star
            | FeaTokenKind::Slash
            | FeaTokenKind::Caret
            | FeaTokenKind::Range
            | FeaTokenKind::Colon
            | FeaTokenKind::PlusMinus
            | FeaTokenKind::At
    )
}

fn is_spaced_operator(kind: FeaTokenKind) -> bool {
    matches!(
        kind,
        FeaTokenKind::PlusMinus
            | FeaTokenKind::At
            | FeaTokenKind::Equal
            | FeaTokenKind::EqualEqual
            | FeaTokenKind::BangEqual
            | FeaTokenKind::Less
            | FeaTokenKind::LessEqual
            | FeaTokenKind::Greater
            | FeaTokenKind::GreaterEqual
            | FeaTokenKind::Plus
            | FeaTokenKind::Minus
            | FeaTokenKind::Star
            | FeaTokenKind::Slash
            | FeaTokenKind::Caret
    )
}
