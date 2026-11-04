use super::{decode_string, is_name_token, is_quantity_value, starts_expression};
use crate::lexical::{FeaToken, FeaTokenKind, FeaTokenStream};
use crate::syntax::*;
use crate::tree::{NodeKind, TreeRecorder};
use crate::{FeaSyntaxDiagnostic, FeaSyntaxDiagnosticKind};
use runmat_types::{merge_span, Span};

/// Which expression forms are in force.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum ExpressionMode {
    /// The `.fea` forms, exactly as before the design dialects.
    Fea,
    /// Adds the design value forms: `a ± t`, `a as <unit>`, `a @ "<pin>"` and `%` units.
    /// Their results are read from the tree; the returned [`FeaExpression`] only drives
    /// precedence.
    Design,
}

pub(crate) fn parse_expression(
    stream: &FeaTokenStream,
    indices: &[usize],
    diagnostics: &mut Vec<FeaSyntaxDiagnostic>,
    tree: &mut TreeRecorder,
    mode: ExpressionMode,
) -> (FeaExpression, usize) {
    let mut parser = ExpressionParser {
        stream,
        indices,
        cursor: 0,
        diagnostics,
        tree,
        mode,
    };
    let expression = parser.parse(0);
    (expression, parser.cursor)
}

struct ExpressionParser<'a> {
    stream: &'a FeaTokenStream,
    indices: &'a [usize],
    cursor: usize,
    diagnostics: &'a mut Vec<FeaSyntaxDiagnostic>,
    tree: &'a mut TreeRecorder,
    mode: ExpressionMode,
}

impl ExpressionParser<'_> {
    /// Records a node over the significant tokens `from..self.cursor`.
    fn record(&mut self, kind: NodeKind, from: usize) {
        if from < self.cursor && self.cursor <= self.indices.len() {
            self.tree
                .node(kind, self.indices[from], self.indices[self.cursor - 1] + 1);
        }
    }

    fn parse(&mut self, minimum_precedence: u8) -> FeaExpression {
        let first = self.cursor;
        let mut left = self.parse_prefix();
        loop {
            if matches!(left.kind, FeaExpressionKind::Reference(_))
                && self.current_kind().is_some_and(starts_expression)
                && !self.at_design_postfix()
            {
                let argument = self.parse(7);
                self.record(NodeKind::PrefixApplication, first);
                let span = merge_span(left.span, argument.span);
                left = FeaExpression {
                    kind: FeaExpressionKind::PrefixApplication {
                        callee: Box::new(left),
                        argument: Box::new(argument),
                    },
                    span,
                };
                continue;
            }
            if self.can_start_unit() && is_quantity_value(&left.kind) {
                let start = self.indices[self.cursor];
                let unit_start = self.cursor;
                while self.cursor < self.indices.len() && self.is_unit_token(self.cursor) {
                    self.cursor += 1;
                }
                if self.cursor > unit_start {
                    self.record(NodeKind::Unit, unit_start);
                    self.record(NodeKind::Quantity, first);
                    let end = self.indices[self.cursor - 1];
                    left = FeaExpression {
                        span: merge_span(left.span, self.token(end).span),
                        kind: FeaExpressionKind::Quantity {
                            value: Box::new(left),
                            unit: FeaTokenRange {
                                start,
                                end: end + 1,
                                span: Span {
                                    start: self.token(start).span.start,
                                    end: self.token(end).span.end,
                                },
                            },
                        },
                    };
                }
            }
            if self.mode == ExpressionMode::Design
                && minimum_precedence <= 2
                && self.current_kind() == Some(FeaTokenKind::PlusMinus)
            {
                self.cursor += 1;
                let tolerance = self.parse(3);
                self.record(NodeKind::Toleranced, first);
                let span = merge_span(left.span, tolerance.span);
                left = FeaExpression {
                    kind: FeaExpressionKind::Group(Box::new(left)),
                    span,
                };
                continue;
            }
            if self.mode == ExpressionMode::Design && minimum_precedence == 0 {
                if self.current_lexeme() == Some("as") {
                    self.cursor += 1;
                    let unit_start = self.cursor;
                    while self.cursor < self.indices.len() && self.is_unit_token(self.cursor) {
                        self.cursor += 1;
                    }
                    if self.cursor == unit_start {
                        let index = self.indices[unit_start - 1];
                        self.invalid_at(index, "expected a unit after `as`");
                    }
                    self.record(NodeKind::Unit, unit_start);
                    self.record(NodeKind::Coerce, first);
                    let span = self.span_from(first);
                    left = FeaExpression {
                        kind: FeaExpressionKind::Group(Box::new(left)),
                        span,
                    };
                    continue;
                }
                if self.current_kind() == Some(FeaTokenKind::At) {
                    self.cursor += 1;
                    match self.current_kind() {
                        Some(FeaTokenKind::String) => {
                            self.cursor += 1;
                            self.record(NodeKind::Version, self.cursor - 1);
                        }
                        _ => {
                            let index = self.indices[self.cursor - 1];
                            self.invalid_at(index, "expected a quoted revision after `@`");
                        }
                    }
                    self.record(NodeKind::Pinned, first);
                    let span = self.span_from(first);
                    left = FeaExpression {
                        kind: FeaExpressionKind::Group(Box::new(left)),
                        span,
                    };
                    continue;
                }
            }
            let Some((operator, precedence, right_associative)) = self.binary_operator() else {
                break;
            };
            if precedence < minimum_precedence {
                break;
            }
            self.cursor += 1;
            let right = self.parse(if right_associative {
                precedence
            } else {
                precedence + 1
            });
            self.record(NodeKind::Binary, first);
            let span = merge_span(left.span, right.span);
            left = FeaExpression {
                kind: FeaExpressionKind::Binary {
                    left: Box::new(left),
                    operator,
                    right: Box::new(right),
                },
                span,
            };
        }
        left
    }

    fn parse_prefix(&mut self) -> FeaExpression {
        let Some(index) = self.indices.get(self.cursor).copied() else {
            return FeaExpression {
                kind: FeaExpressionKind::Missing,
                span: Span::default(),
            };
        };
        let start = self.cursor;
        let mut expression = match self.token(index).kind {
            FeaTokenKind::Plus | FeaTokenKind::Minus => {
                let first = self.cursor;
                self.cursor += 1;
                let operand = self.parse(7);
                self.record(NodeKind::Unary, first);
                FeaExpression {
                    span: merge_span(self.token(index).span, operand.span),
                    kind: FeaExpressionKind::Unary {
                        operator: if self.token(index).kind == FeaTokenKind::Plus {
                            FeaUnaryOperator::Positive
                        } else {
                            FeaUnaryOperator::Negative
                        },
                        operand: Box::new(operand),
                    },
                }
            }
            FeaTokenKind::Number => {
                self.cursor += 1;
                self.record(NodeKind::NumberLiteral, self.cursor - 1);
                FeaExpression {
                    kind: FeaExpressionKind::Number(self.lexeme(index).to_string()),
                    span: self.token(index).span,
                }
            }
            FeaTokenKind::String => {
                self.cursor += 1;
                self.record(NodeKind::StringLiteral, self.cursor - 1);
                FeaExpression {
                    kind: FeaExpressionKind::String(decode_string(self.lexeme(index))),
                    span: self.token(index).span,
                }
            }
            FeaTokenKind::LeftParen => self.parse_group(index),
            FeaTokenKind::LeftBracket => self.parse_list(index),
            FeaTokenKind::LeftBrace => self.parse_record(index),
            kind if is_name_token(kind) => self.parse_reference(index),
            _ => {
                self.cursor += 1;
                self.diagnostics.push(FeaSyntaxDiagnostic {
                    kind: FeaSyntaxDiagnosticKind::InvalidExpression,
                    span: self.token(index).span,
                    message: "expected an FEA expression".to_string(),
                });
                FeaExpression {
                    kind: FeaExpressionKind::Missing,
                    span: self.token(index).span,
                }
            }
        };
        while self.current_kind() == Some(FeaTokenKind::LeftParen) {
            expression = self.parse_call(expression);
            self.record(NodeKind::Call, start);
        }
        expression
    }

    fn parse_reference(&mut self, first: usize) -> FeaExpression {
        let from = self.cursor;
        let mut segments = vec![FeaName {
            text: self.lexeme(first).to_string(),
            span: self.token(first).span,
        }];
        self.cursor += 1;
        while self.current_kind() == Some(FeaTokenKind::Dot) {
            let Some(next) = self.indices.get(self.cursor + 1).copied() else {
                break;
            };
            if !is_name_token(self.token(next).kind) {
                break;
            }
            self.cursor += 2;
            segments.push(FeaName {
                text: self.lexeme(next).to_string(),
                span: self.token(next).span,
            });
        }
        self.record(NodeKind::Reference, from);
        let span = merge_span(segments[0].span, segments.last().unwrap().span);
        FeaExpression {
            kind: FeaExpressionKind::Reference(FeaReference { segments, span }),
            span,
        }
    }

    fn parse_group(&mut self, open: usize) -> FeaExpression {
        let first = self.cursor;
        self.cursor += 1;
        let expression = self.parse(0);
        let close = self.consume_close(FeaTokenKind::RightParen, open, "`)`");
        self.record(NodeKind::Group, first);
        FeaExpression {
            span: merge_span(self.token(open).span, close),
            kind: FeaExpressionKind::Group(Box::new(expression)),
        }
    }

    fn parse_list(&mut self, open: usize) -> FeaExpression {
        let first = self.cursor;
        self.cursor += 1;
        let mut values = Vec::new();
        while self.cursor < self.indices.len()
            && self.current_kind() != Some(FeaTokenKind::RightBracket)
        {
            values.push(self.parse(0));
            if self.current_kind() == Some(FeaTokenKind::Comma) {
                self.cursor += 1;
            } else if self.current_kind() != Some(FeaTokenKind::RightBracket) {
                self.missing_comma();
                break;
            }
        }
        let close = self.consume_close(FeaTokenKind::RightBracket, open, "`]`");
        self.record(NodeKind::List, first);
        FeaExpression {
            span: merge_span(self.token(open).span, close),
            kind: FeaExpressionKind::List(values),
        }
    }

    fn parse_record(&mut self, open: usize) -> FeaExpression {
        let first = self.cursor;
        self.cursor += 1;
        let mut entries = Vec::new();
        while self.cursor < self.indices.len()
            && self.current_kind() != Some(FeaTokenKind::RightBrace)
        {
            let Some(name_index) = self.indices.get(self.cursor).copied() else {
                break;
            };
            let quoted_key = self.mode == ExpressionMode::Design
                && self.token(name_index).kind == FeaTokenKind::String;
            if !is_name_token(self.token(name_index).kind) && !quoted_key {
                self.invalid_at(name_index, "expected a record member name");
                break;
            }
            let name = FeaName {
                text: if quoted_key {
                    decode_string(self.lexeme(name_index))
                } else {
                    self.lexeme(name_index).to_string()
                },
                span: self.token(name_index).span,
            };
            let entry = self.cursor;
            self.cursor += 1;
            if self.current_kind() != Some(FeaTokenKind::Equal) {
                self.invalid_at(name_index, "expected `=` after record member name");
                break;
            }
            self.cursor += 1;
            let value = self.parse(0);
            self.tree.node(NodeKind::Name, name_index, name_index + 1);
            self.record(NodeKind::RecordEntry, entry);
            let span = merge_span(name.span, value.span);
            entries.push(FeaRecordEntry { name, value, span });
            if self.current_kind() == Some(FeaTokenKind::Comma) {
                self.cursor += 1;
            } else if self.current_kind() != Some(FeaTokenKind::RightBrace) {
                self.missing_comma();
                break;
            }
        }
        let close = self.consume_close(FeaTokenKind::RightBrace, open, "`}`");
        self.record(NodeKind::Record, first);
        FeaExpression {
            span: merge_span(self.token(open).span, close),
            kind: FeaExpressionKind::Record(entries),
        }
    }

    fn parse_call(&mut self, callee: FeaExpression) -> FeaExpression {
        let open = self.indices[self.cursor];
        self.cursor += 1;
        let mut arguments = Vec::new();
        while self.cursor < self.indices.len()
            && self.current_kind() != Some(FeaTokenKind::RightParen)
        {
            let named = self
                .indices
                .get(self.cursor)
                .zip(self.indices.get(self.cursor + 1))
                .filter(|(name, equal)| {
                    is_name_token(self.token(**name).kind)
                        && self.token(**equal).kind == FeaTokenKind::Equal
                })
                .map(|(name, _)| *name);
            if let Some(name_index) = named {
                let name = FeaName {
                    text: self.lexeme(name_index).to_string(),
                    span: self.token(name_index).span,
                };
                let argument = self.cursor;
                self.cursor += 2;
                let value = self.parse(0);
                self.tree.node(NodeKind::Name, name_index, name_index + 1);
                self.record(NodeKind::NamedArgument, argument);
                let span = merge_span(name.span, value.span);
                arguments.push(FeaCallArgument::Named { name, value, span });
            } else {
                arguments.push(FeaCallArgument::Positional(self.parse(0)));
            }
            if self.current_kind() == Some(FeaTokenKind::Comma) {
                self.cursor += 1;
            } else if self.current_kind() != Some(FeaTokenKind::RightParen) {
                self.missing_comma();
                break;
            }
        }
        let close = self.consume_close(FeaTokenKind::RightParen, open, "`)`");
        FeaExpression {
            span: merge_span(callee.span, close),
            kind: FeaExpressionKind::Call {
                callee: Box::new(callee),
                arguments,
            },
        }
    }

    fn consume_close(&mut self, expected: FeaTokenKind, open: usize, label: &str) -> Span {
        if self.current_kind() == Some(expected) {
            let close = self.token(self.indices[self.cursor]).span;
            self.cursor += 1;
            close
        } else {
            self.diagnostics.push(FeaSyntaxDiagnostic {
                kind: FeaSyntaxDiagnosticKind::ExpectedToken,
                span: self.token(open).span,
                message: format!("expected {label} to close expression"),
            });
            self.indices
                .get(self.cursor.saturating_sub(1))
                .map_or(self.token(open).span, |index| self.token(*index).span)
        }
    }

    fn binary_operator(&self) -> Option<(FeaBinaryOperator, u8, bool)> {
        Some(match self.current_kind()? {
            FeaTokenKind::Range => (FeaBinaryOperator::Range, 1, false),
            FeaTokenKind::EqualEqual => (FeaBinaryOperator::Equal, 2, false),
            FeaTokenKind::BangEqual => (FeaBinaryOperator::NotEqual, 2, false),
            FeaTokenKind::Less => (FeaBinaryOperator::Less, 2, false),
            FeaTokenKind::LessEqual => (FeaBinaryOperator::LessEqual, 2, false),
            FeaTokenKind::Greater => (FeaBinaryOperator::Greater, 2, false),
            FeaTokenKind::GreaterEqual => (FeaBinaryOperator::GreaterEqual, 2, false),
            FeaTokenKind::Plus => (FeaBinaryOperator::Add, 3, false),
            FeaTokenKind::Minus => (FeaBinaryOperator::Subtract, 3, false),
            FeaTokenKind::Star => (FeaBinaryOperator::Multiply, 4, false),
            FeaTokenKind::Slash => (FeaBinaryOperator::Divide, 4, false),
            FeaTokenKind::Caret => (FeaBinaryOperator::Power, 6, true),
            _ => return None,
        })
    }

    fn can_start_unit(&self) -> bool {
        if self.at_design_postfix() {
            return false;
        }
        self.current_kind().is_some_and(|kind| {
            is_name_token(kind)
                || (self.mode == ExpressionMode::Design && kind == FeaTokenKind::Percent)
        })
    }

    /// In design mode, `as` and `@` end an operand: they start a coercion or a pin.
    fn at_design_postfix(&self) -> bool {
        self.mode == ExpressionMode::Design
            && (self.current_lexeme() == Some("as")
                || self.current_kind() == Some(FeaTokenKind::At))
    }

    fn current_lexeme(&self) -> Option<&str> {
        self.indices
            .get(self.cursor)
            .map(|index| self.lexeme(*index))
    }

    fn span_from(&self, from: usize) -> Span {
        let start = self.token(self.indices[from]).span.start;
        let end = self.token(self.indices[self.cursor - 1]).span.end;
        Span { start, end }
    }

    fn is_unit_token(&self, cursor: usize) -> bool {
        let kind = self.token(self.indices[cursor]).kind;
        if matches!(kind, FeaTokenKind::Plus | FeaTokenKind::Minus) {
            // A sign after a unit power belongs to its exponent; other signs
            // remain arithmetic operators between quantity expressions.
            return cursor > 0
                && self.token(self.indices[cursor - 1]).kind == FeaTokenKind::Caret
                && self
                    .indices
                    .get(cursor + 1)
                    .is_some_and(|index| self.token(*index).kind == FeaTokenKind::Number);
        }
        if self.mode == ExpressionMode::Design {
            if kind == FeaTokenKind::Percent {
                return true;
            }
            let lexeme = self.lexeme(self.indices[cursor]);
            if lexeme == "as" {
                return false;
            }
        }
        is_name_token(kind)
            || matches!(
                kind,
                FeaTokenKind::Slash
                    | FeaTokenKind::Star
                    | FeaTokenKind::Caret
                    | FeaTokenKind::Number
            )
    }

    fn missing_comma(&mut self) {
        if let Some(index) = self.indices.get(self.cursor).copied() {
            self.invalid_at(index, "expected `,` between same-line items");
        }
    }

    fn invalid_at(&mut self, index: usize, message: &str) {
        self.diagnostics.push(FeaSyntaxDiagnostic {
            kind: FeaSyntaxDiagnosticKind::InvalidExpression,
            span: self.token(index).span,
            message: message.to_string(),
        });
    }

    fn current_kind(&self) -> Option<FeaTokenKind> {
        self.indices
            .get(self.cursor)
            .map(|index| self.token(*index).kind)
    }

    fn token(&self, index: usize) -> FeaToken {
        self.stream.tokens[index]
    }

    fn lexeme(&self, index: usize) -> &str {
        self.stream.lexeme(self.token(index))
    }
}
