use super::syntax::*;
use super::{FeaSyntaxDiagnostic, FeaSyntaxDiagnosticKind};
use runmat_lexer::fea::{tokenize_fea, FeaKeyword, FeaTokenKind, FeaTokenStream, FeaTriviaKind};
use runmat_types::{merge_span, Span};

mod cursor;
mod expression;
mod grammar;

use grammar::*;

pub fn parse_fea(source: impl Into<String>) -> FeaSyntaxModule {
    FeaParser::new(tokenize_fea(source)).parse()
}

struct FeaParser {
    stream: FeaTokenStream,
    diagnostics: Vec<FeaSyntaxDiagnostic>,
}

impl FeaParser {
    fn new(stream: FeaTokenStream) -> Self {
        let diagnostics = stream
            .diagnostics
            .iter()
            .map(|diagnostic| FeaSyntaxDiagnostic {
                kind: FeaSyntaxDiagnosticKind::Lexical,
                span: diagnostic.span,
                message: diagnostic.message.clone(),
            })
            .collect();
        Self {
            stream,
            diagnostics,
        }
    }

    fn parse(mut self) -> FeaSyntaxModule {
        let mut items = Vec::new();
        let mut cursor = 0;
        let mut seen_declaration = false;
        while !self.at(cursor, FeaTokenKind::EndOfFile) {
            cursor = self.skip_trivia(cursor);
            if self.at(cursor, FeaTokenKind::EndOfFile) {
                break;
            }
            match self.kind(cursor) {
                Some(FeaTokenKind::Keyword(FeaKeyword::Import)) => {
                    let end = self.statement_end(cursor);
                    if seen_declaration {
                        self.diagnostics.push(FeaSyntaxDiagnostic {
                            kind: FeaSyntaxDiagnosticKind::ImportAfterDeclaration,
                            span: self.token(cursor).span,
                            message: "FEA imports must precede declarations".to_string(),
                        });
                    }
                    if let Some(import) = self.parse_import(cursor, end) {
                        items.push(FeaModuleItem::Import(import));
                    } else if let Some(syntax) = self.trimmed_range(cursor, end) {
                        items.push(FeaModuleItem::Error(FeaErrorNode { syntax }));
                    }
                    cursor = self.consume_terminator(end);
                }
                Some(FeaTokenKind::Keyword(keyword)) if declaration_kind(keyword).is_some() => {
                    seen_declaration = true;
                    let (declaration, next) = self.parse_declaration(cursor);
                    if let Some(declaration) = declaration {
                        items.push(FeaModuleItem::Declaration(declaration));
                    } else if let Some(syntax) = self.trimmed_range(cursor, next) {
                        items.push(FeaModuleItem::Error(FeaErrorNode { syntax }));
                    }
                    cursor = next.max(cursor + 1);
                }
                _ => {
                    let span = self.token(cursor).span;
                    self.diagnostics.push(FeaSyntaxDiagnostic {
                        kind: FeaSyntaxDiagnosticKind::ExpectedDeclaration,
                        span,
                        message: "expected an import or top-level FEA declaration".to_string(),
                    });
                    let end = self.statement_end(cursor);
                    if let Some(syntax) = self.trimmed_range(cursor, end) {
                        items.push(FeaModuleItem::Error(FeaErrorNode { syntax }));
                    }
                    cursor = self.consume_terminator(end);
                }
            }
        }
        FeaSyntaxModule {
            tokens: self.stream,
            items,
            diagnostics: self.diagnostics,
        }
    }

    fn parse_import(&mut self, start: usize, end: usize) -> Option<FeaImport> {
        let significant = self.significant(start, end);
        let fallback = self.range_span(start, end);
        let kind = significant
            .get(1)
            .and_then(|index| import_kind(self.kind(*index)?));
        if kind.is_none() {
            self.diagnostics.push(FeaSyntaxDiagnostic {
                kind: FeaSyntaxDiagnosticKind::ExpectedImportKind,
                span: significant
                    .get(1)
                    .map_or(fallback, |index| self.token(*index).span),
                message: "expected geometry, mesh, material, model, study, sweep, execution, or function after `import`".to_string(),
            });
        }
        let name = significant
            .get(2)
            .and_then(|index| self.name(*index, FeaSyntaxDiagnosticKind::ExpectedIdentifier));
        let from = significant
            .get(3)
            .is_some_and(|index| self.at(*index, FeaTokenKind::Keyword(FeaKeyword::From)));
        if !from {
            self.diagnostics.push(FeaSyntaxDiagnostic {
                kind: FeaSyntaxDiagnosticKind::ExpectedToken,
                span: significant
                    .get(3)
                    .map_or(fallback, |index| self.token(*index).span),
                message: "expected `from` in FEA import".to_string(),
            });
        }
        let source = significant.get(4).and_then(|index| {
            if self.at(*index, FeaTokenKind::String) {
                Some(decode_string(self.lexeme(*index)))
            } else {
                self.diagnostics.push(FeaSyntaxDiagnostic {
                    kind: FeaSyntaxDiagnosticKind::ExpectedString,
                    span: self.token(*index).span,
                    message: "expected a quoted source path in FEA import".to_string(),
                });
                None
            }
        });
        if significant.len() > 5 {
            let index = significant[5];
            self.diagnostics.push(FeaSyntaxDiagnostic {
                kind: FeaSyntaxDiagnosticKind::ExpectedToken,
                span: self.token(index).span,
                message: "unexpected token after FEA import source".to_string(),
            });
        }
        match (kind, name, source) {
            (Some(kind), Some(name), Some(source)) => Some(FeaImport {
                kind,
                name,
                source,
                span: fallback,
            }),
            _ => None,
        }
    }

    fn parse_declaration(&mut self, start: usize) -> (Option<FeaDeclaration>, usize) {
        let Some(FeaTokenKind::Keyword(keyword)) = self.kind(start) else {
            return (None, start + 1);
        };
        let Some(kind) = declaration_kind(keyword) else {
            return (None, start + 1);
        };
        let name_index = self.next_significant(start + 1);
        let name = name_index
            .and_then(|index| self.name(index, FeaSyntaxDiagnosticKind::ExpectedIdentifier));
        let mut cursor = name_index.map_or(start + 1, |index| index + 1);
        let mut open = None;
        while !self.at(cursor, FeaTokenKind::EndOfFile) {
            if self.at(cursor, FeaTokenKind::LeftBrace) {
                open = Some(cursor);
                break;
            }
            if self.is_newline(cursor) || self.at(cursor, FeaTokenKind::Semicolon) {
                break;
            }
            cursor += 1;
        }
        let Some(open) = open else {
            self.diagnostics.push(FeaSyntaxDiagnostic {
                kind: FeaSyntaxDiagnosticKind::ExpectedToken,
                span: self.token(cursor.min(self.stream.tokens.len() - 1)).span,
                message: "expected `{` to begin FEA declaration".to_string(),
            });
            return (None, self.consume_terminator(self.statement_end(start)));
        };
        let (body, next) = self.parse_block(open);
        let header_start = name_index.map_or(start + 1, |index| index + 1);
        let header = self.token_range(header_start, open);
        let span = Span {
            start: self.token(start).span.start,
            end: body.span.end,
        };
        (
            name.map(|name| FeaDeclaration {
                kind,
                name,
                header,
                body,
                span,
            }),
            next,
        )
    }

    fn parse_block(&mut self, open: usize) -> (FeaBlock, usize) {
        let mut statements = Vec::new();
        let mut cursor = open + 1;
        loop {
            cursor = self.skip_trivia(cursor);
            if self.at(cursor, FeaTokenKind::RightBrace) {
                let close = self.token(cursor).span;
                return (
                    FeaBlock {
                        statements,
                        span: Span {
                            start: self.token(open).span.start,
                            end: close.end,
                        },
                    },
                    cursor + 1,
                );
            }
            if self.at(cursor, FeaTokenKind::EndOfFile) {
                let eof = self.token(cursor).span;
                self.diagnostics.push(FeaSyntaxDiagnostic {
                    kind: FeaSyntaxDiagnosticKind::UnclosedBlock,
                    span: self.token(open).span,
                    message: "unclosed FEA block".to_string(),
                });
                return (
                    FeaBlock {
                        statements,
                        span: Span {
                            start: self.token(open).span.start,
                            end: eof.end,
                        },
                    },
                    cursor,
                );
            }

            let start = cursor;
            match self.scan_block_member(start) {
                BlockMemberEnd::Nested(open_block) => {
                    let header = self.token_range(start, open_block);
                    let (body, next) = self.parse_block(open_block);
                    let span = Span {
                        start: self.token(start).span.start,
                        end: body.span.end,
                    };
                    statements.push(FeaStatement {
                        kind: FeaStatementKind::Block { header, body },
                        span,
                    });
                    cursor = next;
                }
                BlockMemberEnd::Statement(end) => {
                    if let Some(statement) = self.parse_statement(start, end) {
                        statements.push(statement);
                    }
                    cursor = self.consume_terminator(end);
                }
                BlockMemberEnd::Close(end) => {
                    if end > start {
                        if let Some(statement) = self.parse_statement(start, end) {
                            statements.push(statement);
                        }
                    }
                    cursor = end;
                }
            }
        }
    }

    fn scan_block_member(&self, start: usize) -> BlockMemberEnd {
        let mut cursor = start;
        let mut parens = 0usize;
        let mut brackets = 0usize;
        let mut records = 0usize;
        let mut saw_relation = false;
        loop {
            match self.kind(cursor) {
                Some(FeaTokenKind::EndOfFile) => return BlockMemberEnd::Close(cursor),
                Some(FeaTokenKind::LeftParen) => parens += 1,
                Some(FeaTokenKind::RightParen) => parens = parens.saturating_sub(1),
                Some(FeaTokenKind::LeftBracket) => brackets += 1,
                Some(FeaTokenKind::RightBracket) => brackets = brackets.saturating_sub(1),
                Some(
                    FeaTokenKind::Equal | FeaTokenKind::LessEqual | FeaTokenKind::GreaterEqual,
                ) if parens == 0 && brackets == 0 && records == 0 => {
                    saw_relation = true;
                }
                Some(FeaTokenKind::LeftBrace) if parens == 0 && brackets == 0 && records == 0 => {
                    if saw_relation {
                        records += 1;
                    } else {
                        return BlockMemberEnd::Nested(cursor);
                    }
                }
                Some(FeaTokenKind::LeftBrace) => records += 1,
                Some(FeaTokenKind::RightBrace) if records > 0 => records -= 1,
                Some(FeaTokenKind::RightBrace) if parens == 0 && brackets == 0 => {
                    return BlockMemberEnd::Close(cursor);
                }
                Some(FeaTokenKind::Semicolon) if parens == 0 && brackets == 0 && records == 0 => {
                    return BlockMemberEnd::Statement(cursor);
                }
                Some(FeaTokenKind::Trivia(FeaTriviaKind::Newline))
                    if parens == 0 && brackets == 0 && records == 0 =>
                {
                    let previous = (start..cursor)
                        .rev()
                        .find_map(|index| self.kind(index).filter(|kind| !kind.is_trivia()));
                    if !previous.is_some_and(continues_on_next_line) {
                        return BlockMemberEnd::Statement(cursor);
                    }
                }
                _ => {}
            }
            cursor += 1;
        }
    }

    fn parse_statement(&mut self, start: usize, end: usize) -> Option<FeaStatement> {
        let range = self.trimmed_range(start, end)?;
        let significant = self.significant(range.start, range.end);
        let first = *significant.first()?;
        let span = range.span;
        let kind = match self.kind(first) {
            Some(FeaTokenKind::Keyword(FeaKeyword::Input)) => {
                self.parse_input_statement(&significant, range)
            }
            Some(FeaTokenKind::Keyword(FeaKeyword::Let)) => {
                self.parse_let_statement(&significant, range)
            }
            Some(FeaTokenKind::Keyword(FeaKeyword::Assign)) => {
                self.parse_material_assignment(&significant, range)
            }
            Some(FeaTokenKind::Keyword(FeaKeyword::Output)) => {
                self.parse_output_statement(&significant, range)
            }
            Some(FeaTokenKind::Keyword(FeaKeyword::Include)) => {
                let expression = self.parse_expression_indices(&significant[1..], span);
                FeaStatementKind::Include(expression)
            }
            Some(FeaTokenKind::Keyword(FeaKeyword::Retain)) => {
                let expression = self.parse_expression_indices(&significant[1..], span);
                FeaStatementKind::Retain(expression)
            }
            _ => self
                .parse_assignment_statement(&significant, range)
                .unwrap_or(FeaStatementKind::Command { syntax: range }),
        };
        Some(FeaStatement { kind, span })
    }

    fn parse_input_statement(
        &mut self,
        significant: &[usize],
        range: FeaTokenRange,
    ) -> FeaStatementKind {
        let name = significant
            .get(1)
            .and_then(|index| self.name(*index, FeaSyntaxDiagnosticKind::ExpectedIdentifier))
            .unwrap_or_else(|| missing_name(range.span));
        let colon = significant
            .iter()
            .position(|index| self.at(*index, FeaTokenKind::Colon));
        let equal = significant
            .iter()
            .position(|index| self.at(*index, FeaTokenKind::Equal));
        let type_start = colon
            .and_then(|position| significant.get(position + 1).copied())
            .unwrap_or(range.end);
        let type_end = equal
            .and_then(|position| significant.get(position).copied())
            .unwrap_or(range.end);
        let default = equal.map(|position| {
            self.parse_expression_indices(&significant[position + 1..], range.span)
        });
        FeaStatementKind::Input {
            name,
            type_syntax: self.token_range(type_start, type_end),
            default,
        }
    }

    fn parse_let_statement(
        &mut self,
        significant: &[usize],
        range: FeaTokenRange,
    ) -> FeaStatementKind {
        let name = significant
            .get(1)
            .and_then(|index| self.name(*index, FeaSyntaxDiagnosticKind::ExpectedIdentifier))
            .unwrap_or_else(|| missing_name(range.span));
        let equal = significant
            .iter()
            .position(|index| self.at(*index, FeaTokenKind::Equal));
        let value = if let Some(position) = equal {
            self.parse_expression_indices(&significant[position + 1..], range.span)
        } else {
            self.missing_expression(range.span, "expected `=` and a binding expression")
        };
        FeaStatementKind::Let { name, value }
    }

    fn parse_material_assignment(
        &mut self,
        significant: &[usize],
        range: FeaTokenRange,
    ) -> FeaStatementKind {
        let to = significant
            .iter()
            .position(|index| self.at(*index, FeaTokenKind::Keyword(FeaKeyword::To)));
        let Some(to) = to else {
            return FeaStatementKind::Command { syntax: range };
        };
        FeaStatementKind::MaterialAssignment {
            material: self.parse_expression_indices(&significant[1..to], range.span),
            target: self.parse_expression_indices(&significant[to + 1..], range.span),
        }
    }

    fn parse_output_statement(
        &mut self,
        significant: &[usize],
        range: FeaTokenRange,
    ) -> FeaStatementKind {
        let name = significant
            .get(1)
            .and_then(|index| self.name(*index, FeaSyntaxDiagnosticKind::ExpectedIdentifier))
            .unwrap_or_else(|| missing_name(range.span));
        let equal = significant
            .iter()
            .position(|index| self.at(*index, FeaTokenKind::Equal));
        let value = if let Some(position) = equal {
            self.parse_expression_indices(&significant[position + 1..], range.span)
        } else {
            self.missing_expression(range.span, "expected `=` and an output expression")
        };
        FeaStatementKind::Output { name, value }
    }

    fn parse_assignment_statement(
        &mut self,
        significant: &[usize],
        range: FeaTokenRange,
    ) -> Option<FeaStatementKind> {
        let relation = significant
            .iter()
            .enumerate()
            .find_map(|(position, index)| {
                Some((
                    position,
                    match self.kind(*index)? {
                        FeaTokenKind::Equal => FeaAssignmentRelation::Equal,
                        FeaTokenKind::LessEqual => FeaAssignmentRelation::LessEqual,
                        FeaTokenKind::GreaterEqual => FeaAssignmentRelation::GreaterEqual,
                        _ => return None,
                    },
                ))
            });
        let (position, relation) = relation?;
        let target = self.parse_reference_indices(&significant[..position])?;
        let value = self.parse_expression_indices(&significant[position + 1..], range.span);
        Some(FeaStatementKind::Assignment {
            target,
            relation,
            value,
        })
    }

    fn parse_expression_indices(&mut self, indices: &[usize], fallback: Span) -> FeaExpression {
        if indices.is_empty() {
            return self.missing_expression(fallback, "expected an expression");
        }
        let (expression, remaining) =
            expression::parse_expression(&self.stream, indices, &mut self.diagnostics);
        if remaining < indices.len() {
            let span = self.token(indices[remaining]).span;
            self.diagnostics.push(FeaSyntaxDiagnostic {
                kind: FeaSyntaxDiagnosticKind::InvalidExpression,
                span,
                message: "unexpected token after FEA expression".to_string(),
            });
        }
        expression
    }

    fn missing_expression(&mut self, span: Span, message: &str) -> FeaExpression {
        self.diagnostics.push(FeaSyntaxDiagnostic {
            kind: FeaSyntaxDiagnosticKind::InvalidExpression,
            span,
            message: message.to_string(),
        });
        FeaExpression {
            kind: FeaExpressionKind::Missing,
            span,
        }
    }

    fn parse_reference_indices(&self, indices: &[usize]) -> Option<FeaReference> {
        if indices.is_empty() {
            return None;
        }
        let mut segments = Vec::new();
        let mut cursor = 0;
        while cursor < indices.len() {
            let index = indices[cursor];
            if !is_name_token(self.kind(index)?) {
                return None;
            }
            segments.push(FeaName {
                text: self.lexeme(index).to_string(),
                span: self.token(index).span,
            });
            cursor += 1;
            if cursor == indices.len() {
                break;
            }
            if !self.at(indices[cursor], FeaTokenKind::Dot) {
                return None;
            }
            cursor += 1;
        }
        Some(FeaReference {
            span: merge_span(segments.first()?.span, segments.last()?.span),
            segments,
        })
    }

    fn name(&mut self, index: usize, kind: FeaSyntaxDiagnosticKind) -> Option<FeaName> {
        if self.at(index, FeaTokenKind::Identifier) {
            Some(FeaName {
                text: self.lexeme(index).to_string(),
                span: self.token(index).span,
            })
        } else {
            self.diagnostics.push(FeaSyntaxDiagnostic {
                kind,
                span: self.token(index).span,
                message: "expected an identifier".to_string(),
            });
            None
        }
    }
}
