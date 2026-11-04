use super::syntax::*;
use super::{FeaSyntaxDiagnostic, FeaSyntaxDiagnosticKind};
use crate::lexical::{tokenize_fea, FeaKeyword, FeaTokenKind, FeaTokenStream, FeaTriviaKind};
use crate::tree::{NodeKind, TreeRecorder};
use runmat_types::{merge_span, Span};

mod cursor;
mod declaration;
mod design;
mod exchange;
mod expression;
mod grammar;
mod model;
mod signal;

pub(crate) use grammar::decode_string;
use grammar::*;

pub fn parse_fea(source: impl Into<String>) -> FeaSyntaxModule {
    FeaParser::new(tokenize_fea(source)).parse()
}

/// Parses a source of any dialect into its lossless tree and syntax diagnostics. `.fea` goes
/// through [`parse_fea`] (its AST is on [`FeaSyntaxModule`]); the design dialects through the
/// generic forms.
pub fn parse_source(dialect: crate::Dialect, source: impl Into<String>) -> crate::SourceSyntax {
    if dialect == crate::Dialect::Fea {
        let module = parse_fea(source);
        return crate::SourceSyntax {
            dialect,
            tree: module.tree,
            diagnostics: module.diagnostics,
            tokens: module.tokens,
        };
    }
    let mut parser = FeaParser::new(crate::lexical::tokenize_design(source));
    parser.parse_design_file(dialect);
    let tree = parser.tree.finish(&parser.stream);
    crate::SourceSyntax {
        dialect,
        tree,
        diagnostics: parser.diagnostics,
        tokens: parser.stream,
    }
}

struct FeaParser {
    stream: FeaTokenStream,
    diagnostics: Vec<FeaSyntaxDiagnostic>,
    tree: TreeRecorder,
    expression_mode: expression::ExpressionMode,
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
            tree: TreeRecorder::default(),
            expression_mode: expression::ExpressionMode::Fea,
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
                        if let Some(range) = self.trimmed_range(cursor, end) {
                            self.tree.node(NodeKind::Import, range.start, range.end);
                        }
                        items.push(FeaModuleItem::Import(import));
                    } else if let Some(syntax) = self.trimmed_range(cursor, end) {
                        self.tree.node(NodeKind::Error, syntax.start, syntax.end);
                        items.push(FeaModuleItem::Error(FeaErrorNode { syntax }));
                    }
                    cursor = self.consume_terminator(end);
                }
                Some(FeaTokenKind::Keyword(keyword)) if declaration_kind(keyword).is_some() => {
                    seen_declaration = true;
                    let (declaration, next) = self.parse_declaration(cursor);
                    if let Some(declaration) = declaration {
                        if let Some(range) = self.trimmed_range(cursor, next) {
                            self.tree
                                .node(NodeKind::Declaration, range.start, range.end);
                        }
                        items.push(FeaModuleItem::Declaration(declaration));
                    } else if let Some(syntax) = self.trimmed_range(cursor, next) {
                        self.tree.node(NodeKind::Error, syntax.start, syntax.end);
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
                        self.tree.node(NodeKind::Error, syntax.start, syntax.end);
                        items.push(FeaModuleItem::Error(FeaErrorNode { syntax }));
                    }
                    cursor = self.consume_terminator(end);
                }
            }
        }
        let tree = self.tree.finish(&self.stream);
        FeaSyntaxModule {
            tokens: self.stream,
            items,
            diagnostics: self.diagnostics,
            tree,
        }
    }

    fn parse_block(&mut self, open: usize) -> (FeaBlock, usize) {
        let mut statements = Vec::new();
        let mut cursor = open + 1;
        loop {
            cursor = self.skip_trivia(cursor);
            if self.at(cursor, FeaTokenKind::RightBrace) {
                self.tree.node(NodeKind::Block, open, cursor + 1);
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
                self.tree.node(NodeKind::Block, open, cursor);
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
                    let kind = self
                        .parse_output_definition(start, open_block, body.clone())
                        .or_else(|| self.parse_model_definition(start, open_block, body.clone()))
                        .or_else(|| self.parse_generated_function(start, open_block, body.clone()))
                        .unwrap_or(FeaStatementKind::Block { header, body });
                    if let Some(header) = self.trimmed_range(start, open_block) {
                        self.tree.node(NodeKind::Header, header.start, header.end);
                    }
                    self.tree.node(statement_node_kind(&kind), start, next);
                    statements.push(FeaStatement { kind, span });
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
            _ if self.lexeme(first) == "sample" => self.parse_uniform_sample(&significant, range),
            _ if self.lexeme(first) == "set" => {
                self.parse_source_parameter_update(&significant, range)
            }
            _ if self.lexeme(first) == "use" => self.parse_use(&significant, range),
            _ if self.lexeme(first) == "require"
                && significant
                    .iter()
                    .any(|index| self.kind(*index) == Some(FeaTokenKind::Equal)) =>
            {
                self.parse_requirement(&significant, range)
            }
            _ if significant
                .get(1)
                .is_some_and(|index| self.lexeme(*index) == "on")
                && significant
                    .iter()
                    .any(|index| self.kind(*index) == Some(FeaTokenKind::Equal)) =>
            {
                self.parse_applied_load(&significant, range)
            }
            Some(FeaTokenKind::Keyword(FeaKeyword::Input)) => {
                self.parse_input_statement(&significant, range)
            }
            Some(FeaTokenKind::Keyword(FeaKeyword::Let)) => {
                self.parse_let_statement(&significant, range)
            }
            Some(FeaTokenKind::Keyword(FeaKeyword::Feature)) => {
                self.parse_feature_statement(&significant, range)
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
        self.tree
            .node(statement_node_kind(&kind), range.start, range.end);
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
        let type_syntax = self.token_range(type_start, type_end);
        if let Some(range) = self.trimmed_range(type_syntax.start, type_syntax.end) {
            self.tree.node(NodeKind::Type, range.start, range.end);
        }
        FeaStatementKind::Input {
            name,
            type_syntax,
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

    fn parse_feature_statement(
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
            self.missing_expression(range.span, "expected `=` and a feature operation")
        };
        FeaStatementKind::Feature { name, value }
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
        FeaStatementKind::PropertyAssignment {
            property: self.parse_expression_indices(&significant[1..to], range.span),
            target: self.parse_expression_indices(&significant[to + 1..], range.span),
        }
    }

    fn parse_model_definition(
        &mut self,
        start: usize,
        open: usize,
        body: FeaBlock,
    ) -> Option<FeaStatementKind> {
        let significant = self.significant(start, open);
        if matches!(
            self.kind(*significant.first()?),
            Some(FeaTokenKind::Keyword(FeaKeyword::Signal))
        ) {
            return self.parse_signal_definition(&significant, body);
        }
        let kind = match self.kind(*significant.first()?)? {
            FeaTokenKind::Keyword(FeaKeyword::Section) => FeaModelDefinitionKind::Section,
            FeaTokenKind::Keyword(FeaKeyword::Layup) => FeaModelDefinitionKind::Layup,
            FeaTokenKind::Keyword(FeaKeyword::Orientation) => FeaModelDefinitionKind::Orientation,
            FeaTokenKind::Keyword(FeaKeyword::Constraint) => FeaModelDefinitionKind::Constraint,
            FeaTokenKind::Keyword(FeaKeyword::Load) => FeaModelDefinitionKind::Load,
            FeaTokenKind::Keyword(FeaKeyword::LoadCase) => FeaModelDefinitionKind::LoadCase,
            FeaTokenKind::Keyword(FeaKeyword::Interface) => FeaModelDefinitionKind::Interface,
            FeaTokenKind::Keyword(FeaKeyword::Connector) => FeaModelDefinitionKind::Connector,
            FeaTokenKind::Keyword(FeaKeyword::Excitation) => FeaModelDefinitionKind::Excitation,
            FeaTokenKind::Keyword(FeaKeyword::ResponseSpectrum) => {
                FeaModelDefinitionKind::ResponseSpectrum
            }
            FeaTokenKind::Keyword(FeaKeyword::RandomExcitation) => {
                FeaModelDefinitionKind::RandomExcitation
            }
            FeaTokenKind::Keyword(FeaKeyword::Initial) => FeaModelDefinitionKind::Initial,
            _ => return None,
        };
        let name = significant
            .get(1)
            .and_then(|index| self.name(*index, FeaSyntaxDiagnosticKind::ExpectedIdentifier))
            .unwrap_or_else(|| missing_name(self.token(start).span));
        if significant.len() != 2 {
            let span = significant
                .get(2)
                .map_or(self.token(start).span, |index| self.token(*index).span);
            self.diagnostics.push(FeaSyntaxDiagnostic {
                kind: FeaSyntaxDiagnosticKind::ExpectedToken,
                span,
                message: "model definitions require exactly one name".to_string(),
            });
        }
        Some(FeaStatementKind::ModelDefinition { kind, name, body })
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

    fn parse_output_definition(
        &mut self,
        start: usize,
        open: usize,
        body: FeaBlock,
    ) -> Option<FeaStatementKind> {
        let significant = self.significant(start, open);
        if !matches!(
            self.kind(*significant.first()?),
            Some(FeaTokenKind::Keyword(FeaKeyword::Output))
        ) {
            return None;
        }
        let name = significant
            .get(1)
            .and_then(|index| self.name(*index, FeaSyntaxDiagnosticKind::ExpectedIdentifier))
            .unwrap_or_else(|| missing_name(self.token(start).span));
        if significant.len() != 2 {
            let span = significant
                .get(2)
                .map_or(self.token(start).span, |index| self.token(*index).span);
            self.diagnostics.push(FeaSyntaxDiagnostic {
                kind: FeaSyntaxDiagnosticKind::ExpectedToken,
                span,
                message: "output blocks require exactly one name".to_string(),
            });
        }
        Some(FeaStatementKind::OutputDefinition { name, body })
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
        let (expression, remaining) = expression::parse_expression(
            &self.stream,
            indices,
            &mut self.diagnostics,
            &mut self.tree,
            self.expression_mode,
        );
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

    fn parse_reference_indices(&mut self, indices: &[usize]) -> Option<FeaReference> {
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
        self.tree.node_over(NodeKind::Path, indices);
        Some(FeaReference {
            span: merge_span(segments.first()?.span, segments.last()?.span),
            segments,
        })
    }

    fn name(&mut self, index: usize, kind: FeaSyntaxDiagnosticKind) -> Option<FeaName> {
        if self.at(index, FeaTokenKind::Identifier) {
            self.tree.node(NodeKind::Name, index, index + 1);
            Some(FeaName {
                text: self.lexeme(index).to_string(),
                span: self.token(index).span,
            })
        } else {
            let message = if matches!(self.kind(index), Some(FeaTokenKind::Keyword(_))) {
                format!(
                    "`{}` is a keyword and cannot be a declared name yet; choose another name [pending:language.keyword-names]",
                    self.lexeme(index)
                )
            } else {
                "expected an identifier".to_string()
            };
            self.diagnostics.push(FeaSyntaxDiagnostic {
                kind,
                span: self.token(index).span,
                message,
            });
            None
        }
    }
}

/// The tree node kind of a `.fea` statement.
fn statement_node_kind(kind: &FeaStatementKind) -> NodeKind {
    match kind {
        FeaStatementKind::GeneratedByFunction { .. } => NodeKind::GeneratedBy,
        FeaStatementKind::UniformSample { .. } => NodeKind::UniformSample,
        FeaStatementKind::AppliedLoad { .. } => NodeKind::AppliedLoad,
        FeaStatementKind::SignalDefinition { .. } => NodeKind::SignalDefinition,
        FeaStatementKind::ModelDefinition { .. } => NodeKind::ModelDefinition,
        FeaStatementKind::Block { .. } => NodeKind::BlockStatement,
        FeaStatementKind::Input { .. } | FeaStatementKind::Let { .. } => NodeKind::ValueStatement,
        FeaStatementKind::Feature { .. } => NodeKind::Feature,
        FeaStatementKind::SourceParameterUpdate { .. } => NodeKind::SourceParameterUpdate,
        FeaStatementKind::Use(_) => NodeKind::PropertyUse,
        FeaStatementKind::Requirement { .. } => NodeKind::Requirement,
        FeaStatementKind::Assignment { .. } => NodeKind::Assignment,
        FeaStatementKind::PropertyAssignment { .. } => NodeKind::PropertyAssignment,
        FeaStatementKind::Output { .. } => NodeKind::Output,
        FeaStatementKind::OutputDefinition { .. } => NodeKind::OutputDefinition,
        FeaStatementKind::Include(_) => NodeKind::Include,
        FeaStatementKind::Retain(_) => NodeKind::Retain,
        FeaStatementKind::Command { .. } => NodeKind::Command,
    }
}
