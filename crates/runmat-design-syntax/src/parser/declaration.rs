use super::*;
use crate::tree::NodeKind;

impl FeaParser {
    pub(super) fn parse_import(&mut self, start: usize, end: usize) -> Option<FeaImport> {
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
                self.tree.node(NodeKind::StringLiteral, *index, *index + 1);
                Some((decode_string(self.lexeme(*index)), self.token(*index).span))
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
            (Some(kind), Some(name), Some((source, source_span))) => Some(FeaImport {
                kind,
                name,
                source,
                source_span,
                // Match the import's syntax node. The terminator's CR and any
                // trailing spaces/comments are trivia, outside the import.
                span: self
                    .trimmed_range(start, end)
                    .map_or(fallback, |range| range.span),
            }),
            _ => None,
        }
    }

    pub(super) fn parse_declaration(&mut self, start: usize) -> (Option<FeaDeclaration>, usize) {
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
        if let Some(range) = self.trimmed_range(header_start, open) {
            self.tree.node(NodeKind::Header, range.start, range.end);
        }
        let header_significant = self.significant(header_start, open);
        let source = self.declaration_source(kind, &header_significant, header.span);
        let material_source = self.material_source(kind, &header_significant, header.span);
        let target = self.mesh_target(kind, &header_significant, header.span);
        let span = Span {
            start: self.token(start).span.start,
            end: body.span.end,
        };
        (
            name.map(|name| FeaDeclaration {
                kind,
                name,
                source,
                material_source,
                target,
                header,
                body,
                span,
            }),
            next,
        )
    }

    fn material_source(
        &mut self,
        kind: FeaDeclarationKind,
        header: &[usize],
        span: Span,
    ) -> Option<FeaExpression> {
        if kind != FeaDeclarationKind::Material
            || !header
                .first()
                .is_some_and(|index| self.at(*index, FeaTokenKind::Keyword(FeaKeyword::From)))
        {
            return None;
        }
        Some(self.parse_expression_indices(&header[1..], span))
    }

    fn declaration_source(
        &mut self,
        kind: FeaDeclarationKind,
        header: &[usize],
        span: Span,
    ) -> Option<FeaReference> {
        let prefix_matches = match kind {
            FeaDeclarationKind::Geometry => header
                .first()
                .is_some_and(|index| self.at(*index, FeaTokenKind::Keyword(FeaKeyword::From))),
            FeaDeclarationKind::Sweep => header
                .first()
                .is_some_and(|index| self.at(*index, FeaTokenKind::Keyword(FeaKeyword::Over))),
            _ => false,
        };
        if !prefix_matches {
            return None;
        }
        let reference = self.parse_reference_indices(&header[1..]);
        if reference.is_none() {
            self.diagnostics.push(FeaSyntaxDiagnostic {
                kind: FeaSyntaxDiagnosticKind::InvalidExpression,
                span,
                message: match kind {
                    FeaDeclarationKind::Geometry => {
                        "expected a geometry reference after `from`".to_string()
                    }
                    FeaDeclarationKind::Sweep => {
                        "expected a study reference after `over`".to_string()
                    }
                    _ => unreachable!(),
                },
            });
        }
        reference
    }

    fn mesh_target(
        &mut self,
        kind: FeaDeclarationKind,
        header: &[usize],
        span: Span,
    ) -> Option<FeaReference> {
        if kind != FeaDeclarationKind::Mesh
            || !header
                .first()
                .is_some_and(|index| self.at(*index, FeaTokenKind::Keyword(FeaKeyword::On)))
        {
            return None;
        }
        let reference = self.parse_reference_indices(&header[1..]);
        if reference.is_none() {
            self.diagnostics.push(FeaSyntaxDiagnostic {
                kind: FeaSyntaxDiagnosticKind::InvalidExpression,
                span,
                message: "expected a geometry reference after `on`".to_string(),
            });
        }
        reference
    }
}
