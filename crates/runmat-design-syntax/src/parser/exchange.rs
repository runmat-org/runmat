use crate::lexical::FeaTokenKind;

use super::super::syntax::{FeaReference, FeaStatementKind, FeaTokenRange};
use super::super::{FeaSyntaxDiagnostic, FeaSyntaxDiagnosticKind};
use super::FeaParser;

impl FeaParser {
    pub(super) fn parse_source_parameter_update(
        &mut self,
        significant: &[usize],
        range: FeaTokenRange,
    ) -> FeaStatementKind {
        let equal = significant
            .iter()
            .position(|index| self.at(*index, FeaTokenKind::Equal));
        let Some(equal) = equal else {
            return FeaStatementKind::SourceParameterUpdate {
                target: FeaReference {
                    segments: Vec::new(),
                    span: range.span,
                },
                value: self.missing_expression(
                    range.span,
                    "source parameter update requires `=` and a value",
                ),
            };
        };
        let target = self
            .parse_reference_indices(&significant[1..equal])
            .unwrap_or_else(|| {
                self.diagnostics.push(FeaSyntaxDiagnostic {
                    kind: FeaSyntaxDiagnosticKind::InvalidExpression,
                    span: range.span,
                    message: "source parameter update requires `source.parameter.<name>`"
                        .to_string(),
                });
                FeaReference {
                    segments: Vec::new(),
                    span: range.span,
                }
            });
        FeaStatementKind::SourceParameterUpdate {
            target,
            value: self.parse_expression_indices(&significant[equal + 1..], range.span),
        }
    }

    pub(super) fn parse_use(
        &mut self,
        significant: &[usize],
        range: FeaTokenRange,
    ) -> FeaStatementKind {
        let property = self
            .parse_reference_indices(&significant[1..])
            .unwrap_or_else(|| {
                self.diagnostics.push(FeaSyntaxDiagnostic {
                    kind: FeaSyntaxDiagnosticKind::InvalidExpression,
                    span: range.span,
                    message: "`use` requires a property path".to_string(),
                });
                FeaReference {
                    segments: Vec::new(),
                    span: range.span,
                }
            });
        FeaStatementKind::Use(property)
    }

    pub(super) fn parse_requirement(
        &mut self,
        significant: &[usize],
        range: FeaTokenRange,
    ) -> FeaStatementKind {
        let equal = significant
            .iter()
            .position(|index| self.at(*index, FeaTokenKind::Equal))
            .expect("caller requires an equals token");
        let target = self
            .parse_reference_indices(&significant[1..equal])
            .unwrap_or_else(|| {
                self.diagnostics.push(FeaSyntaxDiagnostic {
                    kind: FeaSyntaxDiagnosticKind::InvalidExpression,
                    span: range.span,
                    message: "`require` requires a property path before `=`".to_string(),
                });
                FeaReference {
                    segments: Vec::new(),
                    span: range.span,
                }
            });
        FeaStatementKind::Requirement {
            target,
            value: self.parse_expression_indices(&significant[equal + 1..], range.span),
        }
    }
}
