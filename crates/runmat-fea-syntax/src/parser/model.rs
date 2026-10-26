use super::FeaParser;
use crate::lexical::FeaTokenKind;
use crate::syntax::{FeaStatementKind, FeaTokenRange};
use crate::{FeaSyntaxDiagnostic, FeaSyntaxDiagnosticKind};

impl FeaParser {
    pub(super) fn parse_applied_load(
        &mut self,
        significant: &[usize],
        range: FeaTokenRange,
    ) -> FeaStatementKind {
        let on = significant
            .iter()
            .position(|index| self.lexeme(*index) == "on");
        let equal = significant
            .iter()
            .position(|index| self.kind(*index) == Some(FeaTokenKind::Equal));
        let valid = matches!((on, equal), (Some(1), Some(equal)) if equal > 2 && equal + 1 < significant.len());
        if !valid {
            self.diagnostics.push(FeaSyntaxDiagnostic {
                kind: FeaSyntaxDiagnosticKind::ExpectedToken,
                span: range.span,
                message: "applied load requires `<kind> on <target> = <value>`".into(),
            });
        }
        let on = on.unwrap_or(1);
        let equal = equal.unwrap_or(significant.len());
        let kind = self
            .name(significant[0], FeaSyntaxDiagnosticKind::ExpectedIdentifier)
            .unwrap_or_else(|| super::missing_name(range.span));
        FeaStatementKind::AppliedLoad {
            kind,
            target: self.parse_expression_indices(
                significant.get(on + 1..equal).unwrap_or_default(),
                range.span,
            ),
            value: self.parse_expression_indices(
                significant.get(equal + 1..).unwrap_or_default(),
                range.span,
            ),
        }
    }
}
