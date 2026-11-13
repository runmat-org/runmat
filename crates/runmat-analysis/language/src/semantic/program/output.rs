use std::collections::BTreeMap;

use runmat_types::Span;

use crate::{
    CheckedExpression, CheckedFeaStatement, CheckedFeaStatementKind, ExpressionScope,
    FeaDiagnostic, FeaDiagnosticLocation, FeaDiagnosticPhase, FeaType, LogicalSourcePath,
};

pub(super) fn definition_value<'a>(
    statements: &'a [CheckedFeaStatement],
    path: &LogicalSourcePath,
    span: Span,
    diagnostics: &mut Vec<FeaDiagnostic>,
) -> Option<&'a CheckedExpression> {
    let values = statements
        .iter()
        .filter_map(|statement| match &statement.kind {
            CheckedFeaStatementKind::Assignment { target, value, .. }
                if target.as_slice() == ["value"] =>
            {
                Some(value)
            }
            _ => None,
        })
        .collect::<Vec<_>>();
    if values.len() == 1 {
        return values.into_iter().next();
    }
    diagnostics.push(FeaDiagnostic::error(
        "fea.semantic.output_value",
        if values.is_empty() {
            "output block requires exactly one `value = <expression>` assignment"
        } else {
            "output block contains more than one `value` assignment"
        },
        FeaDiagnosticPhase::Semantic,
        Some(FeaDiagnosticLocation {
            path: path.clone(),
            span,
        }),
    ));
    None
}

pub(super) fn register(
    name: &str,
    span: Span,
    value_type: FeaType,
    path: &LogicalSourcePath,
    scope: &mut ExpressionScope,
    outputs: &mut BTreeMap<String, FeaType>,
    diagnostics: &mut Vec<FeaDiagnostic>,
) {
    if outputs
        .insert(name.to_string(), value_type.clone())
        .is_some()
    {
        diagnostics.push(FeaDiagnostic::error(
            "fea.semantic.duplicate_output",
            format!("duplicate output `{name}`"),
            FeaDiagnosticPhase::Semantic,
            Some(FeaDiagnosticLocation {
                path: path.clone(),
                span,
            }),
        ));
    }
    scope.insert_reference(["output".to_string(), name.to_string()], value_type);
}
