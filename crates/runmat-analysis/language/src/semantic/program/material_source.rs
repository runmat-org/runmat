use runmat_design_syntax::{FeaCallArgument, FeaExpression, FeaExpressionKind};

use crate::{
    CheckedCadMaterialSource, FeaDiagnostic, FeaDiagnosticLocation, FeaDiagnosticPhase,
    LogicalSourcePath,
};

pub(super) fn check(
    expression: &FeaExpression,
    path: &LogicalSourcePath,
    diagnostics: &mut Vec<FeaDiagnostic>,
) -> Option<CheckedCadMaterialSource> {
    let FeaExpressionKind::Call { callee, arguments } = &expression.kind else {
        invalid(expression.span, path, diagnostics);
        return None;
    };
    let FeaExpressionKind::Reference(reference) = &callee.kind else {
        invalid(expression.span, path, diagnostics);
        return None;
    };
    let segments = reference
        .segments
        .iter()
        .map(|segment| segment.text.as_str())
        .collect::<Vec<_>>();
    let ["geometry", geometry, "cad_material"] = segments.as_slice() else {
        invalid(expression.span, path, diagnostics);
        return None;
    };
    let [FeaCallArgument::Positional(argument)] = arguments.as_slice() else {
        invalid(expression.span, path, diagnostics);
        return None;
    };
    let FeaExpressionKind::String(material_id) = &argument.kind else {
        invalid(expression.span, path, diagnostics);
        return None;
    };
    if material_id.trim().is_empty() {
        invalid(expression.span, path, diagnostics);
        return None;
    }
    Some(CheckedCadMaterialSource {
        geometry: (*geometry).to_string(),
        material_id: material_id.clone(),
        span: expression.span,
    })
}

fn invalid(
    span: runmat_types::Span,
    path: &LogicalSourcePath,
    diagnostics: &mut Vec<FeaDiagnostic>,
) {
    diagnostics.push(FeaDiagnostic::error(
        "fea.semantic.material_source",
        "material source requires `geometry.<name>.cad_material(\"<persistent-id>\")`",
        FeaDiagnosticPhase::Semantic,
        Some(FeaDiagnosticLocation {
            path: path.clone(),
            span,
        }),
    ));
}
