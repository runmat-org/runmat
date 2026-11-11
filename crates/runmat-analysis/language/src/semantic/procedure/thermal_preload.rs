use super::{header::ProcedureHeader, lower::error, requirements};
use crate::{CheckedFeaDeclaration, CheckedFeaStatement, CheckedFeaStatementKind, FeaDiagnostic};
use runmat_design_syntax::FeaAssignmentRelation;

pub(super) fn temperature(
    declaration: &CheckedFeaDeclaration,
    header: &ProcedureHeader,
    statements: &[CheckedFeaStatement],
    diagnostics: &mut Vec<FeaDiagnostic>,
) -> Option<(String, String)> {
    if !matches!(statements, [CheckedFeaStatement { kind: CheckedFeaStatementKind::Assignment { target, relation: FeaAssignmentRelation::Equal, .. }, .. }] if target.as_slice() == ["temperature"])
    {
        error(declaration, header.span, "thermomechanical preload requires exactly one `temperature = analysis.<producer>.state.<name>` binding", diagnostics);
        return None;
    }
    requirements::state(declaration, header, statements, "temperature", diagnostics)
}
