use super::{header::ProcedureHeader, lower::error, requirements, values};
use crate::{CheckedFeaDeclaration, CheckedFeaStatement, CheckedFeaStatementKind, FeaDiagnostic};
use runmat_analysis_core::{ProcedureStep, SolutionSpace};
use runmat_fea_syntax::FeaAssignmentRelation;
use std::collections::BTreeSet;

pub(super) fn lower(
    id: String,
    declaration: &CheckedFeaDeclaration,
    header: &ProcedureHeader,
    statements: &[CheckedFeaStatement],
    diagnostics: &mut Vec<FeaDiagnostic>,
) -> Option<ProcedureStep> {
    let mut names = BTreeSet::new();
    for statement in statements {
        let valid = matches!(&statement.kind,
            CheckedFeaStatementKind::Assignment {
                target, relation: FeaAssignmentRelation::Equal, ..
            } if target.len() == 1
                && matches!(target[0].as_str(), "prestress" | "basis" | "frequencies" | "static_correction")
                && names.insert(target[0].as_str()));
        if !valid {
            error(declaration, header.span,
                "prestressed_harmonic accepts unique `prestress`, `frequencies` and optional `basis` and `static_correction` assignments",
                diagnostics);
            return None;
        }
    }
    let prestress = requirements::state(declaration, header, statements, "prestress", diagnostics)?;
    let mut step = ProcedureStep::structural_prestressed_harmonic(&id, prestress.0, prestress.1);
    if values::assignment(statements, "basis").is_some() {
        let basis = requirements::state(declaration, header, statements, "basis", diagnostics)?;
        // Reuse the core modal-state contract; do not recreate its persistence,
        // topology or conflict policy in the authoring layer.
        let modal = ProcedureStep::structural_modal_harmonic(id, basis.0, basis.1);
        step.solution_space = SolutionSpace::ModalReduced;
        step.state_inputs.extend(modal.state_inputs);
    }
    step.independent_domain =
        requirements::frequencies(declaration, header, statements, diagnostics)?;
    if step.solution_space == SolutionSpace::ModalReduced {
        super::modal_reduction::static_correction(declaration, statements, &mut step, diagnostics)?;
    } else if values::assignment(statements, "static_correction").is_some() {
        error(
            declaration,
            header.span,
            "`static_correction` applies to a modal-reduced sweep; add `basis`",
            diagnostics,
        );
        return None;
    }
    Some(step)
}
