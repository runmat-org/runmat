//! Authored settings of modal structural procedures: inertia residual vectors and the
//! eigensolver of a modal extraction, and the static correction of a modal-reduced sweep.

use super::{lower::error, values};
use crate::{CheckedExpressionKind, CheckedFeaDeclaration, CheckedFeaStatement, FeaDiagnostic};
use runmat_analysis_core::{
    CoordinateFrame, ModalSolverMethod, ModalStaticCorrection, ProcedureSettings, ProcedureStep,
    SpectrumDirection, StructuralModalBasisSettings, StructuralModalFrequencySettings,
};

/// The settings of a modal extraction: `residual_vectors` and `solver`.
pub(super) fn extraction(
    declaration: &CheckedFeaDeclaration,
    statements: &[CheckedFeaStatement],
    step: &mut ProcedureStep,
    diagnostics: &mut Vec<FeaDiagnostic>,
) -> Option<()> {
    residual_vectors(declaration, statements, step, diagnostics)?;
    solver_method(declaration, statements, step, diagnostics)
}

/// `solver = auto | factorization | iterative`. Absent means `auto`: the resource plan chooses.
/// A forced method whose plan does not fit the admitted memory is refused at planning.
fn solver_method(
    declaration: &CheckedFeaDeclaration,
    statements: &[CheckedFeaStatement],
    step: &mut ProcedureStep,
    diagnostics: &mut Vec<FeaDiagnostic>,
) -> Option<()> {
    let Some(value) = values::assignment(statements, "solver") else {
        return Some(());
    };
    let method = match &value.kind {
        CheckedExpressionKind::Enum(name) => modal_solver_method(name),
        _ => None,
    };
    let Some(method) = method else {
        error(
            declaration,
            value.span,
            "`solver` must be `auto`, `factorization` or `iterative`",
            diagnostics,
        );
        return None;
    };
    if method == ModalSolverMethod::Auto {
        return Some(());
    }
    match &mut step.settings {
        ProcedureSettings::StructuralModalBasis(settings) => settings.solver_method = method,
        settings => {
            *settings = ProcedureSettings::StructuralModalBasis(StructuralModalBasisSettings {
                residual_vector_directions: Vec::new(),
                solver_method: method,
            })
        }
    }
    Some(())
}

/// The closed set of modal solver words (the generated grammar reads this table).
fn modal_solver_method(name: &str) -> Option<ModalSolverMethod> {
    match name {
        "auto" => Some(ModalSolverMethod::Auto),
        "factorization" => Some(ModalSolverMethod::Factorization),
        "iterative" => Some(ModalSolverMethod::Iterative),
        _ => None,
    }
}

/// `residual_vectors = [[0, 0, 1], ...]`: one to three global directions. Absent means none.
fn residual_vectors(
    declaration: &CheckedFeaDeclaration,
    statements: &[CheckedFeaStatement],
    step: &mut ProcedureStep,
    diagnostics: &mut Vec<FeaDiagnostic>,
) -> Option<()> {
    let Some(value) = values::assignment(statements, "residual_vectors") else {
        return Some(());
    };
    let directions = match &value.kind {
        CheckedExpressionKind::List(items) if (1..=3).contains(&items.len()) => items
            .iter()
            .map(values::canonical_vector3)
            .collect::<Option<Vec<_>>>(),
        _ => None,
    };
    let Some(directions) = directions else {
        error(
            declaration,
            value.span,
            "`residual_vectors` is a list of one to three global direction vectors, for example `[[0, 0, 1]]`",
            diagnostics,
        );
        return None;
    };
    step.settings = ProcedureSettings::StructuralModalBasis(StructuralModalBasisSettings {
        residual_vector_directions: directions
            .into_iter()
            .map(|vector| SpectrumDirection {
                frame: CoordinateFrame::Global,
                vector,
            })
            .collect(),
        solver_method: ModalSolverMethod::Auto,
    });
    Some(())
}

/// `static_correction = applied_loads | none`. Absent means none.
pub(super) fn static_correction(
    declaration: &CheckedFeaDeclaration,
    statements: &[CheckedFeaStatement],
    step: &mut ProcedureStep,
    diagnostics: &mut Vec<FeaDiagnostic>,
) -> Option<()> {
    let Some(value) = values::assignment(statements, "static_correction") else {
        return Some(());
    };
    let correction = match &value.kind {
        CheckedExpressionKind::Enum(name) => static_correction_kind(name),
        _ => None,
    };
    let Some(correction) = correction else {
        error(
            declaration,
            value.span,
            "`static_correction` must be `applied_loads` or `none`",
            diagnostics,
        );
        return None;
    };
    step.settings = ProcedureSettings::StructuralModalFrequency(StructuralModalFrequencySettings {
        static_correction: correction,
    });
    Some(())
}

/// The closed set of static-correction words (the generated grammar reads this table).
fn static_correction_kind(name: &str) -> Option<ModalStaticCorrection> {
    match name {
        "applied_loads" => Some(ModalStaticCorrection::AppliedLoads),
        "none" => Some(ModalStaticCorrection::None),
        _ => None,
    }
}
