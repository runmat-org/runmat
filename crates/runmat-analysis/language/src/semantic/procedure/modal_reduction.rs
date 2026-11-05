//! Authored enrichment of modal-reduced structural procedures: inertia residual vectors on a
//! modal extraction and the static correction of a modal-reduced frequency sweep.

use super::{lower::error, values};
use crate::{CheckedExpressionKind, CheckedFeaDeclaration, CheckedFeaStatement, FeaDiagnostic};
use runmat_analysis_core::{
    CoordinateFrame, ModalStaticCorrection, ProcedureSettings, ProcedureStep, SpectrumDirection,
    StructuralModalBasisSettings, StructuralModalFrequencySettings,
};

/// `residual_vectors = [[0, 0, 1], ...]`: one to three global directions. Absent means none.
pub(super) fn residual_vectors(
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
