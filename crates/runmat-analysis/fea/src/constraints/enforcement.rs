//! Deterministic selection and validation of exact equality enforcement.

use runmat_analysis_core::StructuralConstraintKind;

use super::{CanonicalStructuralConstraintRow, ConstraintEnforcement};

const MAX_AFFINE_TERMS_PER_ROW: usize = 32;

pub(super) fn select(
    kind: &StructuralConstraintKind,
    rows: &[CanonicalStructuralConstraintRow],
) -> ConstraintEnforcement {
    if matches!(kind, StructuralConstraintKind::PrescribedMotion(_)) {
        ConstraintEnforcement::AffineReduction
    } else if rows.iter().all(|row| row.terms.len() == 1) {
        ConstraintEnforcement::DirectElimination
    } else if matches!(kind, StructuralConstraintKind::Equation(_))
        && rows
            .iter()
            .any(|row| row.terms.len() > MAX_AFFINE_TERMS_PER_ROW)
    {
        ConstraintEnforcement::LagrangeMultiplier
    } else {
        ConstraintEnforcement::AffineReduction
    }
}

pub(super) fn combine(
    current: ConstraintEnforcement,
    next: ConstraintEnforcement,
) -> ConstraintEnforcement {
    match (current, next) {
        (ConstraintEnforcement::LagrangeMultiplier, _)
        | (_, ConstraintEnforcement::LagrangeMultiplier) => {
            ConstraintEnforcement::LagrangeMultiplier
        }
        (ConstraintEnforcement::AffineReduction, _)
        | (_, ConstraintEnforcement::AffineReduction) => ConstraintEnforcement::AffineReduction,
        _ => ConstraintEnforcement::DirectElimination,
    }
}

pub(super) fn matches_rows(
    enforcement: ConstraintEnforcement,
    rows: &[CanonicalStructuralConstraintRow],
) -> bool {
    if rows.is_empty() {
        return enforcement == ConstraintEnforcement::DirectElimination;
    }
    match enforcement {
        ConstraintEnforcement::DirectElimination => rows.iter().all(|row| row.terms.len() == 1),
        ConstraintEnforcement::AffineReduction => true,
        ConstraintEnforcement::LagrangeMultiplier => rows.iter().any(|row| row.terms.len() > 1),
    }
}
