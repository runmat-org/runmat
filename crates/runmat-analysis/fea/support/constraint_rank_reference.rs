//! Test-only pre-redesign elimination oracle. Never used by production validation.
use std::collections::{BTreeMap, BTreeSet};

use runmat_analysis_fea::constraints::{
    CanonicalStructuralConstraintRow, CanonicalStructuralDof, CanonicalStructuralDofAddress,
    CanonicalStructuralDofOwner, ConstraintPlanError, ReactionRecovery,
};

type DofKey = (CanonicalStructuralDofOwner, CanonicalStructuralDof);

const RELATIVE_RANK_TOLERANCE: f64 = 1.0e-11;
const RIGHT_HAND_SIDE_ROUNDOFF_ULPS: f64 = 64.0;

struct BasisRow {
    pivot: DofKey,
    coefficients: BTreeMap<DofKey, f64>,
    right_hand_side: f64,
    sources: BTreeSet<String>,
}

pub fn classify(
    rows: &mut [CanonicalStructuralConstraintRow],
) -> Result<(usize, usize), ConstraintPlanError> {
    let mut basis = Vec::<BasisRow>::new();
    let mut redundant = 0;

    for row in rows {
        let conflicting_dof = (row.terms.len() == 1).then(|| CanonicalStructuralDofAddress {
            owner: row.terms[0].owner,
            dof: row.terms[0].dof,
        });
        // Equation scaling must not change rank: the prescribed value does not set the
        // tolerance for deciding whether a coefficient is present.
        let coefficient_scale = row
            .terms
            .iter()
            .map(|term| term.coefficient.abs())
            .fold(0.0, f64::max);
        if !coefficient_scale.is_finite() || coefficient_scale == 0.0 {
            return Err(ConstraintPlanError::Invalid(
                "constraint rank classification requires finite, nonzero coefficients".into(),
            ));
        }
        let mut coefficients = row
            .terms
            .iter()
            .map(|term| ((term.owner, term.dof), term.coefficient / coefficient_scale))
            .collect::<BTreeMap<_, _>>();
        let mut right_hand_side = row.right_hand_side / coefficient_scale;
        if !right_hand_side.is_finite() {
            return Err(ConstraintPlanError::Invalid(
                "constraint rank classification cannot normalize its right-hand side".into(),
            ));
        }
        let mut right_hand_side_magnitude = right_hand_side.abs();
        let mut sources = BTreeSet::from([row.constraint_id.clone()]);

        for existing in &basis {
            let Some(factor) = coefficients.remove(&existing.pivot) else {
                continue;
            };
            // A small coefficient can multiply a large prescribed basis value. Eliminate
            // it from the right-hand side before applying any rank tolerance.
            for (key, coefficient) in &existing.coefficients {
                if *key == existing.pivot {
                    continue;
                }
                *coefficients.entry(*key).or_insert(0.0) -= factor * coefficient;
            }
            let eliminated_right_hand_side = factor * existing.right_hand_side;
            right_hand_side_magnitude += eliminated_right_hand_side.abs();
            right_hand_side -= eliminated_right_hand_side;
            if !right_hand_side.is_finite()
                || !right_hand_side_magnitude.is_finite()
                || coefficients
                    .values()
                    .any(|coefficient| !coefficient.is_finite())
            {
                return Err(ConstraintPlanError::Invalid(
                    "constraint rank elimination produced non-finite values".into(),
                ));
            }
            sources.extend(existing.sources.iter().cloned());
        }
        coefficients.retain(|_, coefficient| coefficient.abs() > RELATIVE_RANK_TOLERANCE);

        let Some((&pivot, &pivot_coefficient)) = coefficients.first_key_value() else {
            // Conflict detection is limited by floating-point cancellation, not the much
            // wider tolerance used to decide numerical coefficient rank.
            let right_hand_side_tolerance =
                RIGHT_HAND_SIDE_ROUNDOFF_ULPS * f64::EPSILON * right_hand_side_magnitude.max(1.0);
            if right_hand_side.abs() > right_hand_side_tolerance {
                let first_constraint_id = sources
                    .iter()
                    .find(|source| source.as_str() != row.constraint_id)
                    .cloned()
                    .unwrap_or_else(|| row.constraint_id.clone());
                return Err(ConstraintPlanError::ConflictingRow {
                    first_constraint_id,
                    second_constraint_id: row.constraint_id.clone(),
                    conflicting_dof,
                });
            }
            row.reaction_recovery = ReactionRecovery::RedundantEquation;
            redundant += 1;
            continue;
        };

        for coefficient in coefficients.values_mut() {
            *coefficient /= pivot_coefficient;
        }
        right_hand_side /= pivot_coefficient;
        if !right_hand_side.is_finite()
            || coefficients
                .values()
                .any(|coefficient| !coefficient.is_finite())
        {
            return Err(ConstraintPlanError::Invalid(
                "constraint rank pivot normalization produced non-finite values".into(),
            ));
        }
        basis.push(BasisRow {
            pivot,
            coefficients,
            right_hand_side,
            sources,
        });
    }

    Ok((basis.len(), redundant))
}
