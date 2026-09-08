use std::collections::{BTreeMap, BTreeSet};

use super::{
    CanonicalStructuralConstraintRow, CanonicalStructuralDof, CanonicalStructuralDofOwner,
    ConstraintPlanError, ReactionRecovery,
};

type DofKey = (CanonicalStructuralDofOwner, CanonicalStructuralDof);

const RELATIVE_RANK_TOLERANCE: f64 = 1.0e-11;

struct BasisRow {
    pivot: DofKey,
    coefficients: BTreeMap<DofKey, f64>,
    right_hand_side: f64,
    sources: BTreeSet<String>,
}

pub(super) fn classify(
    rows: &mut [CanonicalStructuralConstraintRow],
) -> Result<(usize, usize), ConstraintPlanError> {
    let mut basis = Vec::<BasisRow>::new();
    let mut redundant = 0;

    for row in rows {
        let scale = row
            .terms
            .iter()
            .map(|term| term.coefficient.abs())
            .fold(row.right_hand_side.abs(), f64::max)
            .max(1.0);
        let tolerance = scale * RELATIVE_RANK_TOLERANCE;
        let mut coefficients = row
            .terms
            .iter()
            .map(|term| ((term.owner, term.dof), term.coefficient))
            .collect::<BTreeMap<_, _>>();
        let mut right_hand_side = row.right_hand_side;
        let mut sources = BTreeSet::from([row.constraint_id.clone()]);

        for existing in &basis {
            let Some(factor) = coefficients.remove(&existing.pivot) else {
                continue;
            };
            if factor.abs() <= tolerance {
                continue;
            }
            for (key, coefficient) in &existing.coefficients {
                if *key == existing.pivot {
                    continue;
                }
                *coefficients.entry(*key).or_insert(0.0) -= factor * coefficient;
            }
            right_hand_side -= factor * existing.right_hand_side;
            sources.extend(existing.sources.iter().cloned());
            coefficients.retain(|_, coefficient| coefficient.abs() > tolerance);
        }

        let Some((&pivot, &pivot_coefficient)) = coefficients.first_key_value() else {
            if right_hand_side.abs() > tolerance {
                let first_constraint_id = sources
                    .iter()
                    .find(|source| source.as_str() != row.constraint_id)
                    .cloned()
                    .unwrap_or_else(|| row.constraint_id.clone());
                return Err(ConstraintPlanError::ConflictingRow {
                    first_constraint_id,
                    second_constraint_id: row.constraint_id.clone(),
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
        basis.push(BasisRow {
            pivot,
            coefficients,
            right_hand_side,
            sources,
        });
    }

    Ok((basis.len(), redundant))
}
