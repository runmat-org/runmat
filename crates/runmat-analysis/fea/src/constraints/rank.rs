//! Borrowed rank classification with component-bounded, explicitly reserved workspace.
use super::{
    CanonicalStructuralConstraintRow, CanonicalStructuralDof, CanonicalStructuralDofAddress,
    CanonicalStructuralDofOwner, ConstraintPlanError, ReactionRecovery,
};

mod graph;
mod profile;
#[cfg(test)]
mod tests;

pub(super) use classify_constraint_rows as classify;
pub use profile::{
    constraint_rank_graph_workspace_bytes, constraint_rank_workspace_profile,
    ConstraintRankWorkspaceProfile,
};

type DofKey = (CanonicalStructuralDofOwner, CanonicalStructuralDof);
const RELATIVE_RANK_TOLERANCE: f64 = 1.0e-11;
const RIGHT_HAND_SIDE_ROUNDOFF_ULPS: f64 = 64.0;

struct BasisRow<'a> {
    pivot: DofKey,
    coefficients: Vec<(DofKey, f64)>,
    right_hand_side: f64,
    sources: Vec<&'a str>,
}

/// Classification in the original row order, without cloning or mutating the input equations.
#[derive(Debug, Eq, PartialEq)]
pub struct ConstraintRankClassification {
    pub classifications: Vec<ReactionRecovery>,
    pub active_row_count: usize,
    pub redundant_row_count: usize,
}

/// Classify borrowed equations using explicitly reserved component-local fill/provenance slots.
/// This preserves input order and the canonical rank/conflict tolerances. It does not validate
/// row identities or replace canonical constraint-plan validation.
pub fn classify_constraint_rows(
    rows: &[CanonicalStructuralConstraintRow],
) -> Result<ConstraintRankClassification, ConstraintPlanError> {
    let graph = graph::Components::derive(rows)?;
    // Check all slot arithmetic before allocating numeric bases or scratch rows.
    let _profile = ConstraintRankWorkspaceProfile::from_components(&graph)?;
    let mut bases = reserved::<Vec<BasisRow<'_>>>(graph.components.len())?;
    for component in &graph.components {
        bases.push(reserved(component.rows.min(component.dofs))?);
    }
    let mut recoveries = reserved(rows.len())?;
    let mut active = 0;
    let mut redundant = 0;
    for (row, &component_index) in rows.iter().zip(&graph.row_components) {
        let component = &graph.components[component_index];
        let basis = &mut bases[component_index];
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
            return Err(invalid(
                "constraint rank classification requires finite, nonzero coefficients",
            ));
        }
        let mut coefficients = reserved(row.terms.len().min(component.dofs))?;
        for term in &row.terms {
            let key = (term.owner, term.dof);
            let value = term.coefficient / coefficient_scale;
            match coefficients.binary_search_by_key(&key, |(key, _)| *key) {
                Ok(index) => coefficients[index].1 = value,
                Err(index) => {
                    bounded_insert(&mut coefficients, index, (key, value), component.dofs)?
                }
            }
        }
        let mut right_hand_side = row.right_hand_side / coefficient_scale;
        if !right_hand_side.is_finite() {
            return Err(invalid(
                "constraint rank classification cannot normalize its right-hand side",
            ));
        }
        let mut right_hand_side_magnitude = right_hand_side.abs();
        let mut sources = reserved(1)?;
        sources.push(row.constraint_id.as_str());
        // Independent components never contain one another's pivots. Restricting the scan
        // preserves the original within-component basis order and floating-point operations.
        for existing in basis.iter() {
            let Ok(index) = coefficients.binary_search_by_key(&existing.pivot, |(key, _)| *key)
            else {
                continue;
            };
            let (_, factor) = coefficients.remove(index);
            // Even a tiny coefficient can multiply a large prescribed basis value.
            for &(key, coefficient) in &existing.coefficients {
                if key == existing.pivot {
                    continue;
                }
                match coefficients.binary_search_by_key(&key, |(key, _)| *key) {
                    Ok(index) => coefficients[index].1 -= factor * coefficient,
                    Err(index) => bounded_insert(
                        &mut coefficients,
                        index,
                        (key, 0.0 - factor * coefficient),
                        component.dofs,
                    )?,
                }
            }
            let eliminated_right_hand_side = factor * existing.right_hand_side;
            right_hand_side_magnitude += eliminated_right_hand_side.abs();
            right_hand_side -= eliminated_right_hand_side;
            if !right_hand_side.is_finite()
                || !right_hand_side_magnitude.is_finite()
                || coefficients
                    .iter()
                    .any(|(_, coefficient)| !coefficient.is_finite())
            {
                return Err(invalid(
                    "constraint rank elimination produced non-finite values",
                ));
            }
            for &source in &existing.sources {
                if let Err(index) = sources.binary_search(&source) {
                    bounded_insert(&mut sources, index, source, component.rows)?;
                }
            }
        }
        coefficients.retain(|(_, coefficient)| coefficient.abs() > RELATIVE_RANK_TOLERANCE);
        let Some(&(pivot, pivot_coefficient)) = coefficients.first() else {
            // Conflict detection remains limited by floating-point cancellation, not the
            // much wider tolerance used to decide numerical coefficient rank.
            let tolerance =
                RIGHT_HAND_SIDE_ROUNDOFF_ULPS * f64::EPSILON * right_hand_side_magnitude.max(1.0);
            if right_hand_side.abs() > tolerance {
                let first_constraint_id = sources
                    .iter()
                    .copied()
                    .find(|source| *source != row.constraint_id)
                    .unwrap_or(row.constraint_id.as_str())
                    .to_owned();
                return Err(ConstraintPlanError::ConflictingRow {
                    first_constraint_id,
                    second_constraint_id: row.constraint_id.clone(),
                    conflicting_dof,
                });
            }
            recoveries.push(ReactionRecovery::RedundantEquation);
            redundant += 1;
            continue;
        };
        for (_, coefficient) in &mut coefficients {
            *coefficient /= pivot_coefficient;
        }
        right_hand_side /= pivot_coefficient;
        if !right_hand_side.is_finite()
            || coefficients
                .iter()
                .any(|(_, coefficient)| !coefficient.is_finite())
        {
            return Err(invalid(
                "constraint rank pivot normalization produced non-finite values",
            ));
        }
        if basis.len() == component.rows.min(component.dofs) {
            return Err(invalid(
                "constraint rank exceeded its component basis bound",
            ));
        }
        basis.push(BasisRow {
            pivot,
            coefficients,
            right_hand_side,
            sources,
        });
        recoveries.push(if row.terms.len() == 1 {
            ReactionRecovery::UnconstrainedEquilibriumResidual
        } else {
            ReactionRecovery::ConstraintMultiplier
        });
        active += 1;
    }
    Ok(ConstraintRankClassification {
        classifications: recoveries,
        active_row_count: active,
        redundant_row_count: redundant,
    })
}

fn invalid(message: &str) -> ConstraintPlanError {
    ConstraintPlanError::Invalid(message.into())
}

fn reserved<T>(slots: usize) -> Result<Vec<T>, ConstraintPlanError> {
    let mut values = Vec::new();
    values.try_reserve_exact(slots).map_err(|_| {
        invalid("constraint rank workspace allocation failed or exceeds addressable storage")
    })?;
    Ok(values)
}

fn bounded_insert<T: Copy>(
    values: &mut Vec<T>,
    index: usize,
    value: T,
    limit: usize,
) -> Result<(), ConstraintPlanError> {
    if values.len() >= limit {
        return Err(invalid(
            "constraint rank exceeded its reserved component workspace",
        ));
    }
    if values.len() == values.capacity() {
        // Reserve only encountered fill. A fresh buffer makes old/new allocation overlap
        // explicit and bounded by the profile; Vec::insert never performs implicit growth.
        let capacity = values
            .capacity()
            .checked_mul(2)
            .unwrap_or(limit)
            .max(1)
            .min(limit);
        let mut replacement = reserved(capacity)?;
        replacement.extend_from_slice(values);
        *values = replacement;
    }
    values.insert(index, value);
    Ok(())
}
