//! Requested retained row expansion. Target, geometry, identity, rank and finalization
//! workspaces are separate owners; this is not whole-compilation or RSS admission.
use super::{
    workspace::{add, check_cancelled, multiply, reserve, slots},
    CanonicalStructuralConstraintRow, CanonicalStructuralConstraintTerm, ConstraintPlanError,
};
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct ConstraintRowExpansionProfile {
    pub row_slots: usize,
    pub term_slots_upper_bound: usize,
    pub retained_id_bytes: u64,
    pub total_requested_bytes: u64,
}
pub fn constraint_row_expansion_profile(
    rows: usize,
    terms_per_row: usize,
    id: &str,
) -> Result<ConstraintRowExpansionProfile, ConstraintPlanError> {
    check_cancelled()?;
    let terms = multiply(rows, terms_per_row)?;
    retained_profile(rows, terms, id)
}

fn retained_profile(
    rows: usize,
    terms: usize,
    id: &str,
) -> Result<ConstraintRowExpansionProfile, ConstraintPlanError> {
    let retained_id_bytes =
        u64::try_from(multiply(rows, id.len())?).map_err(|_| super::workspace::overflow())?;
    let total_requested_bytes = add(
        add(
            slots::<CanonicalStructuralConstraintRow>(rows)?,
            slots::<CanonicalStructuralConstraintTerm>(terms)?,
        )?,
        retained_id_bytes,
    )?;
    Ok(ConstraintRowExpansionProfile {
        row_slots: rows,
        term_slots_upper_bound: terms,
        retained_id_bytes,
        total_requested_bytes,
    })
}
pub(super) fn rows(
    count: usize,
    terms_per_row: usize,
    id: &str,
) -> Result<Vec<CanonicalStructuralConstraintRow>, ConstraintPlanError> {
    variable_rows(
        count,
        multiply(count, terms_per_row)?,
        if count == 0 { 0 } else { terms_per_row },
        id,
    )
}

/// Retained output plus one overlapping row-finalization workspace. Borrowed inputs,
/// targets/geometry and plan/group finalization remain separate; this is not RSS.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct ConstraintRowConstructionProfile {
    pub retained: ConstraintRowExpansionProfile,
    pub maximum_row_workspace: super::row::ConstraintRowWorkspaceProfile,
    pub total_requested_bytes: u64,
}

pub fn constraint_row_construction_profile(
    rows: usize,
    term_slots: usize,
    maximum_terms_per_row: usize,
    id: &str,
) -> Result<ConstraintRowConstructionProfile, ConstraintPlanError> {
    check_cancelled()?;
    if (rows == 0 && (term_slots != 0 || maximum_terms_per_row != 0))
        || maximum_terms_per_row > term_slots
        || term_slots > multiply(rows, maximum_terms_per_row)?
    {
        return Err(super::workspace::overflow());
    }
    let retained = retained_profile(rows, term_slots, id)?;
    let maximum_row_workspace = if rows == 0 {
        super::row::ConstraintRowWorkspaceProfile::default()
    } else {
        super::row::row_workspace_from_term_count(id, maximum_terms_per_row)?
    };
    let total_requested_bytes = add(
        retained.total_requested_bytes,
        maximum_row_workspace.total_requested_bytes,
    )?;
    usize::try_from(total_requested_bytes).map_err(|_| super::workspace::overflow())?;
    Ok(ConstraintRowConstructionProfile {
        retained,
        maximum_row_workspace,
        total_requested_bytes,
    })
}

pub(super) fn variable_rows(
    count: usize,
    term_slots: usize,
    maximum_terms_per_row: usize,
    id: &str,
) -> Result<Vec<CanonicalStructuralConstraintRow>, ConstraintPlanError> {
    let profile =
        constraint_row_construction_profile(count, term_slots, maximum_terms_per_row, id)?;
    reserve(profile.retained.row_slots)
}

pub(super) fn targets<const N: usize>(
    values: [super::ResolvedConstraintTarget; N],
) -> Result<Vec<super::ResolvedConstraintTarget>, ConstraintPlanError> {
    let mut output = reserve(N)?;
    output.extend(values);
    Ok(output)
}
pub(super) fn owned_id(id: &str) -> Result<String, ConstraintPlanError> {
    check_cancelled()?;
    let mut output = String::new();
    output
        .try_reserve_exact(id.len())
        .map_err(|_| super::workspace::overflow())?;
    output.push_str(id);
    Ok(output)
}
#[cfg(test)]
mod tests;
