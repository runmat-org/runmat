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
    let profile = constraint_row_expansion_profile(count, terms_per_row, id)?;
    usize::try_from(profile.total_requested_bytes).map_err(|_| super::workspace::overflow())?;
    reserve(profile.row_slots)
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
