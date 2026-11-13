//! Additional contact/Newton requested payload. Borrowed basis/summary, material
//! evaluations, reduction construction, sources, outputs and RSS are separate.
use super::*;
use crate::algebra::SparsePatternStatistics;
use std::mem::size_of;
#[cfg(test)]
pub(in crate::solve::nonlinear::solid) fn additional_peak(
    operator: &CanonicalContactOperator<'_>,
    pattern: &SparsePattern,
    reduced: &SparsePattern,
) -> Result<u64, SolidNewtonError> {
    additional_peak_counts(
        operator,
        pattern
            .union_statistics(pattern)
            .map_err(constraint_error)?,
        reduced
            .union_statistics(reduced)
            .map_err(constraint_error)?,
    )
}
pub(in crate::solve::nonlinear::solid) fn additional_peak_counts(
    operator: &CanonicalContactOperator<'_>,
    pattern: SparsePatternStatistics,
    reduced: SparsePatternStatistics,
) -> Result<u64, SolidNewtonError> {
    let overflow =
        || SolidNewtonError::Invalid("contact Newton resource arithmetic overflow".into());
    let add = |a: u64, b: u64| a.checked_add(b).ok_or_else(overflow);
    let slots = |count: usize, word: usize| {
        u64::try_from(count)
            .ok()
            .and_then(|n| n.checked_mul(word as u64))
            .ok_or_else(overflow)
    };
    let csr = |p: SparsePatternStatistics| -> Result<u64, SolidNewtonError> {
        add(
            p.resident_bytes().map_err(constraint_error)?,
            add(slots(p.nonzeros, 8)?, 4096)?,
        )
    };
    let contact = operator.resources().map_err(contact_error)?;
    // Original/driver clones, candidates and acceptance validation can coexist.
    // Each complete contact trial profile includes original coordinates/records.
    let histories_and_trials = contact
        .trial_peak_additional_bytes
        .checked_mul(6)
        .ok_or_else(overflow)?;
    // Shared supports plus coexisting material/contact/merged/reduced operators
    // and planned assembly copies; material baseline remains separately charged.
    let matrices = add(
        csr(pattern)?.checked_mul(6).ok_or_else(overflow)?,
        csr(reduced)?.checked_mul(4).ok_or_else(overflow)?,
    )?;
    let positions = slots(operator.node_rows().len(), size_of::<[f64; 3]>())?
        .checked_mul(4)
        .ok_or_else(overflow)?;
    add(histories_and_trials, add(matrices, positions)?)
}
