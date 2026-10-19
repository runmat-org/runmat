//! Additional dependency-index construction and query workspace; other model validation excluded.
use super::*;
use std::mem::size_of;
pub(crate) fn resident_bytes(sequence: &AnalysisSequence) -> Result<u64, AnalysisValidationError> {
    let overflow = || AnalysisValidationError::DependencyGraphResourceOverflow;
    let v = u64::try_from(sequence.steps.len()).map_err(|_| overflow())?;
    let e = u64::try_from(sequence.dependencies.len()).map_err(|_| overflow())?;
    let bytes = |count: u64, size: usize| count.checked_mul(size as u64).ok_or_else(overflow);
    let mut total = bytes(v, size_of::<(usize, &ProcedureStep)>())?;
    // Conservatively coexist original-order duplicate inventories and indexed edge pairs.
    let edge_slot = size_of::<(usize, &crate::SequenceDependency)>()
        + size_of::<usize>()
        + size_of::<(usize, usize)>();
    total = total
        .checked_add(bytes(e, edge_slot)?)
        .ok_or_else(overflow)?;
    // Forward and reverse CSR, plus max cursor/indegree/ready/query scratch.
    let csr_entries = v
        .checked_add(1)
        .and_then(|n| n.checked_add(e))
        .and_then(|n| n.checked_mul(2))
        .ok_or_else(overflow)?;
    total = total
        .checked_add(bytes(csr_entries, size_of::<usize>())?)
        .ok_or_else(overflow)?;
    let scratch_slot = (2 * size_of::<usize>()).max(size_of::<usize>() + size_of::<bool>());
    total
        .checked_add(bytes(v, scratch_slot)?)
        .ok_or_else(overflow)
}
