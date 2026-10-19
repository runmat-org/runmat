//! Additional group construction and incremental retained compilation overlap.
//! Lowered targets/rows/geometry and final plan/rank construction remain separate admission.
use super::*;
use crate::constraints::{
    plan,
    workspace::{add, check_cancelled, count_add, ordering_workspace_bytes, slots},
};

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct ConstraintGroupConstructionWorkspaceProfile {
    pub target_ordering_bytes: u64,
    pub id_copy_bytes: u64,
    pub reference_identity_bytes: u64,
    pub source_workspace_bytes: u64,
    pub total_requested_bytes: u64,
}

/// Additional requested group construction, excluding all borrowed lowered evidence,
/// group-record buffers, allocator overhead and RSS. Coupling retains one reference ID.
pub fn constraint_group_construction_workspace_profile(
    constraint: &StructuralConstraint,
    target_count: usize,
    has_coupling: bool,
) -> Result<ConstraintGroupConstructionWorkspaceProfile, ConstraintPlanError> {
    check_cancelled()?;
    let target_ordering_bytes = ordering_workspace_bytes(target_count)?;
    let id_copy_bytes = constraint.constraint_id.len() as u64;
    let reference_identity_bytes = slots::<runmat_execution::Digest>(usize::from(has_coupling))?;
    let source_workspace_bytes =
        plan::constraint_source_workspace_profile(constraint)?.total_requested_bytes;
    let total_requested_bytes = [
        target_ordering_bytes,
        id_copy_bytes,
        reference_identity_bytes,
        source_workspace_bytes,
    ]
    .into_iter()
    .try_fold(0, add)?;
    Ok(ConstraintGroupConstructionWorkspaceProfile {
        target_ordering_bytes,
        id_copy_bytes,
        reference_identity_bytes,
        source_workspace_bytes,
        total_requested_bytes,
    })
}

/// Shared codec estimates bound retained original evidence incrementally. Every new
/// payload is traversed once; the existing whole plan is never rescanned or copied.
/// One contract envelope belongs to the collection, not each increment. This is a
/// storage inventory/addressability check, not whole-compilation policy or measured RSS.
pub(super) struct AccumulationWorkspace {
    retained_bytes: u64,
}
impl AccumulationWorkspace {
    pub(super) fn new(group_capacity: usize) -> Result<Self, ConstraintPlanError> {
        let retained_bytes = plan::canonical_value_workspace_bytes(&(
            Vec::<ConstraintGroupEvidence>::new(),
            Vec::<CanonicalStructuralConstraintRow>::new(),
        ))?;
        let retained_bytes = add(
            retained_bytes,
            slots::<ConstraintGroupEvidence>(group_capacity)?,
        )?;
        usize::try_from(retained_bytes).map_err(|_| crate::constraints::workspace::overflow())?;
        Ok(Self { retained_bytes })
    }

    pub(super) fn retained_bytes(&self) -> u64 {
        self.retained_bytes
    }

    pub(super) fn advance(
        &mut self,
        group: &ConstraintGroupEvidence,
        incoming: &[CanonicalStructuralConstraintRow],
        incoming_capacity: usize,
        current_row_count: usize,
        current_row_capacity: usize,
    ) -> Result<u64, ConstraintPlanError> {
        check_cancelled()?;
        if current_row_count > current_row_capacity || incoming.len() > incoming_capacity {
            return Err(crate::constraints::workspace::overflow());
        }
        let payload = plan::canonical_value_workspace_profile(&([group], incoming))?
            .materialization_bytes_without_envelope()
            .ok_or_else(crate::constraints::workspace::overflow)?;
        let buffers = add(
            super::storage::row_buffers(incoming, incoming_capacity)?,
            super::storage::evidence_buffers(
                &group.targets,
                &group.pairings,
                group.coupling.as_ref(),
            )?,
        )?;
        let increment = add(payload, buffers)?;
        let retained_bytes = add(self.retained_bytes, increment)?;
        // A moving exact reserve may overlap its old allocation, the new row buffer,
        // and incoming row records. Payload ownership itself moves without cloning.
        let new_row_count = count_add(current_row_count, incoming.len())?;
        let relocation_slots = count_add(
            count_add(current_row_capacity, new_row_count)?,
            incoming_capacity,
        )?;
        let total = add(
            retained_bytes,
            slots::<CanonicalStructuralConstraintRow>(relocation_slots)?,
        )?;
        usize::try_from(total).map_err(|_| crate::constraints::workspace::overflow())?;
        self.retained_bytes = retained_bytes;
        Ok(total)
    }
}

#[cfg(test)]
mod tests;
