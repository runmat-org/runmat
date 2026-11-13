//! The serial owner starts kernel allocation ownership after importing inputs.
//! No input charge is released or reclassified when kernel records transfer.
use super::*;
use crate::geometry_control::{AllocationPhase, RecordStorage};
use runmat_meshing_core::MeshingChunkStream;

impl MeshingStageControl<'_> {
    pub(crate) fn begin_kernel_allocations(&mut self) -> Result<(), Box<MeshingFailure>> {
        if self.allocation_phase.is_some() {
            return Err(failure(
                self.stage,
                MeshingFailureCategory::InternalInvariantViolation,
                "kernel allocation ownership is already active",
                None,
            ));
        }
        self.allocation_phase = Some(Arc::new(AllocationPhase::new(Arc::clone(
            &self.geometry_usage,
        ))));
        Ok(())
    }
    pub(crate) fn finish_kernel_allocations(
        &mut self,
        streams: &[MeshingChunkStream],
    ) -> Result<RecordStorage, Box<MeshingFailure>> {
        let bytes = streams
            .iter()
            .flat_map(|stream| &stream.records)
            .try_fold(0u64, |sum, record| {
                sum.checked_add(record.capacity() as u64)
            })
            .ok_or_else(|| {
                failure(
                    self.stage,
                    MeshingFailureCategory::InternalInvariantViolation,
                    "returned meshing record capacity inventory overflow",
                    None,
                )
            })?;
        let phase = self.allocation_phase.as_ref().ok_or_else(|| {
            failure(
                self.stage,
                MeshingFailureCategory::InternalInvariantViolation,
                "kernel allocation ownership is absent",
                None,
            )
        })?;
        let retained = phase.transfer_records(bytes).ok_or_else(|| {
            failure(
                self.stage,
                MeshingFailureCategory::InternalInvariantViolation,
                "kernel returned records without retaining their admitted byte capacities",
                None,
            )
        })?;
        // Dropping the final phase owner releases only obsolete kernel charges.
        // The transferred records and separately imported input scopes stay live.
        self.allocation_phase = None;
        Ok(retained)
    }
}
