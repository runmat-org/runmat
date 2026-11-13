//! Requested ownership of a reconstructed partition, separate from decoder scratch.
use super::*;
use runmat_geometry_core::{
    GeometryEvaluationControl, GeometryEvaluationError, GeometryEvaluationErrorKind,
    PersistentEntityId,
};
fn overflow() -> GeometryEvaluationError {
    GeometryEvaluationError::new(
        GeometryEvaluationErrorKind::AllocationBudgetExceeded,
        "retained face partition inventory overflow",
    )
}
fn add(bytes: &mut u64, value: u64) -> Result<(), GeometryEvaluationError> {
    *bytes = bytes.checked_add(value).ok_or_else(overflow)?;
    Ok(())
}
fn entity_bytes(entity: &PersistentEntityId) -> Result<u64, GeometryEvaluationError> {
    let mut bytes = (entity.assembly_path.capacity() as u64)
        .checked_mul(std::mem::size_of::<String>() as u64)
        .ok_or_else(overflow)?;
    add(&mut bytes, entity.source_topology_id.capacity() as u64)?;
    for path in &entity.assembly_path {
        add(&mut bytes, path.capacity() as u64)?;
    }
    Ok(bytes)
}
impl ExactFacePartitionResult {
    pub fn retained_storage_bytes(
        &self,
        control: &dyn GeometryEvaluationControl,
    ) -> Result<u64, GeometryEvaluationError> {
        control.checkpoint()?;
        let mut bytes = std::mem::size_of::<Self>() as u64;
        add(
            &mut bytes,
            self.partition
                .owned_buffer_bytes()
                .map_err(|_| overflow())?,
        )?;
        match &self.outcome {
            ExactFacePartitionOutcome::Converged { faces } => {
                add(
                    &mut bytes,
                    (faces.capacity() as u64)
                        .checked_mul(std::mem::size_of::<crate::ExactFaceMesh>() as u64)
                        .ok_or_else(overflow)?,
                )?;
                for face in faces {
                    control.checkpoint()?;
                    add(
                        &mut bytes,
                        face.retained_storage_bytes(control)?
                            .checked_sub(std::mem::size_of::<crate::ExactFaceMesh>() as u64)
                            .ok_or_else(overflow)?,
                    )?;
                }
            }
            ExactFacePartitionOutcome::RequiresCurveSplits { splits } => {
                add(
                    &mut bytes,
                    (splits.capacity() as u64)
                        .checked_mul(std::mem::size_of::<
                            runmat_meshing_curve::SharedCurveSegmentSplit,
                        >() as u64)
                        .ok_or_else(overflow)?,
                )?;
                for split in splits {
                    control.checkpoint()?;
                    add(&mut bytes, entity_bytes(&split.source_edge_id)?)?;
                }
            }
        }
        Ok(bytes)
    }
}
