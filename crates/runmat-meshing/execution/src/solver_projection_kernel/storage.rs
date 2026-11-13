//! Field order releases owned data before its reservation, including on errors.
use super::*;
use crate::codec_control::CodecControl;
use runmat_geometry_core::GeometryAllocationReservation;
use runmat_meshing_core::ValidatedSolverMeshProjection;

pub(super) struct AdmittedProjection<'a> {
    projection: ValidatedSolverMeshProjection,
    storage: GeometryAllocationReservation<'a>,
}
impl std::ops::Deref for AdmittedProjection<'_> {
    type Target = ValidatedSolverMeshProjection;
    fn deref(&self) -> &Self::Target {
        &self.projection
    }
}
impl AdmittedProjection<'_> {
    pub fn retained_bytes(&self) -> u64 {
        self.storage.bytes()
    }
}

pub(super) fn admit<'a>(
    projection: ValidatedSolverMeshProjection,
    scope: &runmat_geometry_core::GeometryAllocationScope<'a>,
    codec: &CodecControl<'_>,
) -> Result<AdmittedProjection<'a>, Box<MeshingFailure>> {
    let retained_bytes = projection
        .retained_storage_bytes(u64::MAX, &mut || {
            if MeshingCancellationSignal::is_cancelled(codec) {
                Err(runmat_meshing_core::MeshingContractError::invalid(
                    "solver projection retention",
                    "control interrupted",
                ))
            } else {
                Ok(())
            }
        })
        .map_err(|error| match codec.take_failure() {
            Some(original) => failure(
                crate::geometry_failure::category(original.kind),
                &original.reason,
            ),
            None => failure(
                MeshingFailureCategory::MemoryBudgetExceeded,
                &error.to_string(),
            ),
        })?;
    let storage = scope
        .transfer_reservation(retained_bytes)
        .map_err(|error| failure(crate::geometry_failure::category(error.kind), &error.reason))?;
    Ok(AdmittedProjection {
        projection,
        storage,
    })
}
