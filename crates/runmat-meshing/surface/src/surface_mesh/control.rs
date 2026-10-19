//! Shared cancellation projection for exact surface decisions and publication.
use super::{ExactSurfaceMeshError, ExactSurfaceMeshErrorKind};
use runmat_meshing_core::MeshingCancellationSignal;

#[track_caller]
pub(super) fn checkpoint(
    control: Option<&dyn MeshingCancellationSignal>,
) -> Result<(), ExactSurfaceMeshError> {
    if let Some(control) = control {
        if !control.is_cancelled() {
            return Ok(());
        }
        return Err(ExactSurfaceMeshError::new(
            ExactSurfaceMeshErrorKind::Cancelled,
            "exact surface operation cancelled",
        ));
    }
    Ok(())
}
