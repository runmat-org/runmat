//! Projection of existing geometry control into refinement's typed error domain.
use super::{ExactFaceRefinementError, ExactFaceRefinementErrorKind};
use runmat_geometry_core::{GeometryEvaluationControl, PersistentEntityId};

#[track_caller]
pub(super) fn checkpoint(
    control: Option<&dyn GeometryEvaluationControl>,
    face: &PersistentEntityId,
) -> Result<(), ExactFaceRefinementError> {
    if let Some(control) = control {
        control.checkpoint().map_err(|failure| {
            ExactFaceRefinementError::new(
                ExactFaceRefinementErrorKind::Metric(
                    crate::ExactFaceMetricErrorKind::GeometryEvaluation(failure.kind),
                ),
                face,
                failure.reason,
            )
        })?;
    }
    Ok(())
}
