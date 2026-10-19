use super::{ExactSurfaceBoundaryError, ExactSurfaceBoundaryErrorKind};
use runmat_geometry_core::ExactBRepTopology;
use runmat_meshing_core::MeshingCancellationSignal;
use runmat_meshing_curve::SharedCurveMesh;

pub(super) fn checkpoint(
    control: Option<&dyn MeshingCancellationSignal>,
) -> Result<(), ExactSurfaceBoundaryError> {
    if control.is_some_and(MeshingCancellationSignal::is_cancelled) {
        return Err(ExactSurfaceBoundaryError::new(
            ExactSurfaceBoundaryErrorKind::Cancelled,
            None,
            "surface boundary construction or admission cancelled",
        ));
    }
    Ok(())
}

pub(super) fn validate_curves(
    curves: &SharedCurveMesh,
    topology: &ExactBRepTopology,
    control: Option<&dyn MeshingCancellationSignal>,
) -> Result<(), ExactSurfaceBoundaryError> {
    checkpoint(control)?;
    let result = match control {
        Some(control) => curves.validate_against_with_cancellation(topology, control),
        None => curves.validate_against(topology),
    };
    result.map_err(|error| {
        let kind = if error.kind
            == runmat_meshing_curve::SharedCurveErrorKind::GeometryEvaluation(
                runmat_geometry_core::GeometryEvaluationErrorKind::Cancelled,
            ) {
            ExactSurfaceBoundaryErrorKind::Cancelled
        } else {
            ExactSurfaceBoundaryErrorKind::InvalidCurveInput
        };
        ExactSurfaceBoundaryError::new(kind, error.edge_id.clone(), error.to_string())
    })
}
