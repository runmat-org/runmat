mod curve;
mod integration;
mod pcurve;
mod projection;
mod spline;
mod surface;
mod surface_curvature;
mod surface_projection;
mod surface_spline;
mod vector;

use super::super::{ExactBRepModel, ExactBRepTopology, GeometryContractError};
use super::{
    CurveEvaluatorId, ExactCurveEvaluatorRecord, ExactEvaluatorRegistry,
    ExactPcurveEvaluatorRecord, ExactSurfaceEvaluatorRecord, GeometryEvaluationError,
    GeometryEvaluationErrorKind, PcurveEvaluatorId, SurfaceEvaluatorId,
};

/// Portable evaluator for analytic and NURBS definitions. Construction performs
/// full topology/registry admission; kernel records remain explicit ABI-owned calls.
pub struct PortableExactEvaluator<'a> {
    registry: &'a ExactEvaluatorRegistry,
}

impl<'a> PortableExactEvaluator<'a> {
    pub fn new(
        registry: &'a ExactEvaluatorRegistry,
        topology: &ExactBRepTopology,
        model: &ExactBRepModel,
    ) -> Result<Self, GeometryContractError> {
        registry.validate_against(topology, model)?;
        Ok(Self { registry })
    }

    fn curve_record(
        &self,
        id: &CurveEvaluatorId,
    ) -> Result<&ExactCurveEvaluatorRecord, GeometryEvaluationError> {
        self.registry
            .curves
            .binary_search_by(|record| record.id.cmp(id))
            .map(|index| &self.registry.curves[index])
            .map_err(|_| unknown("curve", id.as_str()))
    }

    fn pcurve_record(
        &self,
        id: &PcurveEvaluatorId,
    ) -> Result<&ExactPcurveEvaluatorRecord, GeometryEvaluationError> {
        self.registry
            .pcurves
            .binary_search_by(|record| record.id.cmp(id))
            .map(|index| &self.registry.pcurves[index])
            .map_err(|_| unknown("pcurve", id.as_str()))
    }

    fn surface_record(
        &self,
        id: &SurfaceEvaluatorId,
    ) -> Result<&ExactSurfaceEvaluatorRecord, GeometryEvaluationError> {
        self.registry
            .surfaces
            .binary_search_by(|record| record.id.cmp(id))
            .map(|index| &self.registry.surfaces[index])
            .map_err(|_| unknown("surface", id.as_str()))
    }
}

fn unknown(kind: &str, id: &str) -> GeometryEvaluationError {
    GeometryEvaluationError::new(
        GeometryEvaluationErrorKind::UnknownEvaluator,
        format!("unknown {kind} evaluator {id}"),
    )
}

fn kernel_owned(kind: &str) -> GeometryEvaluationError {
    GeometryEvaluationError::new(
        GeometryEvaluationErrorKind::KernelUnavailable,
        format!("{kind} evaluator is owned by the admitted exact-kernel ABI"),
    )
}

fn invalid_result(reason: impl Into<String>) -> GeometryEvaluationError {
    GeometryEvaluationError::new(GeometryEvaluationErrorKind::InvalidResult, reason)
}

fn outside_domain(reason: impl Into<String>) -> GeometryEvaluationError {
    GeometryEvaluationError::new(GeometryEvaluationErrorKind::ParameterOutsideDomain, reason)
}

#[cfg(test)]
mod surface_tests;
#[cfg(test)]
mod tests;
