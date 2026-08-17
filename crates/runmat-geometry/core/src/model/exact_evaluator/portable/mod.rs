mod curve;
mod integration;
mod pcurve;
mod projection;
mod spline;
mod vector;

use super::super::{ExactBRepModelV2, ExactBRepTopologyV2, GeometryContractError};
use super::{
    CurveEvaluatorIdV2, ExactCurveEvaluatorRecordV2, ExactEvaluatorRegistryV2,
    ExactPcurveEvaluatorRecordV2, GeometryEvaluationError, GeometryEvaluationErrorKind,
    PcurveEvaluatorIdV2,
};

/// Portable evaluator for analytic and NURBS definitions. Construction performs
/// full topology/registry admission; kernel records remain explicit ABI-owned calls.
pub struct PortableExactEvaluatorV2<'a> {
    registry: &'a ExactEvaluatorRegistryV2,
}

impl<'a> PortableExactEvaluatorV2<'a> {
    pub fn new(
        registry: &'a ExactEvaluatorRegistryV2,
        topology: &ExactBRepTopologyV2,
        model: &ExactBRepModelV2,
    ) -> Result<Self, GeometryContractError> {
        registry.validate_against(topology, model)?;
        Ok(Self { registry })
    }

    fn curve_record(
        &self,
        id: &CurveEvaluatorIdV2,
    ) -> Result<&ExactCurveEvaluatorRecordV2, GeometryEvaluationError> {
        self.registry
            .curves
            .binary_search_by(|record| record.id.cmp(id))
            .map(|index| &self.registry.curves[index])
            .map_err(|_| unknown("curve", id.as_str()))
    }

    fn pcurve_record(
        &self,
        id: &PcurveEvaluatorIdV2,
    ) -> Result<&ExactPcurveEvaluatorRecordV2, GeometryEvaluationError> {
        self.registry
            .pcurves
            .binary_search_by(|record| record.id.cmp(id))
            .map(|index| &self.registry.pcurves[index])
            .map_err(|_| unknown("pcurve", id.as_str()))
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
mod tests;
