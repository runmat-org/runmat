use runmat_geometry_core::{
    ExactPcurveEvaluator, GeometryEvaluationControl, GeometryEvaluationError,
    GeometryEvaluationErrorKind, ParameterRange, PcurveDerivatives, PcurveEvaluatorId,
};

use super::{evaluator::OcctExactEvaluator, ffi};

impl ExactPcurveEvaluator for OcctExactEvaluator {
    fn parameter_range(
        &self,
        id: &PcurveEvaluatorId,
    ) -> Result<ParameterRange, GeometryEvaluationError> {
        let key = self.pcurve_key(id)?;
        let value = ffi::bridge::exact_pcurve_range(
            self.session_id,
            key.face,
            key.wire,
            key.position,
            key.seam_image,
        )
        .map_err(query_error)?;
        valid_range(value.start, value.end)
    }

    fn point(
        &self,
        id: &PcurveEvaluatorId,
        parameter: f64,
        control: &dyn GeometryEvaluationControl,
    ) -> Result<[f64; 2], GeometryEvaluationError> {
        Ok(self.pcurve_derivatives(id, parameter, control)?.point_uv)
    }

    fn derivatives(
        &self,
        id: &PcurveEvaluatorId,
        parameter: f64,
        control: &dyn GeometryEvaluationControl,
    ) -> Result<PcurveDerivatives, GeometryEvaluationError> {
        self.pcurve_derivatives(id, parameter, control)
    }
}

impl OcctExactEvaluator {
    fn pcurve_key(
        &self,
        id: &PcurveEvaluatorId,
    ) -> Result<super::evaluator_bindings::PcurveKey, GeometryEvaluationError> {
        self.bindings.pcurves.get(id).copied().ok_or_else(|| {
            GeometryEvaluationError::new(
                GeometryEvaluationErrorKind::UnknownEvaluator,
                format!("unknown OCCT pcurve evaluator {}", id.as_str()),
            )
        })
    }

    fn pcurve_derivatives(
        &self,
        id: &PcurveEvaluatorId,
        parameter: f64,
        control: &dyn GeometryEvaluationControl,
    ) -> Result<PcurveDerivatives, GeometryEvaluationError> {
        control.checkpoint()?;
        control.consume_search_work(1)?;
        let key = self.pcurve_key(id)?;
        let value = ffi::bridge::exact_pcurve_derivatives(
            self.session_id,
            key.face,
            key.wire,
            key.position,
            key.seam_image,
            parameter,
        )
        .map_err(query_error)?;
        control.checkpoint()?;
        valid_range(value.range_start, value.range_end)?;
        let result = PcurveDerivatives {
            point_uv: [value.point_u, value.point_v],
            first_uv: [value.first_u, value.first_v],
            second_uv: [value.second_u, value.second_v],
        };
        if result
            .point_uv
            .into_iter()
            .chain(result.first_uv)
            .chain(result.second_uv)
            .any(|component| !component.is_finite())
        {
            return Err(invalid("OCCT pcurve derivatives are not finite"));
        }
        Ok(result)
    }
}

fn valid_range(start: f64, end: f64) -> Result<ParameterRange, GeometryEvaluationError> {
    if !start.is_finite() || !end.is_finite() || start > end {
        return Err(invalid("OCCT pcurve returned an invalid parameter range"));
    }
    Ok(ParameterRange { start, end })
}

fn query_error(error: impl std::fmt::Display) -> GeometryEvaluationError {
    let reason = error.to_string();
    let kind = if reason.contains("outside the edge domain") {
        GeometryEvaluationErrorKind::ParameterOutsideDomain
    } else {
        GeometryEvaluationErrorKind::KernelFailure
    };
    GeometryEvaluationError::new(kind, reason)
}

fn invalid(reason: impl Into<String>) -> GeometryEvaluationError {
    GeometryEvaluationError::new(GeometryEvaluationErrorKind::InvalidResult, reason)
}
