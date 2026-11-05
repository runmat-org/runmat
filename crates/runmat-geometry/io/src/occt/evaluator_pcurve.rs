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
            key.face_reversed,
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
    ) -> Result<[runmat_geometry_core::GeometryParameterCoordinate; 2], GeometryEvaluationError>
    {
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
            key.face_reversed,
            key.wire,
            key.position,
            key.seam_image,
            parameter,
        )
        .map_err(query_error)?;
        control.checkpoint()?;
        valid_range(value.range_start, value.range_end)?;
        let surface = self
            .bindings
            .surface_by_face
            .get(&key.face)
            .map(|surface| format!("occt:{}:{}", self.session_id, surface.as_str()))
            .unwrap_or_else(|| format!("occt:{}:face:{}", self.session_id, key.face));
        let u = super::evaluator::parameter_frame(
            value.u_parameter_unit,
            value.parameter_scale,
            surface.clone(),
            runmat_geometry_core::GeometryParameterAxis::SurfaceU,
        )?;
        let v = super::evaluator::parameter_frame(
            value.v_parameter_unit,
            value.parameter_scale,
            surface,
            runmat_geometry_core::GeometryParameterAxis::SurfaceV,
        )?;
        let parameter = super::evaluator::parameter_frame(
            value.parameter_unit,
            value.parameter_scale,
            format!("occt:{}:{}", self.session_id, id.as_str()),
            runmat_geometry_core::GeometryParameterAxis::Pcurve,
        )?;
        PcurveDerivatives::from_parameter_coefficients(
            [value.point_u, value.point_v],
            [value.first_u, value.first_v],
            [value.second_u, value.second_v],
            [u, v],
            parameter,
        )
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
