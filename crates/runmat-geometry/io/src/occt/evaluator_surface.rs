use runmat_geometry_core::{
    surface_principal_curvature, surface_unit_normal, ExactSurfaceEvaluator,
    GeometryEvaluationControl, GeometryEvaluationError, GeometryEvaluationErrorKind,
    ParameterRange, SurfaceCurvature, SurfaceDerivatives, SurfaceEvaluatorId, SurfaceProjection,
};

use super::{evaluator::OcctExactEvaluator, ffi};

impl ExactSurfaceEvaluator for OcctExactEvaluator {
    fn parameter_bounds(
        &self,
        id: &SurfaceEvaluatorId,
    ) -> Result<[ParameterRange; 2], GeometryEvaluationError> {
        let value = self.properties(id)?;
        Ok([
            valid_range(value.u_start, value.u_end)?,
            valid_range(value.v_start, value.v_end)?,
        ])
    }

    fn periodicity(
        &self,
        id: &SurfaceEvaluatorId,
    ) -> Result<[Option<f64>; 2], GeometryEvaluationError> {
        let value = self.properties(id)?;
        Ok([
            valid_period(value.u_periodic, value.u_period)?,
            valid_period(value.v_periodic, value.v_period)?,
        ])
    }

    fn point(
        &self,
        id: &SurfaceEvaluatorId,
        uv: [f64; 2],
        control: &dyn GeometryEvaluationControl,
    ) -> Result<[f64; 3], GeometryEvaluationError> {
        Ok(self.surface_derivatives(id, uv, control)?.point_m)
    }

    fn derivatives(
        &self,
        id: &SurfaceEvaluatorId,
        uv: [f64; 2],
        control: &dyn GeometryEvaluationControl,
    ) -> Result<SurfaceDerivatives, GeometryEvaluationError> {
        self.surface_derivatives(id, uv, control)
    }

    fn unit_normal(
        &self,
        id: &SurfaceEvaluatorId,
        uv: [f64; 2],
        control: &dyn GeometryEvaluationControl,
    ) -> Result<[f64; 3], GeometryEvaluationError> {
        surface_unit_normal(&self.surface_derivatives(id, uv, control)?)
    }

    fn principal_curvature(
        &self,
        id: &SurfaceEvaluatorId,
        uv: [f64; 2],
        control: &dyn GeometryEvaluationControl,
    ) -> Result<SurfaceCurvature, GeometryEvaluationError> {
        surface_principal_curvature(&self.surface_derivatives(id, uv, control)?)
    }

    fn closest_point(
        &self,
        id: &SurfaceEvaluatorId,
        point_m: [f64; 3],
        absolute_error_m: f64,
        control: &dyn GeometryEvaluationControl,
    ) -> Result<SurfaceProjection, GeometryEvaluationError> {
        if point_m.iter().any(|value| !value.is_finite())
            || !absolute_error_m.is_finite()
            || absolute_error_m <= 0.0
        {
            return Err(invalid(
                "surface projection point and positive tolerance must be finite",
            ));
        }
        control.checkpoint()?;
        control.consume_search_work(1)?;
        let value = ffi::bridge::exact_surface_closest_point(
            self.session_id,
            self.surface_key(id)?,
            &point_m,
            absolute_error_m,
        )
        .map_err(projection_error)?;
        control.checkpoint()?;
        let result = SurfaceProjection {
            uv: [value.u, value.v],
            point_m: [value.point_x, value.point_y, value.point_z],
            distance_m: value.distance,
        };
        if result
            .uv
            .into_iter()
            .chain(result.point_m)
            .chain([result.distance_m])
            .any(|value| !value.is_finite())
            || result.distance_m < 0.0
        {
            return Err(invalid("OCCT surface projection result is invalid"));
        }
        let [u, v] = self.parameter_bounds(id)?;
        if result.uv[0] < u.start
            || result.uv[0] > u.end
            || result.uv[1] < v.start
            || result.uv[1] > v.end
        {
            return Err(invalid("OCCT surface projection is outside its domain"));
        }
        Ok(result)
    }
}

impl OcctExactEvaluator {
    fn surface_key(&self, id: &SurfaceEvaluatorId) -> Result<u64, GeometryEvaluationError> {
        self.surface_keys.get(id).copied().ok_or_else(|| {
            GeometryEvaluationError::new(
                GeometryEvaluationErrorKind::UnknownEvaluator,
                format!("unknown OCCT surface evaluator {}", id.as_str()),
            )
        })
    }

    fn properties(
        &self,
        id: &SurfaceEvaluatorId,
    ) -> Result<ffi::bridge::OcctSurfacePropertiesPayload, GeometryEvaluationError> {
        ffi::bridge::exact_surface_properties(self.session_id, self.surface_key(id)?)
            .map_err(query_error)
    }

    fn surface_derivatives(
        &self,
        id: &SurfaceEvaluatorId,
        uv: [f64; 2],
        control: &dyn GeometryEvaluationControl,
    ) -> Result<SurfaceDerivatives, GeometryEvaluationError> {
        if uv.iter().any(|value| !value.is_finite()) {
            return Err(invalid("surface parameters must be finite"));
        }
        control.checkpoint()?;
        control.consume_search_work(1)?;
        let value = ffi::bridge::exact_surface_derivatives(
            self.session_id,
            self.surface_key(id)?,
            uv[0],
            uv[1],
        )
        .map_err(query_error)?;
        control.checkpoint()?;
        let result = SurfaceDerivatives {
            point_m: [value.point_x, value.point_y, value.point_z],
            du_m: [value.du_x, value.du_y, value.du_z],
            dv_m: [value.dv_x, value.dv_y, value.dv_z],
            duu_m: [value.duu_x, value.duu_y, value.duu_z],
            duv_m: [value.duv_x, value.duv_y, value.duv_z],
            dvv_m: [value.dvv_x, value.dvv_y, value.dvv_z],
        };
        if result
            .point_m
            .into_iter()
            .chain(result.du_m)
            .chain(result.dv_m)
            .chain(result.duu_m)
            .chain(result.duv_m)
            .chain(result.dvv_m)
            .any(|value| !value.is_finite())
        {
            return Err(invalid("OCCT surface derivatives are not finite"));
        }
        Ok(result)
    }
}

fn valid_range(start: f64, end: f64) -> Result<ParameterRange, GeometryEvaluationError> {
    if !start.is_finite() || !end.is_finite() || start > end {
        return Err(invalid("OCCT surface returned an invalid parameter range"));
    }
    Ok(ParameterRange { start, end })
}

fn valid_period(periodic: bool, period: f64) -> Result<Option<f64>, GeometryEvaluationError> {
    if !periodic {
        return Ok(None);
    }
    if !period.is_finite() || period <= 0.0 {
        return Err(invalid("OCCT surface returned an invalid period"));
    }
    Ok(Some(period))
}

fn query_error(error: impl std::fmt::Display) -> GeometryEvaluationError {
    let reason = error.to_string();
    let kind = if reason.contains("outside the face domain") {
        GeometryEvaluationErrorKind::ParameterOutsideDomain
    } else {
        GeometryEvaluationErrorKind::KernelFailure
    };
    GeometryEvaluationError::new(kind, reason)
}

fn projection_error(error: impl std::fmt::Display) -> GeometryEvaluationError {
    let reason = error.to_string();
    let kind = if reason.contains("projection did not converge") {
        GeometryEvaluationErrorKind::ProjectionDidNotConverge
    } else {
        GeometryEvaluationErrorKind::KernelFailure
    };
    GeometryEvaluationError::new(kind, reason)
}

fn invalid(reason: impl Into<String>) -> GeometryEvaluationError {
    GeometryEvaluationError::new(GeometryEvaluationErrorKind::InvalidResult, reason)
}
