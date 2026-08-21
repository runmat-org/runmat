use runmat_geometry_core::{
    CurveDerivatives, CurveEvaluatorId, CurveProjection, ExactCurveEvaluator,
    GeometryEvaluationControl, GeometryEvaluationError, GeometryEvaluationErrorKind,
    ParameterRange,
};

use super::{evaluator_bindings::EvaluatorBindings, ffi};
use crate::exact::ImportedExactCad;

/// Native evaluator for kernel-backed geometry from one admitted OCCT representation.
///
/// The representation is loaded once. Immutable session state is shared across calls, while
/// execution retains authority over cancellation and query-work budgets through the core trait.
pub struct OcctExactEvaluator {
    pub(super) session_id: u64,
    pub(super) bindings: EvaluatorBindings,
}

impl OcctExactEvaluator {
    pub fn new(imported: &ImportedExactCad) -> Result<Self, GeometryEvaluationError> {
        let bindings = EvaluatorBindings::from_import(imported)?;
        let session_id = ffi::bridge::start_exact_evaluator_session(
            &imported.representation,
            imported.meters_per_source_unit,
        )
        .map_err(kernel_error)?;
        Ok(Self {
            session_id,
            bindings,
        })
    }

    /// Runs the geometry-owned adaptive incidence validator across every native edge use.
    pub fn validate_incidence_consistency(
        &self,
        topology: &runmat_geometry_core::ExactBRepTopology,
        tolerance_m: f64,
        control: &dyn GeometryEvaluationControl,
    ) -> Result<(), GeometryEvaluationError> {
        runmat_geometry_core::validate_exact_incidence(topology, self, tolerance_m, control)
    }

    fn shape_key(&self, id: &CurveEvaluatorId) -> Result<u64, GeometryEvaluationError> {
        self.bindings.curves.get(id).copied().ok_or_else(|| {
            GeometryEvaluationError::new(
                GeometryEvaluationErrorKind::UnknownEvaluator,
                format!("unknown OCCT curve evaluator {}", id.as_str()),
            )
        })
    }

    fn raw_derivatives(
        &self,
        id: &CurveEvaluatorId,
        parameter: f64,
        control: &dyn GeometryEvaluationControl,
    ) -> Result<CurveDerivatives, GeometryEvaluationError> {
        control.checkpoint()?;
        control.consume_search_work(1)?;
        let value =
            ffi::bridge::exact_curve_derivatives(self.session_id, self.shape_key(id)?, parameter)
                .map_err(kernel_error)?;
        control.checkpoint()?;
        let result = CurveDerivatives {
            point_m: [value.point_x, value.point_y, value.point_z],
            first_m: [value.first_x, value.first_y, value.first_z],
            second_m: [value.second_x, value.second_y, value.second_z],
        };
        if result
            .point_m
            .into_iter()
            .chain(result.first_m)
            .chain(result.second_m)
            .any(|component| !component.is_finite())
        {
            return Err(invalid_result("OCCT curve derivatives are not finite"));
        }
        Ok(result)
    }
}

impl Drop for OcctExactEvaluator {
    fn drop(&mut self) {
        ffi::bridge::close_exact_evaluator_session(self.session_id);
    }
}

impl ExactCurveEvaluator for OcctExactEvaluator {
    fn parameter_range(
        &self,
        id: &CurveEvaluatorId,
    ) -> Result<ParameterRange, GeometryEvaluationError> {
        let value = ffi::bridge::exact_curve_range(self.session_id, self.shape_key(id)?)
            .map_err(kernel_error)?;
        if !value.start.is_finite() || !value.end.is_finite() || value.start > value.end {
            return Err(invalid_result(
                "OCCT curve returned an invalid parameter range",
            ));
        }
        Ok(ParameterRange {
            start: value.start,
            end: value.end,
        })
    }

    fn point(
        &self,
        id: &CurveEvaluatorId,
        parameter: f64,
        control: &dyn GeometryEvaluationControl,
    ) -> Result<[f64; 3], GeometryEvaluationError> {
        Ok(self.raw_derivatives(id, parameter, control)?.point_m)
    }

    fn unit_tangent(
        &self,
        id: &CurveEvaluatorId,
        parameter: f64,
        control: &dyn GeometryEvaluationControl,
    ) -> Result<[f64; 3], GeometryEvaluationError> {
        let first = self.raw_derivatives(id, parameter, control)?.first_m;
        normalized(first).ok_or_else(|| invalid_result("OCCT curve tangent is singular"))
    }

    fn derivatives(
        &self,
        id: &CurveEvaluatorId,
        parameter: f64,
        control: &dyn GeometryEvaluationControl,
    ) -> Result<CurveDerivatives, GeometryEvaluationError> {
        self.raw_derivatives(id, parameter, control)
    }

    fn curvature_1_per_m(
        &self,
        id: &CurveEvaluatorId,
        parameter: f64,
        control: &dyn GeometryEvaluationControl,
    ) -> Result<f64, GeometryEvaluationError> {
        let derivatives = self.raw_derivatives(id, parameter, control)?;
        let first_norm = norm(derivatives.first_m);
        if first_norm == 0.0 {
            return Err(invalid_result("OCCT curve curvature is singular"));
        }
        let curvature = norm(cross(derivatives.first_m, derivatives.second_m))
            / (first_norm * first_norm * first_norm);
        if !curvature.is_finite() || curvature < 0.0 {
            return Err(invalid_result("OCCT curve curvature is invalid"));
        }
        Ok(curvature)
    }

    fn arc_length_m(
        &self,
        id: &CurveEvaluatorId,
        range: ParameterRange,
        absolute_error_m: f64,
        control: &dyn GeometryEvaluationControl,
    ) -> Result<f64, GeometryEvaluationError> {
        if !absolute_error_m.is_finite() || absolute_error_m <= 0.0 {
            return Err(invalid_result(
                "curve arc-length tolerance must be positive",
            ));
        }
        control.checkpoint()?;
        control.consume_iterations(1)?;
        let length = ffi::bridge::exact_curve_arc_length(
            self.session_id,
            self.shape_key(id)?,
            range.start,
            range.end,
            absolute_error_m,
        )
        .map_err(kernel_error)?;
        control.checkpoint()?;
        if !length.is_finite() || length < 0.0 {
            return Err(invalid_result("OCCT curve arc length is invalid"));
        }
        Ok(length)
    }

    fn inverse_project(
        &self,
        id: &CurveEvaluatorId,
        point_m: [f64; 3],
        absolute_error_m: f64,
        control: &dyn GeometryEvaluationControl,
    ) -> Result<CurveProjection, GeometryEvaluationError> {
        if point_m.iter().any(|value| !value.is_finite())
            || !absolute_error_m.is_finite()
            || absolute_error_m <= 0.0
        {
            return Err(invalid_result(
                "curve projection point and positive tolerance must be finite",
            ));
        }
        control.checkpoint()?;
        control.consume_search_work(1)?;
        let value = ffi::bridge::exact_curve_inverse_project(
            self.session_id,
            self.shape_key(id)?,
            &point_m,
            absolute_error_m,
        )
        .map_err(projection_error)?;
        control.checkpoint()?;
        let result = CurveProjection {
            parameter: value.parameter,
            point_m: [value.point_x, value.point_y, value.point_z],
            distance_m: value.distance,
        };
        if !result.parameter.is_finite()
            || result.point_m.iter().any(|value| !value.is_finite())
            || !result.distance_m.is_finite()
            || result.distance_m < 0.0
        {
            return Err(invalid_result("OCCT curve projection result is invalid"));
        }
        Ok(result)
    }
}

fn normalized(value: [f64; 3]) -> Option<[f64; 3]> {
    let length = norm(value);
    (length.is_finite() && length > 0.0).then(|| value.map(|component| component / length))
}

fn norm(value: [f64; 3]) -> f64 {
    value
        .into_iter()
        .map(|component| component * component)
        .sum::<f64>()
        .sqrt()
}

fn cross(left: [f64; 3], right: [f64; 3]) -> [f64; 3] {
    [
        left[1] * right[2] - left[2] * right[1],
        left[2] * right[0] - left[0] * right[2],
        left[0] * right[1] - left[1] * right[0],
    ]
}

fn kernel_error(error: impl std::fmt::Display) -> GeometryEvaluationError {
    let reason = error.to_string();
    let kind = if reason.contains("outside the edge domain") {
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

fn invalid_result(reason: impl Into<String>) -> GeometryEvaluationError {
    GeometryEvaluationError::new(GeometryEvaluationErrorKind::InvalidResult, reason)
}

#[cfg(test)]
mod tests;
