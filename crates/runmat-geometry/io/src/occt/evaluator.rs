use runmat_geometry_core::{
    CurveDerivatives, CurveEvaluatorId, CurveProjection, ExactCurveEvaluator,
    GeometryEvaluationControl, GeometryEvaluationError, GeometryEvaluationErrorKind,
    ParameterRange,
};
#[cfg(not(target_os = "emscripten"))]
use std::{sync::mpsc, time::Duration};

use super::{evaluator_bindings::EvaluatorBindings, ffi};
use crate::exact::ImportedExactCad;

mod session;

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
        Self::start(
            &imported.representation,
            imported.kernel_units.meters_per_unit(),
            bindings,
        )
    }

    /// Reconstructs an evaluator from the canonical exact-geometry closure transferred to a
    /// worker. Kernel-only mass-property records require private import-time shape inventory and
    /// are rejected; kernel-validated mass properties are fully reconstructible.
    pub fn from_closure(
        representation: &[u8],
        meters_per_source_unit: f64,
        topology: &runmat_geometry_core::ExactBRepTopology,
        evaluators: &runmat_geometry_core::ExactEvaluatorRegistry,
    ) -> Result<Self, GeometryEvaluationError> {
        if !meters_per_source_unit.is_finite() || meters_per_source_unit <= 0.0 {
            return Err(invalid_result(
                "OCCT evaluator unit scale must be finite and positive",
            ));
        }
        let bindings = EvaluatorBindings::from_closure(representation, topology, evaluators, None)?;
        Self::start(representation, meters_per_source_unit, bindings)
    }

    /// Reconstructs a transferred evaluator while preserving the execution
    /// host's cancellation and time-checkpoint contract during native session
    /// startup.
    pub fn from_closure_controlled(
        representation: &[u8],
        meters_per_source_unit: f64,
        topology: &runmat_geometry_core::ExactBRepTopology,
        evaluators: &runmat_geometry_core::ExactEvaluatorRegistry,
        control: &dyn GeometryEvaluationControl,
    ) -> Result<Self, GeometryEvaluationError> {
        control.checkpoint()?;
        control.consume_allocation_bytes(representation.len() as u64)?;
        if !meters_per_source_unit.is_finite() || meters_per_source_unit <= 0.0 {
            return Err(invalid_result(
                "OCCT evaluator unit scale must be finite and positive",
            ));
        }
        let bindings = EvaluatorBindings::from_closure(representation, topology, evaluators, None)?;
        control.checkpoint()?;
        let face_keys = session::face_keys(&bindings, Some(control))?;
        // The browser sidecar already runs off the UI thread. Emscripten's
        // single-core host cannot spawn a native helper thread.
        #[cfg(target_os = "emscripten")]
        let session_id = ffi::bridge::start_exact_evaluator_session(
            representation,
            meters_per_source_unit,
            &face_keys,
        )
        .map_err(kernel_error)?;
        #[cfg(not(target_os = "emscripten"))]
        let session_id = {
            let owned_representation = representation.to_vec();
            let (sender, receiver) = mpsc::sync_channel(1);
            std::thread::spawn(move || {
                let result = ffi::bridge::start_exact_evaluator_session(
                    &owned_representation,
                    meters_per_source_unit,
                    &face_keys,
                )
                .map_err(kernel_error);
                match result {
                    Ok(session_id) => {
                        if sender.send(Ok(session_id)).is_err() {
                            ffi::bridge::close_exact_evaluator_session(session_id);
                        }
                    }
                    Err(error) => {
                        let _ = sender.send(Err(error));
                    }
                }
            });
            loop {
                match receiver.recv_timeout(Duration::from_millis(10)) {
                    Ok(result) => break result?,
                    Err(mpsc::RecvTimeoutError::Timeout) => control.checkpoint()?,
                    Err(mpsc::RecvTimeoutError::Disconnected) => {
                        return Err(invalid_result(
                            "OCCT evaluator session worker stopped before returning a result",
                        ));
                    }
                }
            }
        };
        let evaluator = Self {
            session_id,
            bindings,
        };
        // Own the session before the final checkpoint so cancellation releases it.
        control.checkpoint()?;
        Ok(evaluator)
    }

    fn start(
        representation: &[u8],
        meters_per_source_unit: f64,
        bindings: EvaluatorBindings,
    ) -> Result<Self, GeometryEvaluationError> {
        let face_keys = session::face_keys(&bindings, None)?;
        let session_id = ffi::bridge::start_exact_evaluator_session(
            representation,
            meters_per_source_unit,
            &face_keys,
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
        let frame = parameter_frame(
            value.parameter_unit,
            value.parameter_scale,
            format!("occt:{}:{}", self.session_id, id.as_str()),
            runmat_geometry_core::GeometryParameterAxis::Curve,
        )?;
        let result = CurveDerivatives::from_parameter_coefficients(
            [value.point_x, value.point_y, value.point_z],
            [value.first_x, value.first_y, value.first_z],
            [value.second_x, value.second_y, value.second_z],
            frame,
        )?;
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
    ) -> Result<runmat_geometry_core::physical::Position, GeometryEvaluationError> {
        Ok(self.raw_derivatives(id, parameter, control)?.point_m)
    }

    fn unit_tangent(
        &self,
        id: &CurveEvaluatorId,
        parameter: f64,
        control: &dyn GeometryEvaluationControl,
    ) -> Result<[f64; 3], GeometryEvaluationError> {
        let first = self.raw_derivatives(id, parameter, control)?.first_m;
        first
            .with_canonical_storage(|values, _axes| normalized(*values))
            .ok_or_else(|| invalid_result("OCCT curve tangent is singular"))
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
    ) -> Result<runmat_geometry_core::physical::Curvature, GeometryEvaluationError> {
        let derivatives = self.raw_derivatives(id, parameter, control)?;
        let first_norm = norm(
            derivatives
                .first_m
                .with_canonical_storage(|values, _axes| *values),
        );
        if first_norm == 0.0 {
            return Err(invalid_result("OCCT curve curvature is singular"));
        }
        let curvature = norm(cross(
            derivatives
                .first_m
                .with_canonical_storage(|values, _axes| *values),
            derivatives
                .second_m
                .with_canonical_storage(|values, _axes| *values),
        )) / (first_norm * first_norm * first_norm);
        if !curvature.is_finite() || curvature < 0.0 {
            return Err(invalid_result("OCCT curve curvature is invalid"));
        }
        runmat_geometry_core::physical::Curvature::try_new(curvature)
            .map_err(|e| invalid_result(e.to_string()))
    }

    fn arc_length_m(
        &self,
        id: &CurveEvaluatorId,
        range: ParameterRange,
        absolute_error_m: runmat_geometry_core::physical::Length,
        control: &dyn GeometryEvaluationControl,
    ) -> Result<runmat_geometry_core::physical::Length, GeometryEvaluationError> {
        let absolute_error_m = absolute_error_m.canonical_value();
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
        runmat_geometry_core::physical::Length::try_new(length)
            .map_err(|e| invalid_result(e.to_string()))
    }

    fn inverse_project(
        &self,
        id: &CurveEvaluatorId,
        point_m: runmat_geometry_core::physical::Position,
        absolute_error_m: runmat_geometry_core::physical::Length,
        control: &dyn GeometryEvaluationControl,
    ) -> Result<CurveProjection, GeometryEvaluationError> {
        let point_m = point_m.canonical_value();
        let absolute_error_m = absolute_error_m.canonical_value();
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
        CurveProjection::from_parameter_coefficients(
            value.parameter,
            [value.point_x, value.point_y, value.point_z],
            value.distance,
            parameter_frame(
                value.parameter_unit,
                value.parameter_scale,
                format!("occt:{}:{}", self.session_id, id.as_str()),
                runmat_geometry_core::GeometryParameterAxis::Curve,
            )?,
        )
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

pub(super) fn parameter_frame(
    unit: u8,
    scale: f64,
    evaluator: String,
    axis: runmat_geometry_core::GeometryParameterAxis,
) -> Result<runmat_geometry_core::GeometryParameterFrame, GeometryEvaluationError> {
    use runmat_geometry_core::GeometryParameterFrame;
    match unit {
        0 => Ok(GeometryParameterFrame::native(evaluator, axis)),
        1 => {
            GeometryParameterFrame::length(scale).map_err(|error| invalid_result(error.to_string()))
        }
        2 => Ok(GeometryParameterFrame::radians()),
        _ => Err(invalid_result(
            "OCCT returned an unknown parameter unit tag",
        )),
    }
}
