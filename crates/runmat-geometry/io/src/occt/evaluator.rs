use std::collections::BTreeMap;

use runmat_geometry_core::{
    CurveDerivatives, CurveEvaluatorId, CurveProjection, ExactCurveEvaluator,
    ExactCurveImplementation, GeometryEvaluationControl, GeometryEvaluationError,
    GeometryEvaluationErrorKind, ParameterRange,
};

use super::ffi;
use crate::exact::ImportedExactCad;

/// Native evaluator for kernel-backed curves from one admitted OCCT representation.
///
/// The representation is loaded once. Immutable session state is shared across calls, while
/// execution retains authority over cancellation and query-work budgets through the core trait.
pub struct OcctExactCurveEvaluator {
    session_id: u64,
    curve_keys: BTreeMap<CurveEvaluatorId, u64>,
}

impl OcctExactCurveEvaluator {
    pub fn new(imported: &ImportedExactCad) -> Result<Self, GeometryEvaluationError> {
        let representation_digest = imported.representation_digest();
        let mut curve_keys = BTreeMap::new();
        for record in &imported.evaluators.curves {
            let ExactCurveImplementation::Kernel { reference } = &record.implementation else {
                return Err(inconsistent(
                    "an OCCT import cannot contain a portable curve evaluator",
                ));
            };
            if reference.representation_digest != representation_digest {
                return Err(inconsistent(
                    "curve evaluator does not bind the supplied OCCT representation",
                ));
            }
            let shape_key = parse_edge_token(&reference.entity_token)?;
            if curve_keys.insert(record.id.clone(), shape_key).is_some() {
                return Err(inconsistent("duplicate OCCT curve evaluator identity"));
            }
        }
        if curve_keys.is_empty() {
            return Err(inconsistent(
                "OCCT exact geometry contains no curve evaluators",
            ));
        }
        let topology_curve_ids = imported
            .topology
            .edges
            .iter()
            .map(|edge| &edge.curve_evaluator_id)
            .collect::<std::collections::BTreeSet<_>>();
        if topology_curve_ids != curve_keys.keys().collect() {
            return Err(inconsistent(
                "OCCT curve evaluator inventory does not match exact topology",
            ));
        }
        let session_id = ffi::bridge::start_exact_evaluator_session(
            &imported.representation,
            imported.meters_per_source_unit,
        )
        .map_err(kernel_error)?;
        Ok(Self {
            session_id,
            curve_keys,
        })
    }

    fn shape_key(&self, id: &CurveEvaluatorId) -> Result<u64, GeometryEvaluationError> {
        self.curve_keys.get(id).copied().ok_or_else(|| {
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

impl Drop for OcctExactCurveEvaluator {
    fn drop(&mut self) {
        ffi::bridge::close_exact_evaluator_session(self.session_id);
    }
}

impl ExactCurveEvaluator for OcctExactCurveEvaluator {
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

fn parse_edge_token(token: &str) -> Result<u64, GeometryEvaluationError> {
    let key = token
        .strip_prefix("edge:")
        .and_then(|value| value.parse::<u64>().ok())
        .filter(|key| format!("edge:{key:020}") == token && *key != 0)
        .ok_or_else(|| inconsistent("OCCT curve evaluator has an invalid edge token"))?;
    Ok(key)
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

fn inconsistent(reason: impl Into<String>) -> GeometryEvaluationError {
    GeometryEvaluationError::new(GeometryEvaluationErrorKind::InconsistentGeometry, reason)
}

fn invalid_result(reason: impl Into<String>) -> GeometryEvaluationError {
    GeometryEvaluationError::new(GeometryEvaluationErrorKind::InvalidResult, reason)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{
        import::GeometryImportContext, import_exact_cad, ExactCadImportOptions, GeometryFormat,
    };
    use runmat_geometry_core::{GeometryEvaluationError, UnitSystem};
    use sha2::Digest as _;

    const BOX: &[u8] = include_bytes!("../../tests/fixtures/box.brep");

    struct Unlimited;

    impl GeometryEvaluationControl for Unlimited {
        fn checkpoint(&self) -> Result<(), GeometryEvaluationError> {
            Ok(())
        }

        fn consume_iterations(&self, _count: u64) -> Result<(), GeometryEvaluationError> {
            Ok(())
        }

        fn consume_search_work(&self, _count: u64) -> Result<(), GeometryEvaluationError> {
            Ok(())
        }

        fn consume_allocation_bytes(&self, _count: u64) -> Result<(), GeometryEvaluationError> {
            Ok(())
        }
    }

    struct Cancelled;

    impl GeometryEvaluationControl for Cancelled {
        fn checkpoint(&self) -> Result<(), GeometryEvaluationError> {
            Err(GeometryEvaluationError::new(
                GeometryEvaluationErrorKind::Cancelled,
                "test cancellation",
            ))
        }

        fn consume_iterations(&self, _count: u64) -> Result<(), GeometryEvaluationError> {
            unreachable!()
        }

        fn consume_search_work(&self, _count: u64) -> Result<(), GeometryEvaluationError> {
            unreachable!()
        }

        fn consume_allocation_bytes(&self, _count: u64) -> Result<(), GeometryEvaluationError> {
            unreachable!()
        }
    }

    #[test]
    fn imported_curve_queries_are_exact_scaled_and_digest_bound() {
        let imported = import_exact_cad(
            "box.brep",
            BOX,
            GeometryFormat::Brep,
            &ExactCadImportOptions::default(),
            &GeometryImportContext::new(),
        )
        .unwrap();
        let raw_digest: [u8; 32] = sha2::Sha256::digest(&imported.representation).into();
        assert_eq!(imported.representation_digest(), raw_digest);
        let evaluator = OcctExactCurveEvaluator::new(&imported).unwrap();
        let id = &imported.topology.edges[0].curve_evaluator_id;
        let range = evaluator.parameter_range(id).unwrap();
        let start = evaluator.point(id, range.start, &Unlimited).unwrap();
        let end = evaluator.point(id, range.end, &Unlimited).unwrap();
        let expected_length = norm([end[0] - start[0], end[1] - start[1], end[2] - start[2]]);
        let length = evaluator
            .arc_length_m(id, range, 1.0e-12, &Unlimited)
            .unwrap();
        assert!((length - expected_length).abs() < 1.0e-12);

        let parameter = (range.start + range.end) * 0.5;
        let point = evaluator.point(id, parameter, &Unlimited).unwrap();
        let tangent = evaluator.unit_tangent(id, parameter, &Unlimited).unwrap();
        assert!((norm(tangent) - 1.0).abs() < 1.0e-12);
        assert_eq!(
            evaluator
                .curvature_1_per_m(id, parameter, &Unlimited)
                .unwrap(),
            0.0
        );
        let projection = evaluator
            .inverse_project(id, point, 1.0e-12, &Unlimited)
            .unwrap();
        assert!((projection.parameter - parameter).abs() < 1.0e-12);
        assert!(projection.distance_m < 1.0e-12);
        assert_eq!(
            evaluator
                .point(id, range.end + 1.0, &Unlimited)
                .unwrap_err()
                .kind,
            GeometryEvaluationErrorKind::ParameterOutsideDomain
        );
        assert_eq!(
            evaluator.point(id, parameter, &Cancelled).unwrap_err().kind,
            GeometryEvaluationErrorKind::Cancelled
        );
        assert_eq!(
            evaluator
                .parameter_range(&CurveEvaluatorId::new("curve:unknown").unwrap())
                .unwrap_err()
                .kind,
            GeometryEvaluationErrorKind::UnknownEvaluator
        );

        let millimeter_options = ExactCadImportOptions {
            source_units: UnitSystem::Millimeter,
            ..ExactCadImportOptions::default()
        };
        let millimeter_import = import_exact_cad(
            "box.brep",
            BOX,
            GeometryFormat::Brep,
            &millimeter_options,
            &GeometryImportContext::new(),
        )
        .unwrap();
        let millimeter_evaluator = OcctExactCurveEvaluator::new(&millimeter_import).unwrap();
        let millimeter_length = millimeter_evaluator
            .arc_length_m(
                &millimeter_import.topology.edges[0].curve_evaluator_id,
                range,
                1.0e-15,
                &Unlimited,
            )
            .unwrap();
        assert!((millimeter_length - length * 0.001).abs() < 1.0e-15);

        let mut corrupt = imported.clone();
        corrupt.representation[0] ^= 1;
        assert_eq!(
            OcctExactCurveEvaluator::new(&corrupt).err().unwrap().kind,
            GeometryEvaluationErrorKind::InconsistentGeometry
        );
    }
}
