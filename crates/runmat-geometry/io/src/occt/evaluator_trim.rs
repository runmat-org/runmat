use runmat_geometry_core::{
    ExactTrimClassifier, GeometryEvaluationControl, GeometryEvaluationError,
    GeometryEvaluationErrorKind, TrimClassifierId, TrimDomainLocation,
};

use super::{evaluator::OcctExactEvaluator, ffi};

impl ExactTrimClassifier for OcctExactEvaluator {
    fn classify(
        &self,
        id: &TrimClassifierId,
        uv: [f64; 2],
        boundary_tolerance_uv: f64,
        control: &dyn GeometryEvaluationControl,
    ) -> Result<TrimDomainLocation, GeometryEvaluationError> {
        if uv.iter().any(|value| !value.is_finite())
            || !boundary_tolerance_uv.is_finite()
            || boundary_tolerance_uv < 0.0
        {
            return Err(invalid(
                "trim point and non-negative boundary tolerance must be finite",
            ));
        }
        let face_key = self.bindings.trims.get(id).copied().ok_or_else(|| {
            GeometryEvaluationError::new(
                GeometryEvaluationErrorKind::UnknownEvaluator,
                format!("unknown OCCT trim classifier {}", id.as_str()),
            )
        })?;
        control.checkpoint()?;
        control.consume_search_work(1)?;
        let location = ffi::bridge::exact_trim_classify(
            self.session_id,
            face_key,
            uv[0],
            uv[1],
            boundary_tolerance_uv,
        )
        .map_err(|error| {
            GeometryEvaluationError::new(
                GeometryEvaluationErrorKind::ClassificationDidNotConverge,
                error.to_string(),
            )
        })?;
        control.checkpoint()?;
        match location {
            -1 => Ok(TrimDomainLocation::Outside),
            0 => Ok(TrimDomainLocation::OnBoundary),
            1 => Ok(TrimDomainLocation::Inside),
            _ => Err(invalid("OCCT trim classifier returned an invalid state")),
        }
    }
}

fn invalid(reason: impl Into<String>) -> GeometryEvaluationError {
    GeometryEvaluationError::new(GeometryEvaluationErrorKind::InvalidResult, reason)
}
