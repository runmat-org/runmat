use runmat_geometry_core::{GeometryEvaluationErrorKind, PersistentEntityId};
use runmat_meshing_size::metric::MetricSourceKind;

#[derive(Clone, Copy, Debug, PartialEq)]
pub struct ParametricMetricTensor {
    pub uu: f64,
    pub uv: f64,
    pub vv: f64,
}

impl ParametricMetricTensor {
    pub fn validate(&self) -> Result<(), &'static str> {
        let determinant = self.uu * self.vv - self.uv * self.uv;
        if !self.uu.is_finite()
            || !self.uv.is_finite()
            || !self.vv.is_finite()
            || !determinant.is_finite()
            || self.uu <= 0.0
            || determinant <= 0.0
        {
            Err("must be a finite symmetric positive-definite parametric tensor")
        } else {
            Ok(())
        }
    }

    pub fn squared_length(&self, delta_uv: [f64; 2]) -> Result<f64, &'static str> {
        self.validate()?;
        if delta_uv.iter().any(|value| !value.is_finite()) {
            return Err("parametric displacement must be finite");
        }
        let squared = self.uu * delta_uv[0] * delta_uv[0]
            + 2.0 * self.uv * delta_uv[0] * delta_uv[1]
            + self.vv * delta_uv[1] * delta_uv[1];
        if squared.is_finite() && squared >= 0.0 {
            Ok(squared)
        } else {
            Err("parametric metric length is invalid")
        }
    }
}

#[derive(Clone, Debug, PartialEq)]
pub struct ExactFaceMetricEvaluation {
    pub source_face_id: PersistentEntityId,
    pub uv: [f64; 2],
    pub point_m: [f64; 3],
    pub derivative_u_m: [f64; 3],
    pub derivative_v_m: [f64; 3],
    pub physical_metric: ParametricMetricTensor,
    pub sizing_metric: ParametricMetricTensor,
    pub active_sources: Vec<MetricSourceKind>,
    pub applied_contribution_count: u32,
    pub clipped_contribution_count: u32,
    pub rejected_contribution_count: u32,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum ExactFaceMetricErrorKind {
    InvalidRequest,
    UnknownFace,
    GeometryEvaluation(GeometryEvaluationErrorKind),
    InvalidEvaluation,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct ExactFaceMetricError {
    pub kind: ExactFaceMetricErrorKind,
    pub source_face_id: Option<PersistentEntityId>,
    pub reason: String,
}

impl ExactFaceMetricError {
    pub(super) fn new(
        kind: ExactFaceMetricErrorKind,
        source_face_id: Option<&PersistentEntityId>,
        reason: impl Into<String>,
    ) -> Self {
        Self {
            kind,
            source_face_id: source_face_id.cloned(),
            reason: reason.into(),
        }
    }
}

impl std::fmt::Display for ExactFaceMetricError {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(
            formatter,
            "exact face metric {:?} for {:?}: {}",
            self.kind, self.source_face_id, self.reason
        )
    }
}

impl std::error::Error for ExactFaceMetricError {}
