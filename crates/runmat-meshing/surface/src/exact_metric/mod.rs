pub(crate) mod prepared_validation;
mod resolved;
mod types;
mod validate;

pub use resolved::ResolvedFaceMetricField;
pub use types::{
    ExactFaceMetricError, ExactFaceMetricErrorKind, ExactFaceMetricEvaluation,
    ExactFaceMetricProvenance, ParametricMetricTensor,
};
pub use validate::{
    validate_exact_face_metric_evaluation,
    validate_exact_face_metric_evaluation_in_parameterization,
};

#[cfg(test)]
mod tests;
