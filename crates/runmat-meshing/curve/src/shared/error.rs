use runmat_geometry_core::{GeometryEvaluationErrorKind, PersistentEntityId};

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct SharedCurveError {
    pub edge_id: Option<PersistentEntityId>,
    pub kind: SharedCurveErrorKind,
    pub field: String,
    pub reason: String,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum SharedCurveErrorKind {
    InvalidContract,
    InvalidEncoding,
    InvalidRequest,
    GeometryEvaluation(GeometryEvaluationErrorKind),
    MetricEvaluation,
    ResourceLimit,
    UnsatisfiedConstraint,
    GeometricMismatch,
}

impl SharedCurveError {
    pub(crate) fn new(
        kind: SharedCurveErrorKind,
        field: impl Into<String>,
        reason: impl Into<String>,
    ) -> Self {
        Self {
            edge_id: None,
            kind,
            field: field.into(),
            reason: reason.into(),
        }
    }

    pub(crate) fn invalid_contract(field: impl Into<String>, reason: impl Into<String>) -> Self {
        Self::new(SharedCurveErrorKind::InvalidContract, field, reason)
    }

    pub(crate) fn invalid_request(field: impl Into<String>, reason: impl Into<String>) -> Self {
        Self::new(SharedCurveErrorKind::InvalidRequest, field, reason)
    }

    pub(crate) fn invalid_encoding(field: impl Into<String>, reason: impl Into<String>) -> Self {
        Self::new(SharedCurveErrorKind::InvalidEncoding, field, reason)
    }

    pub(crate) fn for_edge(mut self, edge_id: &PersistentEntityId) -> Self {
        if self.edge_id.is_none() {
            self.edge_id = Some(edge_id.clone());
        }
        self
    }
}

impl std::fmt::Display for SharedCurveError {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        if let Some(edge) = &self.edge_id {
            write!(
                formatter,
                "shared curve {:?} for {:?}, invalid {}: {}",
                self.kind, edge, self.field, self.reason
            )
        } else {
            write!(
                formatter,
                "shared curve {:?}, invalid {}: {}",
                self.kind, self.field, self.reason
            )
        }
    }
}

impl std::error::Error for SharedCurveError {}
