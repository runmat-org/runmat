#[derive(Debug, Clone, PartialEq, Eq)]
pub struct MeshingContractError {
    pub field: String,
    pub reason: String,
}

impl MeshingContractError {
    pub fn invalid(field: impl Into<String>, reason: impl Into<String>) -> Self {
        Self {
            field: field.into(),
            reason: reason.into(),
        }
    }
}

impl std::fmt::Display for MeshingContractError {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(formatter, "invalid {}: {}", self.field, self.reason)
    }
}

impl std::error::Error for MeshingContractError {}

impl From<runmat_geometry_core::GeometryContractError> for MeshingContractError {
    fn from(error: runmat_geometry_core::GeometryContractError) -> Self {
        Self {
            field: error.field,
            reason: error.reason,
        }
    }
}

impl From<runmat_meshing_size::metric::MetricContractError> for MeshingContractError {
    fn from(error: runmat_meshing_size::metric::MetricContractError) -> Self {
        Self {
            field: error.field,
            reason: error.reason,
        }
    }
}
