use runmat_execution::resource::Capability;

#[derive(Clone, Debug, thiserror::Error, Eq, PartialEq)]
pub enum ResourcePlanningError {
    #[error("invalid FEA resource fact: {0}")]
    Invalid(String),
    #[error("FEA resource arithmetic overflow: {0}")]
    Overflow(&'static str),
    #[error(
        "accuracy resource ceiling exceeded for {resource}: requires {required}, maximum {maximum}"
    )]
    AccuracyCeiling {
        resource: &'static str,
        required: u64,
        maximum: u64,
    },
    #[error("worker inventory is insufficient: {0}")]
    Inventory(&'static str),
    #[error("worker lacks required capability `{0:?}`")]
    Capability(Capability),
    #[error(
        "worker lacks accelerator `{class}` count {count} with {memory_bytes_each} bytes each"
    )]
    Accelerator {
        class: String,
        count: u16,
        memory_bytes_each: u64,
    },
    #[error("FEA resource plan identity does not converge")]
    Identity,
    #[error("FEA resource plan encoding failed: {0}")]
    Encoding(String),
}

impl From<runmat_canonical_codec::CanonicalCodecError> for ResourcePlanningError {
    fn from(value: runmat_canonical_codec::CanonicalCodecError) -> Self {
        Self::Encoding(value.to_string())
    }
}

impl ResourcePlanningError {
    /// Preparation can reject a resource envelope before a complete plan exists.
    /// Keep that typed rejection and its original counts across the planning boundary.
    pub(crate) fn from_preparation_error(error: crate::FeaRunError) -> Self {
        match error {
            crate::FeaRunError::Resource(error) => error,
            other => Self::Invalid(other.to_string()),
        }
    }
}
