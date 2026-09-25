use runmat_execution::resource::Capability;

#[derive(Clone, Debug, thiserror::Error, Eq, PartialEq)]
pub enum ResourcePlanningError {
    #[error("invalid FEA resource fact: {0}")]
    Invalid(String),
    #[error(
        "coercivity proof work on the structural stiffness upper-bound pattern is indeterminate; the prepared operator may still be admissible"
    )]
    IndeterminateCoercivityUpperBound,
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
