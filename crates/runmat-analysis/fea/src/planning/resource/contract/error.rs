use runmat_execution::resource::Capability;

#[derive(Clone, Debug, thiserror::Error, Eq, PartialEq)]
pub enum ResourcePlanningError {
    #[error("FEA_CANCELLED: resource planning cancelled")]
    Cancelled,
    /// A numeric array's encoding exceeds the platform's largest single allocation.
    #[error(
        "canonical numeric inventory requires {required} encoded bytes, above the platform allocation ceiling of {maximum}"
    )]
    CanonicalWireCapacity { required: u64, maximum: u64 },
    /// The plan's memory or scratch request exceeds what the worker's admitted budget offers.
    /// Available memory is min(worker inventory, authored resident budget).
    #[error(
        "the study needs {required_memory_bytes} bytes of memory and {required_scratch_bytes} bytes of scratch; the admitted budget offers {available_memory_bytes} bytes of memory and {available_scratch_bytes} bytes of scratch"
    )]
    MemoryBudget {
        required_memory_bytes: u64,
        available_memory_bytes: u64,
        required_scratch_bytes: u64,
        available_scratch_bytes: u64,
    },
    #[error("invalid FEA resource fact: {0}")]
    Invalid(String),
    #[error("FEA resource numerical accounting failed: {0}")]
    Accounting(#[from] runmat_simulation_usage::LedgerError),
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
    /// Neither static solve path fits the effective resident ceiling, min(authored, the memory
    /// the placement grants one worker). Both paths' requirements are named; a requirement the
    /// planner could not size (another authored limit bound first) is reported as unknown.
    #[error(
        "no static solve path fits the resident memory ceiling of {maximum} bytes: the proving factorization requires {} resident bytes; the elastic rigidity multigrid requires {multigrid_resident_bytes} resident bytes and {} scratch bytes",
        display_bytes(.factorization_resident_bytes),
        display_bytes(.multigrid_scratch_bytes)
    )]
    StaticSolveCeiling {
        factorization_resident_bytes: Option<u64>,
        multigrid_resident_bytes: u64,
        multigrid_scratch_bytes: Option<u64>,
        maximum: u64,
    },
    /// Neither thermal solve path fits the run's resident budget, min(authored, the memory the
    /// run is admitted to use). The proving factor and the conduction multigrid requirements are
    /// both named; the multigrid one is absent when the model is outside that path.
    #[error(
        "no thermal solve path fits the resident budget of {maximum} bytes: the proving factorization requires {factorization_resident_bytes} bytes and the conduction multigrid requires {multigrid_resident_bytes} bytes"
    )]
    ThermalSolveCeiling {
        factorization_resident_bytes: u64,
        multigrid_resident_bytes: u64,
        maximum: u64,
    },
    /// Neither modal solve path fits the effective resident ceiling, min(authored, the memory
    /// the run is admitted to use). Both paths' plan requirements are named.
    #[error(
        "no modal solve path fits the resident memory ceiling of {maximum} bytes: the shift-invert factorization requires {factorization_resident_bytes} bytes and the preconditioned LOBPCG path requires {lobpcg_resident_bytes} bytes"
    )]
    ModalSolveCeiling {
        factorization_resident_bytes: u64,
        lobpcg_resident_bytes: u64,
        maximum: u64,
    },
    /// The modes that share the eigenvalue of the last requested mode need more vectors than
    /// the modal subspace the plan admitted for the request. The solver stops at the admitted
    /// size and publishes no result. Requesting more modes admits a larger subspace.
    #[error(
        "the eigenvalue cluster at requested mode {requested_mode_count} needs more than the admitted modal subspace of {admitted_subspace_dimension} vectors; request more modes to admit a larger subspace"
    )]
    ModalSubspaceCapacity {
        requested_mode_count: u64,
        admitted_subspace_dimension: u64,
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

/// The accuracy-ceiling resource name for resident memory.
pub const RESIDENT_MEMORY_RESOURCE: &str = "resident memory bytes";

fn display_bytes(bytes: &Option<u64>) -> String {
    bytes.map_or_else(|| "an unknown number of".into(), |bytes| bytes.to_string())
}

impl ResourcePlanningError {
    /// Preparation can reject a resource envelope before a complete plan exists.
    /// Keep that typed rejection and its original counts across the planning boundary.
    pub(crate) fn from_preparation_error(error: crate::FeaRunError) -> Self {
        match error {
            crate::FeaRunError::Resource(error) => error,
            crate::FeaRunError::Usage(cause) => Self::Accounting(cause),
            crate::FeaRunError::Cancelled => Self::Cancelled,
            other => Self::Invalid(other.to_string()),
        }
    }

    pub(crate) fn from_projection_error(error: crate::topology::AnalysisProjectionError) -> Self {
        match error {
            crate::topology::AnalysisProjectionError::Accounting(cause) => Self::Accounting(cause),
            crate::topology::AnalysisProjectionError::Cancelled => Self::Cancelled,
            other => Self::Invalid(other.to_string()),
        }
    }
}
