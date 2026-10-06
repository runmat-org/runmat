use std::sync::Arc;

use runmat_hir::FunctionId;
use runmat_runtime::user_functions::SourceFunctionInfo;
use runmat_value::FunctionDefinitionLease;

use super::{FunctionRegistry, RuntimeBoundExecutableUnit};

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct RetainedCallableStageLimits {
    pub max_units: usize,
    pub max_functions: usize,
    pub max_instructions: usize,
}

/// A fully validated retained-definition publication. The temporary leases
/// keep every staged body alive until the caller decodes artifact values and
/// those values take ownership of their exact definition graphs.
#[derive(Debug)]
pub struct PreparedRetainedCallableUnits {
    registry: FunctionRegistry,
    exact_origin_facts: Arc<Vec<SourceFunctionInfo>>,
    staging_leases: Vec<FunctionDefinitionLease>,
    source_owners: Vec<RuntimeBoundExecutableUnit>,
}

impl PreparedRetainedCallableUnits {
    pub(super) fn new(
        registry: FunctionRegistry,
        exact_origin_facts: Arc<Vec<SourceFunctionInfo>>,
        staging_leases: Vec<FunctionDefinitionLease>,
        source_owners: Vec<RuntimeBoundExecutableUnit>,
    ) -> Self {
        Self {
            registry,
            exact_origin_facts,
            staging_leases,
            source_owners,
        }
    }

    pub fn registry(&self) -> &FunctionRegistry {
        &self.registry
    }

    /// Exact immutable-origin facts for synchronous value decoding. These are
    /// not a normal named/current-source catalog and must never be published
    /// as one while user code or callbacks can run.
    pub fn exact_origin_facts(&self) -> &Arc<Vec<SourceFunctionInfo>> {
        &self.exact_origin_facts
    }

    pub fn into_parts(
        self,
    ) -> (
        FunctionRegistry,
        Arc<Vec<SourceFunctionInfo>>,
        Vec<FunctionDefinitionLease>,
        Vec<RuntimeBoundExecutableUnit>,
    ) {
        (
            self.registry,
            self.exact_origin_facts,
            self.staging_leases,
            self.source_owners,
        )
    }
}

#[derive(Debug, thiserror::Error)]
pub enum RetainedCallableStageError {
    #[error("{field} exceeds its staging limit of {limit}")]
    Limit { field: &'static str, limit: usize },
    #[error("callable artifact function {function:?} has no authenticated binding origin")]
    MissingOrigin { function: FunctionId },
    #[error("callable artifact repeats one origin for distinct definitions")]
    DuplicateOrigin,
    #[error("callable artifact repeats one origin from conflicting executable units")]
    ConflictingOrigin,
    #[error("callable artifact definition cannot be fingerprinted: {0}")]
    InvalidDefinitionFingerprint(#[from] serde_json::Error),
    #[error("callable artifact contains an incomplete static definition graph")]
    IncompleteDefinitionGraph,
    #[error("callable artifact carries a source issued by a different runtime owner")]
    ForeignSourceOwner,
    #[error("session function identity space is exhausted")]
    FunctionIdentityExhausted,
    #[error("callable artifact function has no authenticated runtime source")]
    MissingRuntimeSource,
    #[error("callable artifact function has no authenticated portable source")]
    MissingPortableSource,
}
