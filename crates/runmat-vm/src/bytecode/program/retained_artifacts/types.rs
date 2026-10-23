use std::sync::Arc;

use runmat_execution::value::GraphStaticReferenceSite;
use runmat_hir::FunctionId;
use runmat_runtime::user_functions::SourceFunctionInfo;
use runmat_value::FunctionDefinitionLease;

use super::{FunctionRegistry, RuntimeBoundExecutableUnit};
use crate::bytecode::program::{
    RetainedImplicitSuperConstructorTarget, SuperConstructorDependencyError,
};
use crate::RetainedExecutableSourceOwner;

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
    linked_constructorless_classes: Vec<runmat_runtime::context::RetainedClassDeclaration>,
}

pub type RetainedCallableParts = (
    FunctionRegistry,
    Arc<Vec<SourceFunctionInfo>>,
    Vec<FunctionDefinitionLease>,
    Vec<RetainedExecutableSourceOwner>,
);

impl PreparedRetainedCallableUnits {
    pub(super) fn new(
        registry: FunctionRegistry,
        exact_origin_facts: Arc<Vec<SourceFunctionInfo>>,
        staging_leases: Vec<FunctionDefinitionLease>,
        source_owners: Vec<RuntimeBoundExecutableUnit>,
        linked_constructorless_classes: Vec<runmat_runtime::context::RetainedClassDeclaration>,
    ) -> Self {
        Self {
            registry,
            exact_origin_facts,
            staging_leases,
            source_owners,
            linked_constructorless_classes,
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

    /// Capture only the selected class declarations and already-issued source
    /// tables. The resulting owner holds no runtime execution service.
    pub fn selected_class_product_for(
        &self,
        class: &runmat_types::ClassIdentity,
    ) -> Result<crate::SelectedClassProduct, runmat_runtime::RuntimeError> {
        crate::SelectedClassProduct::from_bound_units(
            &self.source_owners.iter().collect::<Vec<_>>(),
            class,
        )
    }

    /// Resolve an explicit no-body super call against the same validated,
    /// runtime-bound units being staged. A constructorless script class has no
    /// FunctionBytecode, so its admitted RegisterClass remains here with its
    /// selected source lease instead of becoming a fabricated definition.
    pub fn retained_implicit_super_constructor_targets_for(
        &self,
        selected: &FunctionRegistry,
        caller: FunctionId,
    ) -> Result<Vec<RetainedImplicitSuperConstructorTarget>, SuperConstructorDependencyError> {
        selected.retained_implicit_super_constructor_targets_with_linked(
            caller,
            &self.linked_constructorless_classes,
        )
    }

    pub fn into_parts(self) -> Result<RetainedCallableParts, RetainedCallableStageError> {
        let source_owners = self
            .source_owners
            .iter()
            .map(RetainedExecutableSourceOwner::capture)
            .collect::<Result<Vec<_>, _>>()
            .map_err(RetainedCallableStageError::RetainedSourceOwner)?;
        Ok((
            self.registry,
            self.exact_origin_facts,
            self.staging_leases,
            source_owners,
        ))
    }
}

#[derive(Debug, thiserror::Error)]
pub enum RetainedCallableStageError {
    #[error("staged imported source owner is invalid: {0}")]
    RetainedSourceOwner(&'static str),
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
    #[error("selected fingerprint for {function:?} violates {invariant:?}")]
    SelectedFingerprintGraph {
        function: FunctionId,
        invariant: SelectedFingerprintInvariant,
    },
    #[error("callable artifact carries a source issued by a different runtime owner")]
    ForeignSourceOwner,
    #[error("session function identity space is exhausted")]
    FunctionIdentityExhausted,
    #[error("callable artifact function has no authenticated runtime source")]
    MissingRuntimeSource,
    #[error("callable artifact function has no authenticated portable source")]
    MissingPortableSource,
}

/// The first failed invariant in an exact selected-body fingerprint. These
/// stages distinguish malformed import geometry from missing local ownership
/// without turning a failed graph proof into a name-based fallback.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum SelectedFingerprintInvariant {
    DuplicateSelectedBody,
    MissingSelectedBody,
    RepeatedRequestedBody,
    MissingRootBody,
    MissingLexicalParent(FunctionId),
    MismatchedLexicalParent(FunctionId),
    UnexpectedPortableLexicalParent,
    DuplicateImportSite(GraphStaticReferenceSite),
    InvalidReferenceSites(usize),
    ImportKindMismatch(GraphStaticReferenceSite),
    UnimportedExternalReference(GraphStaticReferenceSite),
    UnobservedImportSite(GraphStaticReferenceSite),
    MissingLocalDefinition(FunctionId),
    ClosedFunctionRemap,
    ImportedSiteRemap,
    FusionStageProjection,
}
