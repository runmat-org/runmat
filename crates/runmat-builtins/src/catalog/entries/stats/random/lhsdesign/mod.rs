//! Canonical lhsdesign contract; execution is bound by Runtime identity only.
mod documentation;
mod inference;
mod contract;
use crate::*;
pub use contract::{
    ERROR_INTERNAL as LHSDESIGN_ERROR_INTERNAL,
    ERROR_INVALID_ARGUMENT as LHSDESIGN_ERROR_INVALID_ARGUMENT,
    INTEGER_DIMENSION_EXTENSION as LHSDESIGN_INTEGER_DIMENSION_EXTENSION,
    INTEGER_ITERATIONS_EXTENSION as LHSDESIGN_INTEGER_ITERATIONS_EXTENSION,
    INTEGER_SMOOTH_EXTENSION as LHSDESIGN_INTEGER_SMOOTH_EXTENSION,
    RESIDENT_INPUT_EXTENSION as LHSDESIGN_RESIDENT_INPUT_EXTENSION,
};
pub const LHSDESIGN_CATALOG_ENTRY: BuiltinCatalogEntry = BuiltinCatalogEntry {
    audience: BuiltinCatalogAudience::Public,
    provenance: BuiltinCatalogProvenance::new(file!(), module_path!()),
    identity: BuiltinCatalogIdentity { name: "lhsdesign" },
    category: "stats/random",
    documentation: documentation::DOCUMENTATION,
    descriptor: &contract::LHSDESIGN_DESCRIPTOR,
    contract: BuiltinContractDeclaration {
        maturity: BuiltinContractMaturity::Complete,
        inference_rule: BuiltinInferenceRule::Identity(IdentityInferenceRule::new(
            inference::infer,
        )),
        compatibility: BuiltinCompatibility::Matlab,
        async_behavior: BuiltinAsyncBehavior::MaySuspend,
        purity: BuiltinPurity::Impure,
        semantic_kind: BuiltinSemanticKind::General,
        workspace_effect: None,
        environment_effect: None,
        effects: &[
            runmat_types::EffectKind::Randomness,
            runmat_types::EffectKind::MaySuspend,
            runmat_types::EffectKind::MayThrow,
        ],
        capabilities: &[],
    },
    placement: BuiltinPlacementContract {
        preparation: BuiltinOperandPreparation::RuntimeOwned,
        portability: BuiltinPortability::NativeAndWasm,
        accelerator: BuiltinAcceleratorPolicy::Forbidden,
        residency: BuiltinResidencyPolicy::GatherToHost,
        fusion: BuiltinFusionPolicy::Never,
        distributed: BuiltinDistributedPolicy::MaterializeArguments,
    },
    link: BuiltinLinkContract {
        reachability: BuiltinReachability::Always,
        policy: BuiltinLinkPolicy::PortableRuntime,
        execution_stack: runmat_types::ExecutionStackRequirement::Any,
        artifact_dependencies: &[],
    },
    bindings: &REQUIRED_DEFAULT_BINDING,
    extensions: &contract::LHSDESIGN_EXTENSIONS,
    integer_capabilities: &contract::LHSDESIGN_INTEGER_CAPABILITIES,
    integer_audit: None,
    suppress_auto_output: false,
};
pub(super) const ENTRIES: &[&BuiltinCatalogEntry] = &[&LHSDESIGN_CATALOG_ENTRY];
