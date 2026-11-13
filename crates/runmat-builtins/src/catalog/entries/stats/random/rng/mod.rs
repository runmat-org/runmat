//! Canonical rng contract; execution is bound by Runtime identity only.
mod documentation;
mod inference;
mod contract;
use crate::*;
pub use contract::{
    RNG_ERROR_CONTEXT_REQUIRED, RNG_ERROR_GENERATOR_UNSUPPORTED, RNG_ERROR_INTERNAL,
    RNG_ERROR_INVALID_ARGUMENT, RNG_ERROR_SEED_NONNEGATIVE, RNG_ERROR_STATE_TYPE_FIELD_MISSING,
    RNG_INTEGER_CAPABILITIES, RNG_TYPED_STATE_EXTENSION, RNG_WIDE_SEED_EXTENSION,
};
pub const RNG_CATALOG_ENTRY: BuiltinCatalogEntry = BuiltinCatalogEntry {
    audience: BuiltinCatalogAudience::Public,
    provenance: BuiltinCatalogProvenance::new(file!(), module_path!()),
    identity: BuiltinCatalogIdentity { name: "rng" },
    category: "stats/random",
    documentation: documentation::DOCUMENTATION,
    descriptor: &contract::RNG_DESCRIPTOR,
    contract: BuiltinContractDeclaration {
        maturity: BuiltinContractMaturity::Incomplete,
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
            runmat_types::EffectKind::EnvironmentRead,
            runmat_types::EffectKind::Randomness,
            runmat_types::EffectKind::Clock,
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
    extensions: &contract::RNG_EXTENSIONS,
    integer_capabilities: &contract::RNG_INTEGER_CAPABILITIES,
    integer_audit: None,
    suppress_auto_output: false,
};
pub(super) const ENTRIES: &[&BuiltinCatalogEntry] = &[&RNG_CATALOG_ENTRY];
