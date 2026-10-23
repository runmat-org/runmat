//! Canonical normrnd contract; execution is bound by Runtime identity only.
mod documentation;
mod inference;
mod contract;
use crate::*;
pub use contract::{
    INTEGER_MU_EXTENSION as NORMRND_INTEGER_MU_EXTENSION,
    INTEGER_SIGMA_EXTENSION as NORMRND_INTEGER_SIGMA_EXTENSION,
    INTEGER_SIZE_EXTENSION as NORMRND_INTEGER_SIZE_EXTENSION,
    LOGICAL_INPUT_EXTENSION as NORMRND_LOGICAL_INPUT_EXTENSION, NORMRND_ERROR_INTERNAL,
    NORMRND_ERROR_INVALID_ARGUMENT, NORMRND_ERROR_SIGMA_MUST_BE_NONNEGATIVE,
    RESIDENT_SIZE_EXTENSION as NORMRND_RESIDENT_SIZE_EXTENSION,
};
pub const NORMRND_CATALOG_ENTRY: BuiltinCatalogEntry = BuiltinCatalogEntry {
    audience: BuiltinCatalogAudience::Public,
    provenance: BuiltinCatalogProvenance::new(file!(), module_path!()),
    identity: BuiltinCatalogIdentity { name: "normrnd" },
    category: "stats/random",
    documentation: documentation::DOCUMENTATION,
    descriptor: &contract::NORMRND_DESCRIPTOR,
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
            runmat_types::EffectKind::Randomness,
            runmat_types::EffectKind::MaySuspend,
            runmat_types::EffectKind::MayThrow,
        ],
        capabilities: &[],
    },
    placement: BuiltinPlacementContract {
        preparation: BuiltinOperandPreparation::RuntimeOwned,
        portability: BuiltinPortability::NativeAndWasm,
        accelerator: BuiltinAcceleratorPolicy::Optional,
        residency: BuiltinResidencyPolicy::PreserveInputs,
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
    extensions: &contract::EXTENSIONS,
    integer_capabilities: &contract::INTEGER_CAPABILITIES,
    integer_audit: None,
    suppress_auto_output: false,
};
pub(super) const ENTRIES: &[&BuiltinCatalogEntry] = &[&NORMRND_CATALOG_ENTRY];
