//! Canonical mvnrnd contract; execution is bound by Runtime identity only.
mod documentation;
mod inference;
mod contract;
use crate::*;
pub use contract::{
    ERROR_INTERNAL as MVNRND_ERROR_INTERNAL,
    ERROR_INVALID_ARGUMENT as MVNRND_ERROR_INVALID_ARGUMENT,
    INTEGER_MU_EXTENSION as MVNRND_INTEGER_MU_EXTENSION,
    INTEGER_N_EXTENSION as MVNRND_INTEGER_N_EXTENSION,
    INTEGER_SIGMA_EXTENSION as MVNRND_INTEGER_SIGMA_EXTENSION,
    LOGICAL_INPUT_EXTENSION as MVNRND_LOGICAL_INPUT_EXTENSION,
    RESIDENT_N_EXTENSION as MVNRND_RESIDENT_N_EXTENSION,
};
pub const MVNRND_CATALOG_ENTRY: BuiltinCatalogEntry = BuiltinCatalogEntry {
    audience: BuiltinCatalogAudience::Public,
    provenance: BuiltinCatalogProvenance::new(file!(), module_path!()),
    identity: BuiltinCatalogIdentity { name: "mvnrnd" },
    category: "stats/random",
    documentation: documentation::DOCUMENTATION,
    descriptor: &contract::MVNRND_DESCRIPTOR,
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
pub(super) const ENTRIES: &[&BuiltinCatalogEntry] = &[&MVNRND_CATALOG_ENTRY];
