use crate::*;
mod contract;
mod inference;
mod documentation;
pub use contract::{
    DESCRIPTOR as MOVMEAN_DESCRIPTOR, ERROR_TOO_MANY_OUTPUTS as MOVMEAN_ERROR_TOO_MANY_OUTPUTS,
    EXTENSIONS as MOVMEAN_EXTENSIONS,
    GPU_SAMPLE_POINTS_EXTENSION as MOVMEAN_GPU_SAMPLE_POINTS_EXTENSION,
    INTEGER_CAPABILITIES as MOVMEAN_INTEGER_CAPABILITIES,
};
pub const MOVMEAN_CATALOG_ENTRY: BuiltinCatalogEntry = BuiltinCatalogEntry {
    audience: BuiltinCatalogAudience::Public,
    provenance: BuiltinCatalogProvenance::new(file!(), module_path!()),
    identity: BuiltinCatalogIdentity { name: "movmean" },
    category: "math/reduction",
    documentation: documentation::DOCUMENTATION,
    descriptor: &MOVMEAN_DESCRIPTOR,
    contract: BuiltinContractDeclaration {
        maturity: BuiltinContractMaturity::Incomplete,
        inference_rule: BuiltinInferenceRule::Identity(IdentityInferenceRule::new(
            inference::infer,
        )),
        compatibility: BuiltinCompatibility::Matlab,
        async_behavior: BuiltinAsyncBehavior::MaySuspend,
        purity: BuiltinPurity::Pure,
        semantic_kind: BuiltinSemanticKind::General,
        workspace_effect: None,
        environment_effect: None,
        effects: &[
            runmat_types::EffectKind::MaySuspend,
            runmat_types::EffectKind::MayThrow,
        ],
        capabilities: &[],
    },
    placement: BuiltinPlacementContract {
        preparation: BuiltinOperandPreparation::RuntimeOwned,
        portability: BuiltinPortability::NativeAndWasm,
        accelerator: BuiltinAcceleratorPolicy::Optional,
        residency: BuiltinResidencyPolicy::Dynamic,
        fusion: BuiltinFusionPolicy::Boundary,
        distributed: BuiltinDistributedPolicy::MaterializeArguments,
    },
    link: BuiltinLinkContract {
        reachability: BuiltinReachability::Always,
        policy: BuiltinLinkPolicy::PortableRuntime,
        execution_stack: runmat_types::ExecutionStackRequirement::Any,
        artifact_dependencies: &[],
    },
    bindings: &REQUIRED_DEFAULT_BINDING,
    extensions: &MOVMEAN_EXTENSIONS,
    integer_capabilities: &MOVMEAN_INTEGER_CAPABILITIES,
    integer_audit: None,
    suppress_auto_output: false,
};
pub(super) const ENTRIES: &[&BuiltinCatalogEntry] = &[&MOVMEAN_CATALOG_ENTRY];
