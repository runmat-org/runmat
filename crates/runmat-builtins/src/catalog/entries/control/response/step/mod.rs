use crate::*;

pub(super) const ENTRIES: &[&BuiltinCatalogEntry] = &[&STEP_CATALOG_ENTRY];
use runmat_types::{
    CallInference, CallRequest, CapabilityRequirement, EffectKind, ExecutionStackRequirement,
};

mod contract;
mod documentation;
#[cfg(test)]
mod tests;

pub use contract::{
    STEP_DESCRIPTOR, STEP_ERROR_DISCRETE_LIMIT, STEP_ERROR_INTERNAL, STEP_ERROR_INVALID_ARGUMENT,
    STEP_ERROR_INVALID_MODEL, STEP_ERROR_INVALID_TIME, STEP_ERROR_PLOT_FAILED,
    STEP_ERROR_UNSUPPORTED_MODEL, STEP_EXTENSIONS, STEP_INTEGER_CAPABILITIES,
    STEP_INTEGER_TIME_EXTENSION,
};

pub const STEP_CATALOG_ENTRY: BuiltinCatalogEntry = BuiltinCatalogEntry {
    audience: crate::BuiltinCatalogAudience::Public,
    provenance: BuiltinCatalogProvenance::new(file!(), module_path!()),
    identity: BuiltinCatalogIdentity { name: "step" },
    category: "control/response",
    documentation: documentation::DOCUMENTATION,
    descriptor: &STEP_DESCRIPTOR,
    contract: BuiltinContractDeclaration {
        maturity: BuiltinContractMaturity::Incomplete,
        inference_rule: BuiltinInferenceRule::Identity(IdentityInferenceRule::new(
            super::inference::infer_step as fn(&CallRequest, &BuiltinCatalogEntry) -> CallInference,
        )),
        compatibility: BuiltinCompatibility::Matlab,
        async_behavior: BuiltinAsyncBehavior::MaySuspend,
        purity: BuiltinPurity::Impure,
        semantic_kind: BuiltinSemanticKind::General,
        workspace_effect: None,
        environment_effect: None,
        effects: &[
            EffectKind::UserInterface,
            EffectKind::MaySuspend,
            EffectKind::MayThrow,
        ],
        capabilities: &[CapabilityRequirement::UserInterface],
    },
    placement: BuiltinPlacementContract {
        preparation: BuiltinOperandPreparation::RuntimeOwned,
        portability: BuiltinPortability::NativeAndWasm,
        accelerator: BuiltinAcceleratorPolicy::Forbidden,
        residency: BuiltinResidencyPolicy::GatherToHost,
        fusion: BuiltinFusionPolicy::Boundary,
        distributed: BuiltinDistributedPolicy::Unsupported,
    },
    link: BuiltinLinkContract {
        reachability: BuiltinReachability::Always,
        policy: BuiltinLinkPolicy::PortableRuntime,
        execution_stack: ExecutionStackRequirement::Any,
        artifact_dependencies: &[],
    },
    bindings: &REQUIRED_DEFAULT_BINDING,
    extensions: &contract::STEP_AND_MODEL_EXTENSIONS,
    integer_capabilities: &STEP_INTEGER_CAPABILITIES,
    integer_audit: None,
    suppress_auto_output: true,
};
