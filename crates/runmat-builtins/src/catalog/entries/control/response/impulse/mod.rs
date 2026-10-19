use crate::*;

pub(super) const ENTRIES: &[&BuiltinCatalogEntry] = &[&IMPULSE_CATALOG_ENTRY];
use runmat_types::{
    CallInference, CallRequest, CapabilityRequirement, EffectKind, ExecutionStackRequirement,
};

mod contract;
mod documentation;
#[cfg(test)]
mod tests;

pub use contract::{
    IMPULSE_DESCRIPTOR, IMPULSE_ERROR_DISCRETE_LIMIT, IMPULSE_ERROR_INTERNAL,
    IMPULSE_ERROR_INVALID_ARGUMENT, IMPULSE_ERROR_INVALID_MODEL, IMPULSE_ERROR_INVALID_TIME,
    IMPULSE_ERROR_PLOT_FAILED, IMPULSE_ERROR_UNSUPPORTED_MODEL, IMPULSE_EXTENSIONS,
    IMPULSE_INTEGER_CAPABILITIES, IMPULSE_INTEGER_EXTENSION, IMPULSE_LOGICAL_EXTENSION,
    IMPULSE_RESIDENT_EXTENSION, IMPULSE_SINGLE_EXTENSION,
};

pub const IMPULSE_CATALOG_ENTRY: BuiltinCatalogEntry = BuiltinCatalogEntry {
    audience: crate::BuiltinCatalogAudience::Public,
    provenance: BuiltinCatalogProvenance::new(file!(), module_path!()),
    identity: BuiltinCatalogIdentity { name: "impulse" },
    category: "control/response",
    documentation: documentation::DOCUMENTATION,
    descriptor: &IMPULSE_DESCRIPTOR,
    contract: BuiltinContractDeclaration {
        maturity: BuiltinContractMaturity::Incomplete,
        inference_rule: BuiltinInferenceRule::Identity(IdentityInferenceRule::new(
            super::inference::infer_impulse
                as fn(&CallRequest, &BuiltinCatalogEntry) -> CallInference,
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
    extensions: &IMPULSE_EXTENSIONS,
    integer_capabilities: &IMPULSE_INTEGER_CAPABILITIES,
    integer_audit: None,
    suppress_auto_output: true,
};
