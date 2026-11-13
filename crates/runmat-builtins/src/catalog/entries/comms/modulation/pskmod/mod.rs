use crate::*;
use runmat_types::{CallInference, CallRequest, EffectKind, ExecutionStackRequirement};

mod contract;
mod documentation;
#[cfg(test)]
mod tests;

pub use contract::{
    INTEGER_CAPABILITIES as PSKMOD_INTEGER_CAPABILITIES, PSKMOD_DESCRIPTOR, PSKMOD_EXTENSIONS,
    PSKMOD_INTEGER_CUSTOM_ORDER_EXTENSION, PSKMOD_INTEGER_ORDER_EXTENSION,
    PSKMOD_INTEGER_PHASE_EXTENSION,
};

pub const PSKMOD_CATALOG_ENTRY: BuiltinCatalogEntry = BuiltinCatalogEntry {
    audience: crate::BuiltinCatalogAudience::Public,
    provenance: BuiltinCatalogProvenance::new(file!(), module_path!()),
    identity: BuiltinCatalogIdentity { name: "pskmod" },
    category: "comms/modulation",
    documentation: documentation::DOCUMENTATION,
    descriptor: &PSKMOD_DESCRIPTOR,
    contract: BuiltinContractDeclaration {
        maturity: BuiltinContractMaturity::Incomplete,
        inference_rule: BuiltinInferenceRule::Identity(IdentityInferenceRule::new(
            super::inference::infer_pskmod
                as fn(&CallRequest, &BuiltinCatalogEntry) -> CallInference,
        )),
        compatibility: BuiltinCompatibility::Matlab,
        async_behavior: BuiltinAsyncBehavior::MaySuspend,
        purity: BuiltinPurity::Pure,
        semantic_kind: BuiltinSemanticKind::General,
        workspace_effect: None,
        environment_effect: None,
        effects: &[EffectKind::MaySuspend, EffectKind::MayThrow],
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
        execution_stack: ExecutionStackRequirement::Any,
        artifact_dependencies: &[],
    },
    bindings: &REQUIRED_DEFAULT_BINDING,
    extensions: &PSKMOD_EXTENSIONS,
    integer_capabilities: &PSKMOD_INTEGER_CAPABILITIES,
    integer_audit: None,
    suppress_auto_output: false,
};
