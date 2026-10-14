use crate::*;
use runmat_types::{CallInference, CallRequest, EffectKind, ExecutionStackRequirement};

mod contract;
mod documentation;
#[cfg(test)]
mod tests;

pub use contract::{
    BI2DE_DESCRIPTOR, BI2DE_ERROR_BASE, BI2DE_ERROR_INPUT, BI2DE_ERROR_ORDER, BI2DE_ERROR_PROVIDER,
    BI2DE_ERROR_RANGE, BI2DE_EXTENSIONS, BI2DE_GPU_INPUT_EXTENSION, BI2DE_INTEGER_BASE_EXTENSION,
    BI2DE_INTEGER_CAPABILITIES,
};

pub const BI2DE_CATALOG_ENTRY: BuiltinCatalogEntry = BuiltinCatalogEntry {
    audience: crate::BuiltinCatalogAudience::Public,
    provenance: BuiltinCatalogProvenance::new(file!(), module_path!()),
    identity: BuiltinCatalogIdentity { name: "bi2de" },
    category: "comms/conversion",
    documentation: documentation::DOCUMENTATION,
    descriptor: &BI2DE_DESCRIPTOR,
    contract: BuiltinContractDeclaration {
        maturity: BuiltinContractMaturity::Incomplete,
        inference_rule: BuiltinInferenceRule::Identity(IdentityInferenceRule::new(
            super::inference::infer_bi2de
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
        accelerator: BuiltinAcceleratorPolicy::Forbidden,
        residency: BuiltinResidencyPolicy::GatherToHost,
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
    extensions: &BI2DE_EXTENSIONS,
    integer_capabilities: &BI2DE_INTEGER_CAPABILITIES,
    integer_audit: None,
    suppress_auto_output: false,
};
