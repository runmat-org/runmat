use crate::*;
use runmat_types::{CallInference, CallRequest, EffectKind, ExecutionStackRequirement};

mod contract;
mod documentation;
#[cfg(test)]
mod tests;

pub use contract::{
    DE2BI_DESCRIPTOR, DE2BI_ERROR_BASE, DE2BI_ERROR_INPUT, DE2BI_ERROR_ORDER, DE2BI_ERROR_PROVIDER,
    DE2BI_ERROR_RESOURCE, DE2BI_ERROR_WIDTH, DE2BI_EXTENSIONS, DE2BI_GPU_INPUT_EXTENSION,
    DE2BI_INTEGER_CAPABILITIES, DE2BI_INTEGER_CONTROL_EXTENSION, DE2BI_LOGICAL_INPUT_EXTENSION,
    DE2BI_ND_INPUT_EXTENSION, DE2BI_WIDE_INTEGER_EXTENSION,
};

pub const DE2BI_CATALOG_ENTRY: BuiltinCatalogEntry = BuiltinCatalogEntry {
    audience: crate::BuiltinCatalogAudience::Public,
    provenance: BuiltinCatalogProvenance::new(file!(), module_path!()),
    identity: BuiltinCatalogIdentity { name: "de2bi" },
    category: "comms/conversion",
    documentation: documentation::DOCUMENTATION,
    descriptor: &DE2BI_DESCRIPTOR,
    contract: BuiltinContractDeclaration {
        maturity: BuiltinContractMaturity::Incomplete,
        inference_rule: BuiltinInferenceRule::Identity(IdentityInferenceRule::new(
            super::inference::infer_de2bi
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
    extensions: &DE2BI_EXTENSIONS,
    integer_capabilities: &DE2BI_INTEGER_CAPABILITIES,
    integer_audit: None,
    suppress_auto_output: false,
};
