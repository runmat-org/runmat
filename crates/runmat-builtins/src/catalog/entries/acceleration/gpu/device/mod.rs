use crate::*;
use runmat_types::{EffectKind, ExecutionStackRequirement};

mod contract;
mod documentation;
mod inference;
mod selector;
#[cfg(test)]
mod tests;

pub use contract::*;
pub(super) const ENTRIES: &[&BuiltinCatalogEntry] = &[&GPU_DEVICE_CATALOG_ENTRY];
pub use selector::{
    gpu_device_index_from_float, gpu_device_index_from_integer, GpuDeviceIndexError,
    GPU_DEVICE_ACTIVE_INDEX,
};

pub const GPU_DEVICE_CATALOG_ENTRY: BuiltinCatalogEntry = BuiltinCatalogEntry {
    audience: crate::BuiltinCatalogAudience::Public,
    provenance: BuiltinCatalogProvenance::new(file!(), module_path!()),
    identity: BuiltinCatalogIdentity { name: "gpuDevice" },
    category: "acceleration/gpu",
    documentation: documentation::DOCUMENTATION,
    descriptor: &GPU_DEVICE_DESCRIPTOR,
    contract: BuiltinContractDeclaration {
        maturity: BuiltinContractMaturity::Incomplete,
        inference_rule: BuiltinInferenceRule::Identity(IdentityInferenceRule::new(
            inference::infer,
        )),
        compatibility: BuiltinCompatibility::RunMat,
        async_behavior: BuiltinAsyncBehavior::NeverSuspends,
        purity: BuiltinPurity::DeterministicReadOnly,
        semantic_kind: BuiltinSemanticKind::General,
        workspace_effect: None,
        environment_effect: None,
        effects: &[EffectKind::EnvironmentRead, EffectKind::MayThrow],
        capabilities: &[],
    },
    placement: BuiltinPlacementContract {
        preparation: BuiltinOperandPreparation::RuntimeOwned,
        portability: BuiltinPortability::NativeAndWasm,
        accelerator: BuiltinAcceleratorPolicy::Optional,
        residency: BuiltinResidencyPolicy::Host,
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
    extensions: &GPU_DEVICE_EXTENSIONS,
    integer_capabilities: &GPU_DEVICE_INTEGER_CAPABILITIES,
    integer_audit: None,
    suppress_auto_output: false,
};
