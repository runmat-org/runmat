use crate::*;
use runmat_types::{CapabilityRequirement, EffectKind, ExecutionStackRequirement};

mod contract;
mod documentation;
mod inference;
mod selector;
mod properties;
mod methods;
pub use properties::{GpuDeviceProperty, GPU_DEVICE_CLASS};
#[cfg(test)]
mod tests;

pub use contract::*;
pub use methods::LIFECYCLE_ERROR as GPU_DEVICE_ERROR_LIFECYCLE;
pub(super) const ENTRIES: &[&BuiltinCatalogEntry] = &[
    &GPU_DEVICE_CATALOG_ENTRY,
    &methods::NAME,
    &methods::INDEX,
    &methods::TOTAL_MEMORY,
    &methods::AVAILABLE_MEMORY,
    &methods::SUPPORTS_DOUBLE,
    &methods::AVAILABLE,
    &methods::SELECTED,
    &methods::RESET,
    &methods::WAIT,
];
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
        compatibility: BuiltinCompatibility::Matlab,
        async_behavior: BuiltinAsyncBehavior::MaySuspend,
        purity: BuiltinPurity::Impure,
        semantic_kind: BuiltinSemanticKind::General,
        workspace_effect: None,
        environment_effect: None,
        effects: &[
            EffectKind::EnvironmentRead,
            EffectKind::EnvironmentWrite,
            EffectKind::MaySuspend,
            EffectKind::MayThrow,
        ],
        capabilities: &[CapabilityRequirement::Accelerator],
    },
    placement: BuiltinPlacementContract {
        preparation: BuiltinOperandPreparation::RuntimeOwned,
        portability: BuiltinPortability::NativeAndWasm,
        // Provider capability is required by the call contract above. Device
        // inspection and lifecycle control execute on the host, not a kernel.
        accelerator: BuiltinAcceleratorPolicy::Forbidden,
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
