//! ONNX serialization of supported feed-forward network objects.

use crate::*;
use runmat_types::{
    CallInference, CallRequest, CapabilityRequirement, EffectKind, ExecutionStackRequirement,
};

mod contract;
mod documentation;
mod inference;
mod options;
#[cfg(test)]
mod tests;

pub use contract::{
    EXPORT_ONNX_DESCRIPTOR, EXPORT_ONNX_INTEGER_EXTENSION, EXPORT_ONNX_INVALID_INPUT,
    EXPORT_ONNX_UNSUPPORTED,
};
pub use options::{ExportOnnxOptionKey, ExportOnnxOptionValueRole};

pub(super) const ENTRIES: &[&BuiltinCatalogEntry] = &[&EXPORT_ONNX_CATALOG_ENTRY];

pub const EXPORT_ONNX_CATALOG_ENTRY: BuiltinCatalogEntry = BuiltinCatalogEntry {
    audience: crate::BuiltinCatalogAudience::Public,
    provenance: BuiltinCatalogProvenance::new(file!(), module_path!()),
    identity: BuiltinCatalogIdentity {
        name: "exportONNXNetwork",
    },
    category: "deep_learning",
    documentation: documentation::DOCUMENTATION,
    descriptor: &EXPORT_ONNX_DESCRIPTOR,
    contract: BuiltinContractDeclaration {
        maturity: BuiltinContractMaturity::Incomplete,
        inference_rule: BuiltinInferenceRule::Identity(IdentityInferenceRule::new(
            inference::infer as fn(&CallRequest, &BuiltinCatalogEntry) -> CallInference,
        )),
        compatibility: BuiltinCompatibility::Matlab,
        async_behavior: BuiltinAsyncBehavior::MaySuspend,
        purity: BuiltinPurity::Impure,
        semantic_kind: BuiltinSemanticKind::Filesystem,
        workspace_effect: None,
        environment_effect: None,
        effects: &[
            EffectKind::FilesystemWrite,
            EffectKind::MaySuspend,
            EffectKind::MayThrow,
        ],
        capabilities: &[CapabilityRequirement::Filesystem],
    },
    placement: BuiltinPlacementContract {
        preparation: BuiltinOperandPreparation::RuntimeOwned,
        portability: BuiltinPortability::NativeAndWasm,
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
    extensions: &contract::EXTENSIONS,
    integer_capabilities: &contract::INTEGER_CAPABILITIES,
    integer_audit: None,
    suppress_auto_output: true,
};
