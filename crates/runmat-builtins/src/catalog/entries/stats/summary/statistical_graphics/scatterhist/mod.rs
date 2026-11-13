//! Scatterhist owns its public form, option and output facts.
mod contract;
mod documentation;
mod inference;
mod options;
#[cfg(test)]
mod tests;

use crate::*;
use runmat_types::{CapabilityRequirement, EffectKind, ExecutionStackRequirement};

pub use contract::{
    SCATTERHIST_DESCRIPTOR, SCATTERHIST_ERROR_INTERNAL, SCATTERHIST_ERROR_INVALID_ARGUMENT,
    SCATTERHIST_EXTENSIONS, SCATTERHIST_GPU_INPUT_EXTENSION, SCATTERHIST_INTEGER_CAPABILITIES,
    SCATTERHIST_INTEGER_DATA_EXTENSION, SCATTERHIST_INTEGER_GROUP_EXTENSION,
    SCATTERHIST_INTEGER_OPTION_EXTENSION, SCATTERHIST_MAX_BINS,
};
pub use options::{ScatterhistIntegerRole, ScatterhistOptionKind};

pub const SCATTERHIST_CATALOG_ENTRY: BuiltinCatalogEntry = BuiltinCatalogEntry {
    audience: crate::BuiltinCatalogAudience::Public,
    provenance: BuiltinCatalogProvenance::new(file!(), module_path!()),
    identity: BuiltinCatalogIdentity {
        name: "scatterhist",
    },
    category: "stats/summary",
    documentation: documentation::DOCUMENTATION,
    descriptor: &SCATTERHIST_DESCRIPTOR,
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
            EffectKind::UserInterface,
            EffectKind::HostCallback,
            EffectKind::MaySuspend,
            EffectKind::MayThrow,
        ],
        capabilities: &[CapabilityRequirement::UserInterface],
    },
    placement: BuiltinPlacementContract {
        preparation: BuiltinOperandPreparation::RuntimeOwned,
        portability: BuiltinPortability::NativeAndWasm,
        accelerator: BuiltinAcceleratorPolicy::Optional,
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
    extensions: &SCATTERHIST_EXTENSIONS,
    integer_capabilities: &SCATTERHIST_INTEGER_CAPABILITIES,
    integer_audit: None,
    suppress_auto_output: true,
};
