//! Boxplot public identity, option names, and conditional static contract.
mod contract;
mod documentation;
mod inference;
mod options;
#[cfg(test)]
mod tests;

use crate::*;
use runmat_types::{CapabilityRequirement, EffectKind, ExecutionStackRequirement};

pub use contract::{
    BOXPLOT_DESCRIPTOR, BOXPLOT_ERROR_INTERNAL, BOXPLOT_ERROR_INVALID_ARGUMENT, BOXPLOT_EXTENSIONS,
    BOXPLOT_GPU_INPUT_EXTENSION, BOXPLOT_INTEGER_CAPABILITIES, BOXPLOT_INTEGER_DATA_EXTENSION,
    BOXPLOT_INTEGER_GROUP_EXTENSION, BOXPLOT_INTEGER_OPTION_EXTENSION,
    BOXPLOT_LOGICAL_DATA_EXTENSION, BOXPLOT_LOGICAL_GROUP_EXTENSION,
    BOXPLOT_LOGICAL_OPTION_EXTENSION,
};
pub use options::BoxplotOptionKind;

pub const BOXPLOT_CATALOG_ENTRY: BuiltinCatalogEntry = BuiltinCatalogEntry {
    audience: crate::BuiltinCatalogAudience::Public,
    provenance: BuiltinCatalogProvenance::new(file!(), module_path!()),
    identity: BuiltinCatalogIdentity { name: "boxplot" },
    category: "stats/summary",
    documentation: documentation::DOCUMENTATION,
    descriptor: &BOXPLOT_DESCRIPTOR,
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
    extensions: &BOXPLOT_EXTENSIONS,
    integer_capabilities: &BOXPLOT_INTEGER_CAPABILITIES,
    integer_audit: None,
    suppress_auto_output: true,
};
