mod contract;
mod control;
mod documentation;
mod inference;

#[cfg(test)]
mod tests;

pub use contract::{
    CLOSE_DESCRIPTOR, CLOSE_ERROR_INVALID_ARGUMENT, CLOSE_ERROR_TOO_MANY_OUTPUTS, CLOSE_EXTENSIONS,
    CLOSE_INTEGER_CAPABILITIES, CLOSE_INTEGER_FIGURE_NUMBER_EXTENSION,
    CLOSE_VARIADIC_TARGETS_EXTENSION,
};
pub use control::CloseNetworkSelector;

use crate::*;
use runmat_types::{EffectKind, ExecutionStackRequirement};

pub const CLOSE_CATALOG_ENTRY: BuiltinCatalogEntry = BuiltinCatalogEntry {
    provenance: BuiltinCatalogProvenance::new(file!(), module_path!()),
    identity: BuiltinCatalogIdentity { name: "close" },
    category: "general",
    documentation: documentation::DOCUMENTATION,
    descriptor: &CLOSE_DESCRIPTOR,
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
            EffectKind::Network,
            EffectKind::HostCallback,
            EffectKind::MaySuspend,
            EffectKind::MayThrow,
            EffectKind::Unknown,
        ],
        // Graphics does not require TCP, and TCP does not require graphics.
        // The inferred form and runtime route select the relevant service.
        capabilities: &[],
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
    extensions: &CLOSE_EXTENSIONS,
    integer_capabilities: &CLOSE_INTEGER_CAPABILITIES,
    integer_audit: None,
    suppress_auto_output: true,
};
