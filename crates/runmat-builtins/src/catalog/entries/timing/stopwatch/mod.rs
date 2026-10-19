mod contract;
mod documentation;
mod inference;

#[cfg(test)]
mod tests;

pub use contract::{
    TIC_ERROR_CONTEXT_UNAVAILABLE, TIC_ERROR_TOKEN_EXHAUSTED, TOC_ERROR_INVALID_HANDLE,
    TOC_ERROR_NO_MATCHING_TIC, TOC_ERROR_TOO_MANY_INPUTS, TOC_ERROR_TOO_MANY_OUTPUTS,
};

use crate::*;
use runmat_types::{EffectKind, ExecutionStackRequirement};

pub(crate) const ENTRIES: &[&BuiltinCatalogEntry] = &[&TIC_CATALOG_ENTRY, &TOC_CATALOG_ENTRY];

const PLACEMENT: BuiltinPlacementContract = BuiltinPlacementContract {
    preparation: BuiltinOperandPreparation::RuntimeOwned,
    portability: BuiltinPortability::NativeAndWasm,
    accelerator: BuiltinAcceleratorPolicy::Forbidden,
    residency: BuiltinResidencyPolicy::Host,
    fusion: BuiltinFusionPolicy::Boundary,
    distributed: BuiltinDistributedPolicy::Unsupported,
};
const LINK: BuiltinLinkContract = BuiltinLinkContract {
    reachability: BuiltinReachability::Always,
    policy: BuiltinLinkPolicy::PortableRuntime,
    execution_stack: ExecutionStackRequirement::Any,
    artifact_dependencies: &[],
};

pub const TIC_CATALOG_ENTRY: BuiltinCatalogEntry = BuiltinCatalogEntry {
    audience: crate::BuiltinCatalogAudience::Public,
    provenance: BuiltinCatalogProvenance::new(file!(), module_path!()),
    identity: BuiltinCatalogIdentity { name: "tic" },
    category: "timing",
    documentation: documentation::TIC,
    descriptor: &contract::TIC_DESCRIPTOR,
    contract: BuiltinContractDeclaration {
        maturity: BuiltinContractMaturity::Incomplete,
        inference_rule: BuiltinInferenceRule::Identity(IdentityInferenceRule::new(inference::tic)),
        compatibility: BuiltinCompatibility::Matlab,
        async_behavior: BuiltinAsyncBehavior::NeverSuspends,
        purity: BuiltinPurity::Impure,
        semantic_kind: BuiltinSemanticKind::General,
        workspace_effect: None,
        environment_effect: None,
        effects: &[
            EffectKind::EnvironmentWrite,
            EffectKind::Clock,
            EffectKind::MayThrow,
        ],
        capabilities: &[],
    },
    placement: PLACEMENT,
    link: LINK,
    bindings: &REQUIRED_DEFAULT_BINDING,
    extensions: &[],
    integer_capabilities: &[],
    integer_audit: None,
    suppress_auto_output: true,
};

pub const TOC_CATALOG_ENTRY: BuiltinCatalogEntry = BuiltinCatalogEntry {
    audience: crate::BuiltinCatalogAudience::Public,
    provenance: BuiltinCatalogProvenance::new(file!(), module_path!()),
    identity: BuiltinCatalogIdentity { name: "toc" },
    category: "timing",
    documentation: documentation::TOC,
    descriptor: &contract::TOC_DESCRIPTOR,
    contract: BuiltinContractDeclaration {
        maturity: BuiltinContractMaturity::Incomplete,
        inference_rule: BuiltinInferenceRule::Identity(IdentityInferenceRule::new(inference::toc)),
        compatibility: BuiltinCompatibility::Matlab,
        async_behavior: BuiltinAsyncBehavior::NeverSuspends,
        purity: BuiltinPurity::Impure,
        semantic_kind: BuiltinSemanticKind::General,
        workspace_effect: None,
        environment_effect: None,
        effects: &[
            EffectKind::EnvironmentRead,
            EffectKind::UserInterface,
            EffectKind::Clock,
            EffectKind::HostCallback,
            EffectKind::MayThrow,
        ],
        capabilities: &[runmat_types::CapabilityRequirement::UserInterface],
    },
    placement: PLACEMENT,
    link: LINK,
    bindings: &REQUIRED_DEFAULT_BINDING,
    extensions: &[],
    integer_capabilities: &contract::TOC_INTEGER_CAPABILITIES,
    integer_audit: None,
    suppress_auto_output: true,
};
