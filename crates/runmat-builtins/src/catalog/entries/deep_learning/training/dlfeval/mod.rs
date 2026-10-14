//! Callback execution with an execution-owned reverse-mode tape.

use crate::*;
use runmat_types::{CallInference, CallRequest, EffectKind, ExecutionStackRequirement};

mod contract;
mod documentation;
mod inference;
#[cfg(test)]
mod tests;

pub use contract::{DLFEVAL_DESCRIPTOR, DLFEVAL_INVALID_FUNCTION, DLFEVAL_UNSUPPORTED};

pub(super) const ENTRIES: &[&BuiltinCatalogEntry] = &[&DLFEVAL_CATALOG_ENTRY];

pub const DLFEVAL_CATALOG_ENTRY: BuiltinCatalogEntry = BuiltinCatalogEntry {
    audience: crate::BuiltinCatalogAudience::Public,
    provenance: BuiltinCatalogProvenance::new(file!(), module_path!()),
    identity: BuiltinCatalogIdentity { name: "dlfeval" },
    category: "deep_learning",
    documentation: documentation::DOCUMENTATION,
    descriptor: &DLFEVAL_DESCRIPTOR,
    contract: BuiltinContractDeclaration {
        maturity: BuiltinContractMaturity::Incomplete,
        inference_rule: BuiltinInferenceRule::Identity(IdentityInferenceRule::new(
            inference::infer as fn(&CallRequest, &BuiltinCatalogEntry) -> CallInference,
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
            EffectKind::HostCallback,
            EffectKind::MaySuspend,
            EffectKind::MayThrow,
            EffectKind::Unknown,
        ],
        capabilities: &[],
    },
    placement: BuiltinPlacementContract {
        preparation: BuiltinOperandPreparation::RuntimeOwned,
        portability: BuiltinPortability::NativeAndWasm,
        accelerator: BuiltinAcceleratorPolicy::Optional,
        residency: BuiltinResidencyPolicy::Dynamic,
        fusion: BuiltinFusionPolicy::Boundary,
        distributed: BuiltinDistributedPolicy::Unsupported,
    },
    link: BuiltinLinkContract {
        reachability: BuiltinReachability::Dynamic,
        policy: BuiltinLinkPolicy::PortableRuntime,
        execution_stack: ExecutionStackRequirement::Process,
        artifact_dependencies: &[],
    },
    bindings: &REQUIRED_DEFAULT_BINDING,
    extensions: &[],
    integer_capabilities: &contract::INTEGER_CAPABILITIES,
    integer_audit: None,
    suppress_auto_output: false,
};
