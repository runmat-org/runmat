//! A value-backed sequential network and its learnables metadata.

use crate::*;
use runmat_types::{CallInference, CallRequest, EffectKind, ExecutionStackRequirement};

mod contract;
mod documentation;
mod inference;
mod option_schema;
mod parameter_table_schema;
mod property_access;
mod state_access;
#[cfg(test)]
mod tests;

pub use contract::{DLNETWORK_DESCRIPTOR, DLNETWORK_INVALID_INPUT};
pub(super) use inference::state_rows_table_fact;
pub use option_schema::DeepLearningNetworkOption;
pub use parameter_table_schema::DeepLearningParameterTableColumn;

pub(super) const ENTRIES: &[&BuiltinCatalogEntry] = &[
    &DLNETWORK_CATALOG_ENTRY,
    &property_access::GET_LEARNABLES_CATALOG_ENTRY,
    &property_access::SET_LEARNABLES_CATALOG_ENTRY,
    &state_access::GET_STATE_CATALOG_ENTRY,
    &state_access::SET_STATE_CATALOG_ENTRY,
];

pub const DLNETWORK_CATALOG_ENTRY: BuiltinCatalogEntry = BuiltinCatalogEntry {
    audience: crate::BuiltinCatalogAudience::Public,
    provenance: BuiltinCatalogProvenance::new(file!(), module_path!()),
    identity: BuiltinCatalogIdentity { name: "dlnetwork" },
    category: "deep_learning",
    documentation: documentation::DOCUMENTATION,
    descriptor: &DLNETWORK_DESCRIPTOR,
    contract: BuiltinContractDeclaration {
        maturity: BuiltinContractMaturity::Incomplete,
        inference_rule: BuiltinInferenceRule::Identity(IdentityInferenceRule::new(
            inference::infer as fn(&CallRequest, &BuiltinCatalogEntry) -> CallInference,
        )),
        compatibility: BuiltinCompatibility::Matlab,
        async_behavior: BuiltinAsyncBehavior::NeverSuspends,
        purity: BuiltinPurity::Pure,
        semantic_kind: BuiltinSemanticKind::General,
        workspace_effect: None,
        environment_effect: None,
        effects: &[EffectKind::MayThrow],
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
    extensions: &[],
    integer_capabilities: &[],
    integer_audit: None,
    suppress_auto_output: false,
};
