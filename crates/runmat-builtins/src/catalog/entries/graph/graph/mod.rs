use crate::*;
use runmat_types::{CallInference, CallRequest, EffectKind, ExecutionStackRequirement};

mod contract;
mod documentation;
mod inference;
pub use super::shared::{
    DIGRAPH_CLASS, DIGRAPH_INTEGER_ADJACENCY_EXTENSION, DIGRAPH_INTEGER_WEIGHTS_EXTENSION,
    GRAPH_CLASS, GRAPH_ERROR_INTERNAL, GRAPH_ERROR_INVALID_ARGUMENT,
    GRAPH_INTEGER_ADJACENCY_EXTENSION, GRAPH_INTEGER_WEIGHTS_EXTENSION,
};
pub use contract::GRAPH_DESCRIPTOR;
pub(super) const ENTRIES: &[&BuiltinCatalogEntry] = &[&GRAPH_CATALOG_ENTRY];

pub const GRAPH_CATALOG_ENTRY: BuiltinCatalogEntry = BuiltinCatalogEntry {
    audience: crate::BuiltinCatalogAudience::Public,
    provenance: BuiltinCatalogProvenance::new(file!(), module_path!()),
    identity: BuiltinCatalogIdentity { name: "graph" },
    category: "graph",
    documentation: documentation::DOCUMENTATION,
    descriptor: &GRAPH_DESCRIPTOR,
    contract: BuiltinContractDeclaration {
        maturity: BuiltinContractMaturity::Incomplete,
        inference_rule: BuiltinInferenceRule::Identity(IdentityInferenceRule::new(
            inference::infer as fn(&CallRequest, &BuiltinCatalogEntry) -> CallInference,
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
        distributed: BuiltinDistributedPolicy::Unsupported,
    },
    link: BuiltinLinkContract {
        reachability: BuiltinReachability::Always,
        policy: BuiltinLinkPolicy::PortableRuntime,
        execution_stack: ExecutionStackRequirement::Any,
        artifact_dependencies: &[],
    },
    bindings: &REQUIRED_DEFAULT_BINDING,
    extensions: &super::shared::GRAPH_INTEGER_EXTENSIONS,
    integer_capabilities: &super::shared::GRAPH_CONSTRUCTOR_INTEGER_CAPABILITIES,
    integer_audit: None,
    suppress_auto_output: false,
};
