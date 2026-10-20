use crate::*;

mod contract;
mod documentation;
mod inference;
pub use contract::*;

const DOCUMENT_PLACEMENT: BuiltinPlacementContract = BuiltinPlacementContract {
    preparation: BuiltinOperandPreparation::RuntimeOwned,
    portability: BuiltinPortability::NativeAndWasm,
    accelerator: BuiltinAcceleratorPolicy::Forbidden,
    residency: BuiltinResidencyPolicy::Host,
    fusion: BuiltinFusionPolicy::Boundary,
    distributed: BuiltinDistributedPolicy::Unsupported,
};
const DOCUMENT_LINK: BuiltinLinkContract = BuiltinLinkContract {
    reachability: BuiltinReachability::Always,
    policy: BuiltinLinkPolicy::PortableRuntime,
    execution_stack: runmat_types::ExecutionStackRequirement::Any,
    artifact_dependencies: &[],
};
const fn contract(
    infer: fn(&runmat_types::CallRequest, &BuiltinCatalogEntry) -> runmat_types::CallInference,
) -> BuiltinContractDeclaration {
    BuiltinContractDeclaration {
        maturity: BuiltinContractMaturity::Incomplete,
        inference_rule: BuiltinInferenceRule::Identity(IdentityInferenceRule::new(infer)),
        compatibility: BuiltinCompatibility::Matlab,
        async_behavior: BuiltinAsyncBehavior::MaySuspend,
        purity: BuiltinPurity::Pure,
        semantic_kind: BuiltinSemanticKind::General,
        workspace_effect: None,
        environment_effect: None,
        effects: &[
            runmat_types::EffectKind::MaySuspend,
            runmat_types::EffectKind::MayThrow,
        ],
        capabilities: &[],
    }
}

pub const TOKEN_DETAILS_CATALOG_ENTRY: BuiltinCatalogEntry = BuiltinCatalogEntry {
    audience: BuiltinCatalogAudience::Public,
    provenance: BuiltinCatalogProvenance::new(file!(), module_path!()),
    identity: BuiltinCatalogIdentity {
        name: "tokenDetails",
    },
    category: "strings/text_analytics",
    documentation: documentation::TOKEN_DETAILS_DOCUMENTATION,
    descriptor: &TOKEN_DETAILS_DESCRIPTOR,
    contract: contract(inference::token_details),
    placement: DOCUMENT_PLACEMENT,
    link: DOCUMENT_LINK,
    bindings: &REQUIRED_DEFAULT_BINDING,
    extensions: &[],
    integer_capabilities: &[],
    integer_audit: Some(&TOKEN_DETAILS_INTEGER_AUDIT),
    suppress_auto_output: false,
};
pub const ADD_TYPE_DETAILS_CATALOG_ENTRY: BuiltinCatalogEntry = BuiltinCatalogEntry {
    audience: BuiltinCatalogAudience::Public,
    provenance: BuiltinCatalogProvenance::new(file!(), module_path!()),
    identity: BuiltinCatalogIdentity {
        name: "addTypeDetails",
    },
    category: "strings/text_analytics",
    documentation: documentation::ADD_TYPE_DETAILS_DOCUMENTATION,
    descriptor: &ADD_TYPE_DETAILS_DESCRIPTOR,
    contract: contract(inference::add_type_details),
    placement: DOCUMENT_PLACEMENT,
    link: DOCUMENT_LINK,
    bindings: &REQUIRED_DEFAULT_BINDING,
    extensions: &[],
    integer_capabilities: &[],
    integer_audit: Some(&ADD_TYPE_DETAILS_INTEGER_AUDIT),
    suppress_auto_output: false,
};
pub const ADD_SENTENCE_DETAILS_CATALOG_ENTRY: BuiltinCatalogEntry = BuiltinCatalogEntry {
    audience: BuiltinCatalogAudience::Public,
    provenance: BuiltinCatalogProvenance::new(file!(), module_path!()),
    identity: BuiltinCatalogIdentity {
        name: "addSentenceDetails",
    },
    category: "strings/text_analytics",
    documentation: documentation::ADD_SENTENCE_DETAILS_DOCUMENTATION,
    descriptor: &ADD_SENTENCE_DETAILS_DESCRIPTOR,
    contract: contract(inference::add_sentence_details),
    placement: DOCUMENT_PLACEMENT,
    link: DOCUMENT_LINK,
    bindings: &REQUIRED_DEFAULT_BINDING,
    extensions: &[],
    integer_capabilities: &[],
    integer_audit: Some(&ADD_SENTENCE_DETAILS_INTEGER_AUDIT),
    suppress_auto_output: false,
};
pub(super) const ENTRIES: &[&BuiltinCatalogEntry] = &[
    &TOKEN_DETAILS_CATALOG_ENTRY,
    &ADD_TYPE_DETAILS_CATALOG_ENTRY,
    &ADD_SENTENCE_DETAILS_CATALOG_ENTRY,
];
