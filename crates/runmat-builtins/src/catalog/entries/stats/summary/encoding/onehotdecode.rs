//! Canonical onehotdecode identity and binding.
use super::{common::*, contract::*, documentation, inference};
use crate::*;

fn decode_infer(
    request: &runmat_types::CallRequest,
    entry: &BuiltinCatalogEntry,
) -> runmat_types::CallInference {
    inference::decode(request, entry)
}

pub const ONEHOTDECODE_CATALOG_ENTRY: BuiltinCatalogEntry = BuiltinCatalogEntry {
    audience: crate::BuiltinCatalogAudience::Public,
    provenance: BuiltinCatalogProvenance::new(file!(), module_path!()),
    identity: BuiltinCatalogIdentity {
        name: "onehotdecode",
    },
    category: "stats/summary",
    documentation: documentation::ONEHOTDECODE,
    descriptor: &ONEHOTDECODE_DESCRIPTOR,
    contract: BuiltinContractDeclaration {
        maturity: BuiltinContractMaturity::Incomplete,
        inference_rule: BuiltinInferenceRule::Identity(IdentityInferenceRule::new(decode_infer)),
        compatibility: BuiltinCompatibility::Matlab,
        async_behavior: BuiltinAsyncBehavior::MaySuspend,
        purity: BuiltinPurity::Pure,
        semantic_kind: BuiltinSemanticKind::General,
        workspace_effect: None,
        environment_effect: None,
        effects: EFFECTS,
        capabilities: &[],
    },
    placement: PLACEMENT,
    link: LINK,
    bindings: &REQUIRED_DEFAULT_BINDING,
    extensions: &ONEHOTDECODE_EXTENSIONS,
    integer_capabilities: &ONEHOTDECODE_INTEGER_CAPABILITIES,
    integer_audit: None,
    suppress_auto_output: false,
};
