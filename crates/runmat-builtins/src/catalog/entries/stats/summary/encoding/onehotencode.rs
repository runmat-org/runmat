//! Canonical onehotencode identity and binding.
use super::{common::*, contract::*, documentation, inference};
use crate::*;

fn encode_infer(
    request: &runmat_types::CallRequest,
    entry: &BuiltinCatalogEntry,
) -> runmat_types::CallInference {
    inference::encode(request, entry)
}
pub const ONEHOTENCODE_CATALOG_ENTRY: BuiltinCatalogEntry = BuiltinCatalogEntry {
    audience: crate::BuiltinCatalogAudience::Public,
    provenance: BuiltinCatalogProvenance::new(file!(), module_path!()),
    identity: BuiltinCatalogIdentity {
        name: "onehotencode",
    },
    category: "stats/summary",
    documentation: documentation::ONEHOTENCODE,
    descriptor: &ONEHOTENCODE_DESCRIPTOR,
    contract: BuiltinContractDeclaration {
        maturity: BuiltinContractMaturity::Incomplete,
        inference_rule: BuiltinInferenceRule::Identity(IdentityInferenceRule::new(encode_infer)),
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
    extensions: &ONEHOTENCODE_EXTENSIONS,
    integer_capabilities: &ONEHOTENCODE_INTEGER_CAPABILITIES,
    integer_audit: None,
    suppress_auto_output: false,
};
