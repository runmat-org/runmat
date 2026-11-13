//! Canonical dummyvar identity and binding.
use super::{common::*, contract::*, documentation, inference};
use crate::*;

fn dummyvar_infer(
    request: &runmat_types::CallRequest,
    entry: &BuiltinCatalogEntry,
) -> runmat_types::CallInference {
    inference::dummyvar(request, entry)
}
pub const DUMMYVAR_CATALOG_ENTRY: BuiltinCatalogEntry = BuiltinCatalogEntry {
    audience: crate::BuiltinCatalogAudience::Public,
    provenance: BuiltinCatalogProvenance::new(file!(), module_path!()),
    identity: BuiltinCatalogIdentity { name: "dummyvar" },
    category: "stats/summary",
    documentation: documentation::DUMMYVAR,
    descriptor: &DUMMYVAR_DESCRIPTOR,
    contract: BuiltinContractDeclaration {
        maturity: BuiltinContractMaturity::Incomplete,
        inference_rule: BuiltinInferenceRule::Identity(IdentityInferenceRule::new(dummyvar_infer)),
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
    extensions: &DUMMYVAR_EXTENSIONS,
    integer_capabilities: &DUMMYVAR_INTEGER_CAPABILITIES,
    integer_audit: None,
    suppress_auto_output: false,
};
