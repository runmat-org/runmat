mod descriptor;
mod documentation;
mod inference;

pub use descriptor::MEXEXT_DESCRIPTOR;

use super::contract::{BUILD_PLACEMENT, EXTENSION_INTEGER_AUDIT, EXTENSION_LINK};
use crate::*;
use runmat_types::EffectKind;

pub const MEXEXT_CATALOG_ENTRY: BuiltinCatalogEntry = BuiltinCatalogEntry {
    audience: crate::BuiltinCatalogAudience::Public,
    provenance: BuiltinCatalogProvenance::new(file!(), module_path!()),
    identity: BuiltinCatalogIdentity { name: "mexext" },
    category: "interop/mex",
    documentation: documentation::DOCUMENTATION,
    descriptor: &MEXEXT_DESCRIPTOR,
    contract: BuiltinContractDeclaration {
        maturity: BuiltinContractMaturity::Incomplete,
        inference_rule: BuiltinInferenceRule::Identity(IdentityInferenceRule::new(
            inference::infer,
        )),
        compatibility: BuiltinCompatibility::RunMat,
        async_behavior: BuiltinAsyncBehavior::NeverSuspends,
        purity: BuiltinPurity::DeterministicReadOnly,
        semantic_kind: BuiltinSemanticKind::General,
        workspace_effect: None,
        environment_effect: None,
        effects: &[EffectKind::MayThrow],
        capabilities: &[],
    },
    placement: BUILD_PLACEMENT,
    link: EXTENSION_LINK,
    bindings: &REQUIRED_DEFAULT_BINDING,
    extensions: &[],
    integer_capabilities: &[],
    integer_audit: Some(&EXTENSION_INTEGER_AUDIT),
    suppress_auto_output: false,
};
