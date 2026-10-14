mod descriptor;
mod documentation;
pub(super) mod inference;

pub use descriptor::PYARGS_DESCRIPTOR;

use super::contract::{FOREIGN_LINK, HOST_PLACEMENT, INTEGER_AUDIT};
use crate::*;
use runmat_types::EffectKind;

pub const PYARGS_CATALOG_ENTRY: BuiltinCatalogEntry = BuiltinCatalogEntry {
    audience: crate::BuiltinCatalogAudience::Public,
    provenance: BuiltinCatalogProvenance::new(file!(), module_path!()),
    identity: BuiltinCatalogIdentity { name: "pyargs" },
    category: "interop/python",
    documentation: documentation::PYARGS,
    descriptor: &PYARGS_DESCRIPTOR,
    contract: BuiltinContractDeclaration {
        maturity: BuiltinContractMaturity::Incomplete,
        inference_rule: BuiltinInferenceRule::Identity(IdentityInferenceRule::new(
            inference::infer,
        )),
        compatibility: BuiltinCompatibility::Matlab,
        async_behavior: BuiltinAsyncBehavior::NeverSuspends,
        purity: BuiltinPurity::Pure,
        semantic_kind: BuiltinSemanticKind::ArrayConstructor,
        workspace_effect: None,
        environment_effect: None,
        effects: &[EffectKind::MayThrow],
        capabilities: &[],
    },
    placement: BuiltinPlacementContract {
        portability: BuiltinPortability::NativeAndWasm,
        ..HOST_PLACEMENT
    },
    link: BuiltinLinkContract {
        policy: BuiltinLinkPolicy::PortableRuntime,
        execution_stack: ExecutionStackRequirement::Any,
        ..FOREIGN_LINK
    },
    bindings: &REQUIRED_DEFAULT_BINDING,
    extensions: &[],
    integer_capabilities: &[],
    integer_audit: Some(&INTEGER_AUDIT),
    suppress_auto_output: false,
};
