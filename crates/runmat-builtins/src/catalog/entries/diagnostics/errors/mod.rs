mod contract;
mod documentation;
mod inference;

#[cfg(test)]
mod tests;

pub use contract::{
    ERROR_DESCRIPTOR, ERROR_ERROR_CORRECTION_UNSUPPORTED, ERROR_ERROR_EXTRA_ARGS_MEXCEPTION,
    ERROR_ERROR_EXTRA_ARGS_STRUCT, ERROR_ERROR_INVALID_IDENTIFIER, ERROR_ERROR_INVALID_INPUT,
    ERROR_ERROR_MISSING_MESSAGE, ERROR_ERROR_STRUCT_NO_FIELDS,
    ERROR_ERROR_STRUCT_STACK_UNSUPPORTED, ERROR_EXTENSIONS, ERROR_INTEGER_CAPABILITIES,
    ERROR_MEXCEPTION_EXTENSION, ERROR_STRUCT_ALIAS_EXTENSION,
    ERROR_UNQUALIFIED_IDENTIFIER_EXTENSION, RETHROW_ACTIVE_CATCH_EXTENSION, RETHROW_DESCRIPTOR,
    RETHROW_ERROR_INVALID_ARITY, RETHROW_ERROR_NO_ACTIVE_CATCH, RETHROW_EXTENSIONS,
    RETHROW_INTEGER_AUDIT, RETHROW_TEXT_EXTENSION,
};

use crate::*;
use runmat_types::{EffectKind, ExecutionStackRequirement};

pub(crate) const ENTRIES: &[&BuiltinCatalogEntry] = &[&ERROR_CATALOG_ENTRY, &RETHROW_CATALOG_ENTRY];

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

pub const ERROR_CATALOG_ENTRY: BuiltinCatalogEntry = BuiltinCatalogEntry {
    audience: crate::BuiltinCatalogAudience::Public,
    provenance: BuiltinCatalogProvenance::new(file!(), module_path!()),
    identity: BuiltinCatalogIdentity { name: "error" },
    category: "diagnostics",
    documentation: documentation::ERROR,
    descriptor: &ERROR_DESCRIPTOR,
    contract: BuiltinContractDeclaration {
        maturity: BuiltinContractMaturity::Incomplete,
        inference_rule: BuiltinInferenceRule::Identity(IdentityInferenceRule::new(
            inference::error,
        )),
        compatibility: BuiltinCompatibility::Matlab,
        async_behavior: BuiltinAsyncBehavior::MaySuspend,
        purity: BuiltinPurity::Impure,
        semantic_kind: BuiltinSemanticKind::General,
        workspace_effect: None,
        environment_effect: None,
        effects: &[EffectKind::MaySuspend, EffectKind::MayThrow],
        capabilities: &[],
    },
    placement: PLACEMENT,
    link: LINK,
    bindings: &REQUIRED_DEFAULT_BINDING,
    extensions: &ERROR_EXTENSIONS,
    integer_capabilities: &ERROR_INTEGER_CAPABILITIES,
    integer_audit: None,
    suppress_auto_output: true,
};

pub const RETHROW_CATALOG_ENTRY: BuiltinCatalogEntry = BuiltinCatalogEntry {
    audience: crate::BuiltinCatalogAudience::Public,
    provenance: BuiltinCatalogProvenance::new(file!(), module_path!()),
    identity: BuiltinCatalogIdentity { name: "rethrow" },
    category: "diagnostics",
    documentation: documentation::RETHROW,
    descriptor: &RETHROW_DESCRIPTOR,
    contract: BuiltinContractDeclaration {
        maturity: BuiltinContractMaturity::Incomplete,
        inference_rule: BuiltinInferenceRule::Identity(IdentityInferenceRule::new(
            inference::rethrow,
        )),
        compatibility: BuiltinCompatibility::Matlab,
        async_behavior: BuiltinAsyncBehavior::NeverSuspends,
        purity: BuiltinPurity::Impure,
        semantic_kind: BuiltinSemanticKind::General,
        workspace_effect: None,
        environment_effect: None,
        effects: &[EffectKind::MayThrow],
        capabilities: &[],
    },
    placement: PLACEMENT,
    link: LINK,
    bindings: &REQUIRED_DEFAULT_BINDING,
    extensions: &RETHROW_EXTENSIONS,
    integer_capabilities: &[],
    integer_audit: Some(&RETHROW_INTEGER_AUDIT),
    suppress_auto_output: true,
};
