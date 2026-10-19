mod contract;
mod documentation;
mod inference;

#[cfg(test)]
mod tests;

pub use contract::{
    ASSERT_COMPLEX_CONDITION_EXTENSION, ASSERT_DESCRIPTOR, ASSERT_ERROR_ASSERTION_FAILED,
    ASSERT_ERROR_INVALID_CONDITION, ASSERT_ERROR_INVALID_INPUT, ASSERT_ERROR_NOT_ENOUGH_INPUTS,
    ASSERT_ERROR_TOO_MANY_OUTPUTS, ASSERT_EXTENSIONS, ASSERT_INTEGER_CAPABILITIES,
    ASSERT_UNQUALIFIED_IDENTIFIER_EXTENSION,
};

use crate::*;
use runmat_types::{EffectKind, ExecutionStackRequirement};

pub(crate) const ENTRIES: &[&BuiltinCatalogEntry] = &[&ASSERT_CATALOG_ENTRY];

pub const ASSERT_CATALOG_ENTRY: BuiltinCatalogEntry = BuiltinCatalogEntry {
    audience: crate::BuiltinCatalogAudience::Public,
    provenance: BuiltinCatalogProvenance::new(file!(), module_path!()),
    identity: BuiltinCatalogIdentity { name: "assert" },
    category: "diagnostics",
    documentation: documentation::DOCUMENTATION,
    descriptor: &ASSERT_DESCRIPTOR,
    contract: BuiltinContractDeclaration {
        maturity: BuiltinContractMaturity::Incomplete,
        inference_rule: BuiltinInferenceRule::Identity(IdentityInferenceRule::new(
            inference::infer,
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
    extensions: &ASSERT_EXTENSIONS,
    integer_capabilities: &ASSERT_INTEGER_CAPABILITIES,
    integer_audit: None,
    suppress_auto_output: true,
};
