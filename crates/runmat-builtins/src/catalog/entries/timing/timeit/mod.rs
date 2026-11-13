mod contract;
mod documentation;
mod inference;

#[cfg(test)]
mod tests;

pub use contract::{
    DESCRIPTOR as TIMEIT_DESCRIPTOR, ERROR_CONTEXT_UNAVAILABLE as TIMEIT_ERROR_CONTEXT_UNAVAILABLE,
    ERROR_EMPTY_HANDLE as TIMEIT_ERROR_EMPTY_HANDLE,
    ERROR_EXPECTS_AT_HANDLE_STRING as TIMEIT_ERROR_EXPECTS_AT_HANDLE_STRING,
    ERROR_FIRST_ARG_KIND as TIMEIT_ERROR_FIRST_ARG_KIND,
    ERROR_HANDLE_KIND as TIMEIT_ERROR_HANDLE_KIND,
    ERROR_NUM_OUTPUTS_FINITE as TIMEIT_ERROR_NUM_OUTPUTS_FINITE,
    ERROR_NUM_OUTPUTS_INTEGER as TIMEIT_ERROR_NUM_OUTPUTS_INTEGER,
    ERROR_NUM_OUTPUTS_NONNEG as TIMEIT_ERROR_NUM_OUTPUTS_NONNEG,
    ERROR_NUM_OUTPUTS_SCALAR as TIMEIT_ERROR_NUM_OUTPUTS_SCALAR,
    ERROR_TOO_MANY_INPUTS as TIMEIT_ERROR_TOO_MANY_INPUTS, EXPLICIT_GPU_NUM_OUTPUTS_EXTENSION,
    INTEGER_NUM_OUTPUTS_EXTENSION,
};

use crate::*;
use runmat_types::{EffectKind, ExecutionStackRequirement};

pub(crate) const ENTRIES: &[&BuiltinCatalogEntry] = &[&TIMEIT_CATALOG_ENTRY];

pub const TIMEIT_CATALOG_ENTRY: BuiltinCatalogEntry = BuiltinCatalogEntry {
    audience: crate::BuiltinCatalogAudience::Public,
    provenance: BuiltinCatalogProvenance::new(file!(), module_path!()),
    identity: BuiltinCatalogIdentity { name: "timeit" },
    category: "timing",
    documentation: documentation::DOCUMENTATION,
    descriptor: &contract::DESCRIPTOR,
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
        effects: &[
            EffectKind::Clock,
            EffectKind::HostCallback,
            EffectKind::MaySuspend,
            EffectKind::MayThrow,
        ],
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
        execution_stack: ExecutionStackRequirement::Process,
        artifact_dependencies: &[],
    },
    bindings: &REQUIRED_DEFAULT_BINDING,
    extensions: &contract::EXTENSIONS,
    integer_capabilities: &contract::INTEGER_CAPABILITIES,
    integer_audit: None,
    suppress_auto_output: false,
};
