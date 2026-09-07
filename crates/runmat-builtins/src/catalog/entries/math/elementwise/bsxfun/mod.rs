mod contract;
mod documentation;
mod inference;

#[cfg(test)]
mod inference_tests;

use crate::{
    BuiltinAcceleratorPolicy, BuiltinAsyncBehavior, BuiltinCatalogEntry, BuiltinCatalogIdentity,
    BuiltinCompatibility, BuiltinContractDeclaration, BuiltinContractMaturity,
    BuiltinDistributedPolicy, BuiltinFusionPolicy, BuiltinInferenceRule, BuiltinLinkContract,
    BuiltinLinkPolicy, BuiltinPlacementContract, BuiltinPortability, BuiltinPurity,
    BuiltinReachability, BuiltinResidencyPolicy, BuiltinSemanticKind, MathInferenceRule,
    REQUIRED_DEFAULT_BINDING,
};
use runmat_types::{EffectKind, ExecutionStackRequirement};

pub use contract::{
    BSXFUN_DESCRIPTOR, BSXFUN_ERROR_FUNCTION_ERROR, BSXFUN_ERROR_INTERNAL,
    BSXFUN_ERROR_INVALID_FUNCTION, BSXFUN_ERROR_INVALID_INPUT, BSXFUN_ERROR_SIZE_MISMATCH,
    BSXFUN_EXTENSIONS, BSXFUN_INTEGER_CAPABILITIES,
};

const EFFECTS: &[EffectKind] = &[
    EffectKind::HostCallback,
    EffectKind::MaySuspend,
    EffectKind::MayThrow,
    EffectKind::Unknown,
];

pub const BSXFUN_CATALOG_ENTRY: BuiltinCatalogEntry = BuiltinCatalogEntry {
    identity: BuiltinCatalogIdentity { name: "bsxfun" },
    category: "math/elementwise",
    documentation: documentation::DOCUMENTATION,
    descriptor: &BSXFUN_DESCRIPTOR,
    contract: BuiltinContractDeclaration {
        maturity: BuiltinContractMaturity::Complete,
        inference_rule: BuiltinInferenceRule::Math(MathInferenceRule::Bsxfun),
        compatibility: BuiltinCompatibility::Matlab,
        async_behavior: BuiltinAsyncBehavior::MaySuspend,
        purity: BuiltinPurity::Impure,
        semantic_kind: BuiltinSemanticKind::General,
        workspace_effect: None,
        environment_effect: None,
        effects: EFFECTS,
        capabilities: &[],
    },
    placement: BuiltinPlacementContract {
        portability: BuiltinPortability::NativeAndWasm,
        accelerator: BuiltinAcceleratorPolicy::Optional,
        residency: BuiltinResidencyPolicy::GatherToHost,
        fusion: BuiltinFusionPolicy::Boundary,
        distributed: BuiltinDistributedPolicy::MaterializeArguments,
    },
    link: BuiltinLinkContract {
        reachability: BuiltinReachability::Dynamic,
        policy: BuiltinLinkPolicy::PortableRuntime,
        execution_stack: ExecutionStackRequirement::Process,
        artifact_dependencies: &[],
    },
    bindings: &REQUIRED_DEFAULT_BINDING,
    extensions: &BSXFUN_EXTENSIONS,
    integer_capabilities: &BSXFUN_INTEGER_CAPABILITIES,
    integer_audit: None,
    suppress_auto_output: false,
};

pub(in crate::catalog) use inference::infer;
pub(super) const ENTRIES: &[&BuiltinCatalogEntry] = &[&BSXFUN_CATALOG_ENTRY];
