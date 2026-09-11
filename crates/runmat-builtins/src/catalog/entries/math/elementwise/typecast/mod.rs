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
    TYPECAST_DESCRIPTOR, TYPECAST_ERROR_GPU_UNSUPPORTED, TYPECAST_ERROR_INTERNAL,
    TYPECAST_ERROR_INVALID_ARGUMENT, TYPECAST_ERROR_INVALID_INPUT, TYPECAST_INTEGER_CAPABILITIES,
};

const EFFECTS: &[EffectKind] = &[EffectKind::MaySuspend, EffectKind::MayThrow];

pub const TYPECAST_CATALOG_ENTRY: BuiltinCatalogEntry = BuiltinCatalogEntry {
    provenance: crate::BuiltinCatalogProvenance::new(file!(), module_path!()),
    identity: BuiltinCatalogIdentity { name: "typecast" },
    category: "math/elementwise",
    documentation: documentation::DOCUMENTATION,
    descriptor: &TYPECAST_DESCRIPTOR,
    contract: BuiltinContractDeclaration {
        maturity: BuiltinContractMaturity::Complete,
        inference_rule: BuiltinInferenceRule::Math(MathInferenceRule::Typecast),
        compatibility: BuiltinCompatibility::Matlab,
        async_behavior: BuiltinAsyncBehavior::MaySuspend,
        purity: BuiltinPurity::Pure,
        semantic_kind: BuiltinSemanticKind::General,
        workspace_effect: None,
        environment_effect: None,
        effects: EFFECTS,
        capabilities: &[],
    },
    placement: BuiltinPlacementContract {
        portability: BuiltinPortability::NativeAndWasm,
        accelerator: BuiltinAcceleratorPolicy::Optional,
        residency: BuiltinResidencyPolicy::PreserveInputs,
        fusion: BuiltinFusionPolicy::Boundary,
        distributed: BuiltinDistributedPolicy::MaterializeArguments,
    },
    link: BuiltinLinkContract {
        reachability: BuiltinReachability::Always,
        policy: BuiltinLinkPolicy::PortableRuntime,
        execution_stack: ExecutionStackRequirement::Any,
        artifact_dependencies: &[],
    },
    bindings: &REQUIRED_DEFAULT_BINDING,
    extensions: &[],
    integer_capabilities: &TYPECAST_INTEGER_CAPABILITIES,
    integer_audit: None,
    suppress_auto_output: false,
};

pub(in crate::catalog) use inference::infer;
pub(super) const ENTRIES: &[&BuiltinCatalogEntry] = &[&TYPECAST_CATALOG_ENTRY];
