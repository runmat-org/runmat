use super::{
    documentation, GETFIELD_DESCRIPTOR, GETFIELD_EXTENSIONS, GETFIELD_INTEGER_CAPABILITIES,
};
use crate::{
    BuiltinAcceleratorPolicy, BuiltinAsyncBehavior, BuiltinCatalogEntry, BuiltinCatalogIdentity,
    BuiltinCompatibility, BuiltinContractDeclaration, BuiltinContractMaturity,
    BuiltinDistributedPolicy, BuiltinFusionPolicy, BuiltinInferenceRule, BuiltinLinkContract,
    BuiltinLinkPolicy, BuiltinPlacementContract, BuiltinPortability, BuiltinPurity,
    BuiltinReachability, BuiltinResidencyPolicy, BuiltinSemanticKind, IdentityInferenceRule,
    REQUIRED_DEFAULT_BINDING,
};
use runmat_types::{EffectKind, ExecutionStackRequirement};

const EFFECTS: &[EffectKind] = &[
    EffectKind::MaySuspend,
    EffectKind::MayThrow,
    EffectKind::Unknown,
];
pub const GETFIELD_CATALOG_ENTRY: BuiltinCatalogEntry = BuiltinCatalogEntry {
    provenance: crate::BuiltinCatalogProvenance::new(file!(), module_path!()),
    identity: BuiltinCatalogIdentity { name: "getfield" },
    category: "structs/core",
    documentation: documentation::DOCUMENTATION,
    descriptor: &GETFIELD_DESCRIPTOR,
    contract: BuiltinContractDeclaration {
        maturity: BuiltinContractMaturity::Complete,
        inference_rule: BuiltinInferenceRule::Identity(IdentityInferenceRule::new(super::infer)),
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
        accelerator: BuiltinAcceleratorPolicy::Forbidden,
        residency: BuiltinResidencyPolicy::Dynamic,
        fusion: BuiltinFusionPolicy::Boundary,
        distributed: BuiltinDistributedPolicy::InspectHandles,
    },
    link: BuiltinLinkContract {
        reachability: BuiltinReachability::Always,
        policy: BuiltinLinkPolicy::PortableRuntime,
        execution_stack: ExecutionStackRequirement::Any,
        artifact_dependencies: &[],
    },
    bindings: &REQUIRED_DEFAULT_BINDING,
    extensions: GETFIELD_EXTENSIONS,
    integer_capabilities: GETFIELD_INTEGER_CAPABILITIES,
    integer_audit: None,
    suppress_auto_output: false,
};
