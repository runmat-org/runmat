use crate::{
    AccelerationInferenceRule, BuiltinAcceleratorPolicy, BuiltinAsyncBehavior, BuiltinCatalogEntry,
    BuiltinCatalogIdentity, BuiltinCompatibility, BuiltinContractDeclaration,
    BuiltinContractMaturity, BuiltinDistributedPolicy, BuiltinFusionPolicy, BuiltinInferenceRule,
    BuiltinLinkContract, BuiltinLinkPolicy, BuiltinPlacementContract, BuiltinPortability,
    BuiltinPurity, BuiltinReachability, BuiltinResidencyPolicy, BuiltinSemanticKind,
    REQUIRED_DEFAULT_BINDING,
};
use runmat_types::{EffectKind, ExecutionStackRequirement};

use super::{
    documentation, ARRAYFUN_DESCRIPTOR, ARRAYFUN_EXTENSIONS, ARRAYFUN_INTEGER_CAPABILITIES,
};

const EFFECTS: &[EffectKind] = &[
    EffectKind::HostCallback,
    EffectKind::MaySuspend,
    EffectKind::MayThrow,
    EffectKind::Unknown,
];

pub const ARRAYFUN_CATALOG_ENTRY: BuiltinCatalogEntry = BuiltinCatalogEntry {
    identity: BuiltinCatalogIdentity { name: "arrayfun" },
    category: "acceleration/gpu",
    documentation: documentation::DOCUMENTATION,
    descriptor: &ARRAYFUN_DESCRIPTOR,
    contract: BuiltinContractDeclaration {
        maturity: BuiltinContractMaturity::Complete,
        inference_rule: BuiltinInferenceRule::Acceleration(AccelerationInferenceRule::Arrayfun),
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
        residency: BuiltinResidencyPolicy::Dynamic,
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
    extensions: &ARRAYFUN_EXTENSIONS,
    integer_capabilities: &ARRAYFUN_INTEGER_CAPABILITIES,
    integer_audit: None,
    suppress_auto_output: false,
};
