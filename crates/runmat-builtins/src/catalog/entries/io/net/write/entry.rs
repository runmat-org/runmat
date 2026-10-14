use super::*;
use crate::*;
use runmat_types::{CapabilityRequirement, EffectKind, ExecutionStackRequirement};

pub const WRITE_CATALOG_ENTRY: BuiltinCatalogEntry = BuiltinCatalogEntry {
    audience: crate::BuiltinCatalogAudience::Public,
    provenance: BuiltinCatalogProvenance::new(file!(), module_path!()),
    identity: BuiltinCatalogIdentity { name: "write" },
    category: "io/net",
    documentation: documentation::DOCUMENTATION,
    descriptor: &WRITE_DESCRIPTOR,
    contract: BuiltinContractDeclaration {
        maturity: BuiltinContractMaturity::Incomplete,
        inference_rule: BuiltinInferenceRule::Identity(IdentityInferenceRule::new(
            inference::infer,
        )),
        compatibility: BuiltinCompatibility::RunMat,
        async_behavior: BuiltinAsyncBehavior::MaySuspend,
        purity: BuiltinPurity::Impure,
        semantic_kind: BuiltinSemanticKind::Network,
        workspace_effect: None,
        environment_effect: None,
        effects: &[
            EffectKind::Network,
            EffectKind::MaySuspend,
            EffectKind::MayThrow,
        ],
        capabilities: &[CapabilityRequirement::Network],
    },
    placement: BuiltinPlacementContract {
        preparation: BuiltinOperandPreparation::RuntimeOwned,
        portability: BuiltinPortability::NativeOnly,
        accelerator: BuiltinAcceleratorPolicy::Forbidden,
        residency: BuiltinResidencyPolicy::GatherToHost,
        fusion: BuiltinFusionPolicy::Boundary,
        distributed: BuiltinDistributedPolicy::Unsupported,
    },
    link: BuiltinLinkContract {
        reachability: BuiltinReachability::Always,
        policy: BuiltinLinkPolicy::HostRuntime,
        execution_stack: ExecutionStackRequirement::Any,
        artifact_dependencies: &[],
    },
    bindings: &REQUIRED_DEFAULT_BINDING,
    extensions: &WRITE_EXTENSIONS,
    integer_capabilities: &WRITE_INTEGER_CAPABILITIES,
    integer_audit: None,
    suppress_auto_output: false,
};
