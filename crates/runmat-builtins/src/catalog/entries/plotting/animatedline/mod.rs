use crate::*;
mod contract;
mod documentation;
mod inference;
#[cfg(test)]
mod tests;
pub use contract::*;
pub const ANIMATEDLINE_CATALOG_ENTRY: BuiltinCatalogEntry = BuiltinCatalogEntry {
    audience: crate::BuiltinCatalogAudience::Public,
    provenance: BuiltinCatalogProvenance::new(file!(), module_path!()),
    identity: BuiltinCatalogIdentity {
        name: "animatedline",
    },
    category: "plotting",
    documentation: documentation::DOCUMENTATION,
    descriptor: &ANIMATEDLINE_DESCRIPTOR,
    contract: BuiltinContractDeclaration {
        maturity: BuiltinContractMaturity::Complete,
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
            runmat_types::EffectKind::EnvironmentRead,
            runmat_types::EffectKind::UserInterface,
            runmat_types::EffectKind::HostCallback,
            runmat_types::EffectKind::MaySuspend,
            runmat_types::EffectKind::MayThrow,
        ],
        capabilities: &[runmat_types::CapabilityRequirement::UserInterface],
    },
    placement: BuiltinPlacementContract {
        preparation: BuiltinOperandPreparation::RuntimeOwned,
        portability: BuiltinPortability::NativeAndWasm,
        accelerator: BuiltinAcceleratorPolicy::Optional,
        residency: BuiltinResidencyPolicy::GatherToHost,
        fusion: BuiltinFusionPolicy::Boundary,
        distributed: BuiltinDistributedPolicy::Unsupported,
    },
    link: BuiltinLinkContract {
        reachability: BuiltinReachability::Always,
        policy: BuiltinLinkPolicy::PortableRuntime,
        execution_stack: runmat_types::ExecutionStackRequirement::Any,
        artifact_dependencies: &[],
    },
    bindings: &REQUIRED_DEFAULT_BINDING,
    extensions: &[],
    integer_capabilities: &ANIMATEDLINE_INTEGER_CAPABILITIES,
    integer_audit: None,
    suppress_auto_output: true,
};
pub(super) const ENTRIES: &[&BuiltinCatalogEntry] = &[&ANIMATEDLINE_CATALOG_ENTRY];
