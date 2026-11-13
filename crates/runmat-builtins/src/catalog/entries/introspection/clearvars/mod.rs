use crate::*;
mod contract;
mod inference;
mod documentation_clearvars;
#[cfg(test)]
mod tests;
pub use contract::*;

pub const CLEARVARS_CATALOG_ENTRY: BuiltinCatalogEntry = BuiltinCatalogEntry {
    audience: crate::BuiltinCatalogAudience::Public,
    provenance: BuiltinCatalogProvenance::new(file!(), module_path!()),
    identity: BuiltinCatalogIdentity { name: "clearvars" },
    category: "introspection",
    documentation: documentation_clearvars::CLEARVARS_DOCUMENTATION,
    descriptor: &CLEARVARS_DESCRIPTOR,
    contract: BuiltinContractDeclaration {
        maturity: BuiltinContractMaturity::Incomplete,
        inference_rule: BuiltinInferenceRule::Identity(IdentityInferenceRule::new(
            inference::infer_clearvars,
        )),
        compatibility: BuiltinCompatibility::Matlab,
        async_behavior: BuiltinAsyncBehavior::MaySuspend,
        purity: BuiltinPurity::Impure,
        semantic_kind: BuiltinSemanticKind::Workspace,
        workspace_effect: Some(BuiltinWorkspaceEffect::ClearsBinding),
        environment_effect: None,
        effects: &[
            runmat_types::EffectKind::WorkspaceWrite,
            runmat_types::EffectKind::HostCallback,
            runmat_types::EffectKind::MaySuspend,
            runmat_types::EffectKind::MayThrow,
        ],
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
        execution_stack: runmat_types::ExecutionStackRequirement::Process,
        artifact_dependencies: &[],
    },
    bindings: &REQUIRED_DEFAULT_BINDING,
    extensions: &[],
    integer_capabilities: &[],
    integer_audit: Some(&CLEARVARS_INTEGER_AUDIT),
    suppress_auto_output: true,
};

pub(super) const ENTRIES: &[&BuiltinCatalogEntry] = &[&CLEARVARS_CATALOG_ENTRY];
