mod contract;
pub use contract::*;
mod inference;
mod documentation;
use crate::*;
pub const ZOOM_CATALOG_ENTRY: BuiltinCatalogEntry = BuiltinCatalogEntry {
    audience: BuiltinCatalogAudience::Public,
    provenance: BuiltinCatalogProvenance::new(file!(), module_path!()),
    identity: BuiltinCatalogIdentity { name: "zoom" },
    category: "plotting",
    documentation: documentation::DOCUMENTATION,
    descriptor: &ZOOM_DESCRIPTOR,
    contract: BuiltinContractDeclaration {
        maturity: BuiltinContractMaturity::Incomplete,
        inference_rule: BuiltinInferenceRule::Identity(IdentityInferenceRule::new(
            inference::infer,
        )),
        compatibility: BuiltinCompatibility::Matlab,
        async_behavior: BuiltinAsyncBehavior::NeverSuspends,
        purity: BuiltinPurity::Impure,
        semantic_kind: BuiltinSemanticKind::General,
        workspace_effect: None,
        environment_effect: None,
        effects: &[
            runmat_types::EffectKind::UserInterface,
            runmat_types::EffectKind::HostCallback,
            runmat_types::EffectKind::MayThrow,
        ],
        capabilities: &[runmat_types::CapabilityRequirement::UserInterface],
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
        execution_stack: runmat_types::ExecutionStackRequirement::Any,
        artifact_dependencies: &[],
    },
    bindings: &REQUIRED_DEFAULT_BINDING,
    extensions: &ZOOM_EXTENSIONS,
    integer_capabilities: &ZOOM_INTEGER_CAPABILITIES,
    integer_audit: None,
    suppress_auto_output: true,
};
pub(super) const ENTRIES: &[&BuiltinCatalogEntry] = &[&ZOOM_CATALOG_ENTRY];
