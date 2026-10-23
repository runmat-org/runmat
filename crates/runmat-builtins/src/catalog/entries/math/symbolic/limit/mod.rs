use crate::*;
mod contract;
mod documentation;
pub use contract::*;
pub const LIMIT_CATALOG_ENTRY: BuiltinCatalogEntry = BuiltinCatalogEntry {
    audience: BuiltinCatalogAudience::Public,
    provenance: BuiltinCatalogProvenance::new(file!(), module_path!()),
    identity: BuiltinCatalogIdentity { name: "limit" },
    category: "math/symbolic",
    documentation: documentation::DOCUMENTATION,
    descriptor: &LIMIT_DESCRIPTOR,
    contract: BuiltinContractDeclaration {
        maturity: BuiltinContractMaturity::Incomplete,
        inference_rule: BuiltinInferenceRule::Identity(IdentityInferenceRule::new(infer)),
        compatibility: BuiltinCompatibility::Matlab,
        async_behavior: BuiltinAsyncBehavior::NeverSuspends,
        purity: BuiltinPurity::Pure,
        semantic_kind: BuiltinSemanticKind::General,
        workspace_effect: None,
        environment_effect: None,
        effects: &[runmat_types::EffectKind::MayThrow],
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
        execution_stack: runmat_types::ExecutionStackRequirement::Any,
        artifact_dependencies: &[],
    },
    bindings: &REQUIRED_DEFAULT_BINDING,
    extensions: &LIMIT_EXTENSIONS,
    integer_capabilities: &LIMIT_INTEGER_CAPABILITIES,
    integer_audit: None,
    suppress_auto_output: false,
};
fn infer(
    request: &runmat_types::CallRequest,
    entry: &BuiltinCatalogEntry,
) -> runmat_types::CallInference {
    super::inference::infer(
        super::scalar_contract::SymbolicScalarOperation::Limit,
        request,
        entry,
    )
}
pub(super) const ENTRIES: &[&BuiltinCatalogEntry] = &[&LIMIT_CATALOG_ENTRY];
