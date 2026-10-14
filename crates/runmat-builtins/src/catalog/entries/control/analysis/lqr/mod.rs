use crate::*;

pub(super) const ENTRIES: &[&BuiltinCatalogEntry] = &[&LQR_CATALOG_ENTRY];
use runmat_types::{CallInference, CallRequest, EffectKind, ExecutionStackRequirement};

mod contract;
mod documentation;
mod inference;
#[cfg(test)]
mod tests;

pub use contract::{
    LQR_DESCRIPTOR, LQR_ERROR_INTERNAL, LQR_ERROR_INVALID_ARGUMENT, LQR_ERROR_INVALID_DIMENSIONS,
    LQR_ERROR_INVALID_MODEL, LQR_ERROR_SOLVER_FAILED, LQR_ERROR_UNSUPPORTED_INPUT,
    LQR_ERROR_UNSUPPORTED_MODEL, LQR_EXTENSIONS, LQR_INTEGER_CAPABILITIES,
    LQR_INTEGER_MATRIX_EXTENSION,
};

pub const LQR_CATALOG_ENTRY: BuiltinCatalogEntry = BuiltinCatalogEntry {
    audience: crate::BuiltinCatalogAudience::Public,
    provenance: BuiltinCatalogProvenance::new(file!(), module_path!()),
    identity: BuiltinCatalogIdentity { name: "lqr" },
    category: "control/analysis",
    documentation: documentation::DOCUMENTATION,
    descriptor: &LQR_DESCRIPTOR,
    contract: BuiltinContractDeclaration {
        maturity: BuiltinContractMaturity::Incomplete,
        inference_rule: BuiltinInferenceRule::Identity(IdentityInferenceRule::new(
            inference::infer as fn(&CallRequest, &BuiltinCatalogEntry) -> CallInference,
        )),
        compatibility: BuiltinCompatibility::Matlab,
        async_behavior: BuiltinAsyncBehavior::MaySuspend,
        purity: BuiltinPurity::Pure,
        semantic_kind: BuiltinSemanticKind::General,
        workspace_effect: None,
        environment_effect: None,
        effects: &[EffectKind::MaySuspend, EffectKind::MayThrow],
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
        execution_stack: ExecutionStackRequirement::Any,
        artifact_dependencies: &[],
    },
    bindings: &REQUIRED_DEFAULT_BINDING,
    extensions: &LQR_EXTENSIONS,
    integer_capabilities: &LQR_INTEGER_CAPABILITIES,
    integer_audit: None,
    suppress_auto_output: false,
};
