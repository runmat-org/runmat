use crate::*;

pub(super) const ENTRIES: &[&BuiltinCatalogEntry] = &[&NYQUIST_CATALOG_ENTRY];
use runmat_types::{
    CallInference, CallRequest, CapabilityRequirement, EffectKind, ExecutionStackRequirement,
};

mod contract;
mod documentation;
mod inference;
#[cfg(test)]
mod tests;

pub use contract::{
    NYQUIST_DESCRIPTOR, NYQUIST_ERROR_PLOT_FAILED, NYQUIST_EXTENSIONS, NYQUIST_INTEGER_CAPABILITIES,
};

pub const NYQUIST_CATALOG_ENTRY: BuiltinCatalogEntry = BuiltinCatalogEntry {
    audience: crate::BuiltinCatalogAudience::Public,
    provenance: BuiltinCatalogProvenance::new(file!(), module_path!()),
    identity: BuiltinCatalogIdentity { name: "nyquist" },
    category: "control/response",
    documentation: documentation::DOCUMENTATION,
    descriptor: &NYQUIST_DESCRIPTOR,
    contract: BuiltinContractDeclaration {
        maturity: BuiltinContractMaturity::Incomplete,
        inference_rule: BuiltinInferenceRule::Identity(IdentityInferenceRule::new(
            inference::infer as fn(&CallRequest, &BuiltinCatalogEntry) -> CallInference,
        )),
        compatibility: BuiltinCompatibility::Matlab,
        async_behavior: BuiltinAsyncBehavior::MaySuspend,
        purity: BuiltinPurity::Impure,
        semantic_kind: BuiltinSemanticKind::General,
        workspace_effect: None,
        environment_effect: None,
        effects: &[
            EffectKind::UserInterface,
            EffectKind::MaySuspend,
            EffectKind::MayThrow,
        ],
        capabilities: &[CapabilityRequirement::UserInterface],
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
    extensions: &NYQUIST_EXTENSIONS,
    integer_capabilities: &NYQUIST_INTEGER_CAPABILITIES,
    integer_audit: None,
    suppress_auto_output: true,
};
