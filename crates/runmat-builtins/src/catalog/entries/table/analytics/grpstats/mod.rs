use crate::*;
use runmat_types::{CallInference, CallRequest};

mod descriptor;
mod documentation;
mod extensions;
mod inference;
mod integer;
#[cfg(test)]
mod tests;

pub use descriptor::GRPSTATS_DESCRIPTOR;
pub use extensions::{
    GRPSTATS_EXTENSIONS, GRPSTATS_INTEGER_ALPHA_EXTENSION, GRPSTATS_INTEGER_DATA_EXTENSION,
    GRPSTATS_INTEGER_SELECTOR_EXTENSION, GRPSTATS_RESIDENT_INPUT_EXTENSION,
};
pub use integer::GRPSTATS_INTEGER_CAPABILITIES;

pub(super) const ENTRY: BuiltinCatalogEntry = BuiltinCatalogEntry {
    audience: crate::BuiltinCatalogAudience::Public,
    provenance: BuiltinCatalogProvenance::new(file!(), module_path!()),
    identity: BuiltinCatalogIdentity { name: "grpstats" },
    category: "stats/summary",
    documentation: documentation::DOCUMENTATION,
    descriptor: &GRPSTATS_DESCRIPTOR,
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
        effects: &[
            runmat_types::EffectKind::MaySuspend,
            runmat_types::EffectKind::MayThrow,
        ],
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
        execution_stack: runmat_types::ExecutionStackRequirement::Any,
        artifact_dependencies: &[],
    },
    bindings: &REQUIRED_DEFAULT_BINDING,
    extensions: &GRPSTATS_EXTENSIONS,
    integer_capabilities: &GRPSTATS_INTEGER_CAPABILITIES,
    integer_audit: None,
    suppress_auto_output: false,
};
