//! One-shot discriminant classification identity.
mod contract;
mod documentation;
mod inference;
mod options;
#[cfg(test)]
mod tests;

use crate::*;
use runmat_types::{EffectKind, ExecutionStackRequirement};

pub use contract::{
    CLASSIFY_DESCRIPTOR, CLASSIFY_EXTENSIONS, CLASSIFY_INTEGER_CAPABILITIES,
    CLASSIFY_INTEGER_GROUP_EXTENSION, CLASSIFY_INTEGER_PRIOR_EXTENSION,
    CLASSIFY_INTEGER_SAMPLE_EXTENSION, CLASSIFY_INTEGER_TRAINING_EXTENSION,
    CLASSIFY_LOGICAL_PREDICTOR_EXTENSION, CLASSIFY_RESIDENT_INPUT_EXTENSION,
    ERROR_INTERNAL as CLASSIFY_ERROR_INTERNAL,
    ERROR_INVALID_ARGUMENT as CLASSIFY_ERROR_INVALID_ARGUMENT,
    ERROR_NUMERICAL as CLASSIFY_ERROR_NUMERICAL,
};
pub use options::{DiscriminantType, PriorKeyword};

pub const CLASSIFY_CATALOG_ENTRY: BuiltinCatalogEntry = BuiltinCatalogEntry {
    audience: crate::BuiltinCatalogAudience::Public,
    provenance: BuiltinCatalogProvenance::new(file!(), module_path!()),
    identity: BuiltinCatalogIdentity { name: "classify" },
    category: "stats/ml",
    documentation: documentation::DOCUMENTATION,
    descriptor: &CLASSIFY_DESCRIPTOR,
    contract: BuiltinContractDeclaration {
        maturity: BuiltinContractMaturity::Incomplete,
        inference_rule: BuiltinInferenceRule::Identity(IdentityInferenceRule::new(
            inference::infer,
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
        accelerator: BuiltinAcceleratorPolicy::Optional,
        residency: BuiltinResidencyPolicy::GatherToHost,
        fusion: BuiltinFusionPolicy::Boundary,
        distributed: BuiltinDistributedPolicy::MaterializeArguments,
    },
    link: BuiltinLinkContract {
        reachability: BuiltinReachability::Always,
        policy: BuiltinLinkPolicy::PortableRuntime,
        execution_stack: ExecutionStackRequirement::Any,
        artifact_dependencies: &[],
    },
    bindings: &REQUIRED_DEFAULT_BINDING,
    extensions: &CLASSIFY_EXTENSIONS,
    integer_capabilities: &CLASSIFY_INTEGER_CAPABILITIES,
    integer_audit: None,
    suppress_auto_output: false,
};

pub(super) const ENTRIES: &[&BuiltinCatalogEntry] = &[&CLASSIFY_CATALOG_ENTRY];
