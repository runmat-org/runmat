use crate::*;
use runmat_types::{CallInference, CallRequest, EffectKind, ExecutionStackRequirement};

mod contract;
mod documentation;
mod inference;
#[cfg(test)]
mod tests;

pub(super) const ENTRIES: &[&BuiltinCatalogEntry] = &[&TF_CATALOG_ENTRY];

pub use contract::{
    DEFAULT_DISCRETE_SAMPLE_TIME_EXTENSION, EXPLICIT_GPU_INPUT_EXTENSION,
    INTEGER_COEFFICIENT_EXTENSION, INTEGER_SAMPLE_TIME_EXTENSION, LOGICAL_COEFFICIENT_EXTENSION,
    TF_CONSUMED_PROPERTY_EXTENSIONS, TF_DESCRIPTOR, TF_ERROR_DENOMINATOR_INVALID,
    TF_ERROR_INTERNAL, TF_ERROR_INVALID_ARGUMENT, TF_ERROR_INVALID_COEFFICIENTS,
    TF_ERROR_INVALID_MODEL, TF_ERROR_INVALID_OPTION, TF_ERROR_INVALID_SAMPLE_TIME,
    TF_ERROR_INVALID_VARIABLE, TF_ERROR_UNSUPPORTED_MODEL, TF_EXTENSIONS, TF_INTEGER_CAPABILITIES,
    TF_OR_SS_CONSUMED_PROPERTY_EXTENSIONS, VARIABLE_SHORTHAND_EXTENSION,
};

pub const TF_CATALOG_ENTRY: BuiltinCatalogEntry = BuiltinCatalogEntry {
    audience: crate::BuiltinCatalogAudience::Public,
    provenance: BuiltinCatalogProvenance::new(file!(), module_path!()),
    identity: BuiltinCatalogIdentity { name: "tf" },
    category: "control",
    documentation: documentation::DOCUMENTATION,
    descriptor: &TF_DESCRIPTOR,
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
    extensions: &TF_EXTENSIONS,
    integer_capabilities: &TF_INTEGER_CAPABILITIES,
    integer_audit: None,
    suppress_auto_output: false,
};
