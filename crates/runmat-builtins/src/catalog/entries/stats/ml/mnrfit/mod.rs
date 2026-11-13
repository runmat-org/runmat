//! Nominal multinomial logistic fitting identity.
mod contract;
mod documentation;
mod inference;
mod options;
#[cfg(test)]
mod tests;

use crate::*;
use runmat_types::{EffectKind, ExecutionStackRequirement};

pub use contract::{
    ERROR_CONVERGENCE as MNRFIT_ERROR_CONVERGENCE, ERROR_INTERNAL as MNRFIT_ERROR_INTERNAL,
    ERROR_INVALID_ARGUMENT as MNRFIT_ERROR_INVALID_ARGUMENT, EXTENSIONS as MNRFIT_EXTENSIONS,
    INTEGER_CAPABILITIES as MNRFIT_INTEGER_CAPABILITIES,
    INTEGER_CONTROL_EXTENSION as MNRFIT_INTEGER_CONTROL_EXTENSION,
    INTEGER_PARAMETER_EXTENSION as MNRFIT_INTEGER_PARAMETER_EXTENSION,
    INTEGER_X_EXTENSION as MNRFIT_INTEGER_X_EXTENSION,
    INTEGER_Y_EXTENSION as MNRFIT_INTEGER_Y_EXTENSION,
    LEGACY_OPTIONS_EXTENSION as MNRFIT_LEGACY_OPTIONS_EXTENSION,
    LOGICAL_X_EXTENSION as MNRFIT_LOGICAL_X_EXTENSION, MNRFIT_DESCRIPTOR,
    RESIDENT_INPUT_EXTENSION as MNRFIT_RESIDENT_INPUT_EXTENSION,
};
pub use options::{MnrfitConstant, MnrfitOption, MnrfitStatsetField};

pub const MNRFIT_CATALOG_ENTRY: BuiltinCatalogEntry = BuiltinCatalogEntry {
    audience: crate::BuiltinCatalogAudience::Public,
    provenance: BuiltinCatalogProvenance::new(file!(), module_path!()),
    identity: BuiltinCatalogIdentity { name: "mnrfit" },
    category: "stats/ml",
    documentation: documentation::DOCUMENTATION,
    descriptor: &MNRFIT_DESCRIPTOR,
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
    extensions: &MNRFIT_EXTENSIONS,
    integer_capabilities: &MNRFIT_INTEGER_CAPABILITIES,
    integer_audit: None,
    suppress_auto_output: false,
};

pub(super) const ENTRIES: &[&BuiltinCatalogEntry] = &[&MNRFIT_CATALOG_ENTRY];
