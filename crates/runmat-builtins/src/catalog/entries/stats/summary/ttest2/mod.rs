//! Two-sample Student-t test identity and contracts.
mod contract;
mod documentation;
mod inference;
mod options;
#[cfg(test)]
mod tests;

use crate::*;
use runmat_types::{EffectKind, ExecutionStackRequirement};

pub use contract::{
    ERROR_INTERNAL as TTEST2_ERROR_INTERNAL,
    ERROR_INVALID_ARGUMENT as TTEST2_ERROR_INVALID_ARGUMENT, OUTPUT_H as TTEST2_OUTPUT_H,
    OUTPUT_P as TTEST2_OUTPUT_P, TTEST2_DESCRIPTOR, TTEST2_EXTENSIONS, TTEST2_INTEGER_CAPABILITIES,
    TTEST2_INTEGER_OPTION_EXTENSION, TTEST2_INTEGER_SAMPLE_EXTENSION,
};
pub use options::{Ttest2OptionName, Ttest2Tail, Ttest2Variance};

pub const TTEST2_CATALOG_ENTRY: BuiltinCatalogEntry = BuiltinCatalogEntry {
    audience: crate::BuiltinCatalogAudience::Public,
    provenance: BuiltinCatalogProvenance::new(file!(), module_path!()),
    identity: BuiltinCatalogIdentity { name: "ttest2" },
    category: "stats/summary",
    documentation: documentation::DOCUMENTATION,
    descriptor: &TTEST2_DESCRIPTOR,
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
        residency: BuiltinResidencyPolicy::Dynamic,
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
    extensions: &TTEST2_EXTENSIONS,
    integer_capabilities: &TTEST2_INTEGER_CAPABILITIES,
    integer_audit: None,
    suppress_auto_output: false,
};

pub(super) const ENTRIES: &[&BuiltinCatalogEntry] = &[&TTEST2_CATALOG_ENTRY];
