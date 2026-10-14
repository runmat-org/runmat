//! One-sample Kolmogorov-Smirnov identity and contracts.
mod contract;
mod documentation;
mod inference;
mod options;
#[cfg(test)]
mod tests;

use crate::*;
use runmat_types::{EffectKind, ExecutionStackRequirement};

pub use contract::{
    KSTEST_DESCRIPTOR, KSTEST_ERROR_INTERNAL, KSTEST_ERROR_INVALID_ARGUMENT, KSTEST_EXTENSIONS,
    KSTEST_INTEGER_CAPABILITIES, KSTEST_INTEGER_CDF_EXTENSION, KSTEST_INTEGER_SAMPLE_EXTENSION,
    KSTEST_RESIDENT_INPUT_EXTENSION,
};
pub use options::{KstestNamedCdf, KstestOptionName, KstestTail};

pub const KSTEST_CATALOG_ENTRY: BuiltinCatalogEntry = BuiltinCatalogEntry {
    audience: crate::BuiltinCatalogAudience::Public,
    provenance: BuiltinCatalogProvenance::new(file!(), module_path!()),
    identity: BuiltinCatalogIdentity { name: "kstest" },
    category: "stats/summary",
    documentation: documentation::DOCUMENTATION,
    descriptor: &KSTEST_DESCRIPTOR,
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
    extensions: &KSTEST_EXTENSIONS,
    integer_capabilities: &KSTEST_INTEGER_CAPABILITIES,
    integer_audit: None,
    suppress_auto_output: false,
};

pub(super) const ENTRIES: &[&BuiltinCatalogEntry] = &[&KSTEST_CATALOG_ENTRY];
