mod contract;
mod documentation;
mod inference;
mod options;
#[cfg(test)]
mod tests;

use crate::*;
use runmat_types::ExecutionStackRequirement;

use super::support::ReductionNaN;
pub use contract::{
    MAX_DESCRIPTOR, MAX_ERROR_INTERNAL, MAX_ERROR_INVALID_ARGUMENT, MAX_ERROR_INVALID_INPUT,
    MAX_ERROR_SIZE_MISMATCH, MAX_INTEGER_CAPABILITIES,
};
pub use options::{MaxComparisonMethod, MaxControl};

pub const MAX_CATALOG_ENTRY: BuiltinCatalogEntry = BuiltinCatalogEntry {
    audience: crate::BuiltinCatalogAudience::Public,
    provenance: BuiltinCatalogProvenance::new(file!(), module_path!()),
    identity: BuiltinCatalogIdentity { name: "max" },
    category: "math/reduction",
    documentation: documentation::DOCUMENTATION,
    descriptor: &MAX_DESCRIPTOR,
    contract: BuiltinContractDeclaration {
        maturity: BuiltinContractMaturity::Incomplete,
        inference_rule: BuiltinInferenceRule::Identity(IdentityInferenceRule::with_fusion_fact(
            inference::infer,
            inference::infer_fusion_fact,
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
        accelerator: BuiltinAcceleratorPolicy::Optional,
        residency: BuiltinResidencyPolicy::Dynamic,
        fusion: BuiltinFusionPolicy::Candidate,
        distributed: BuiltinDistributedPolicy::MaterializeArguments,
    },
    link: BuiltinLinkContract {
        reachability: BuiltinReachability::Always,
        policy: BuiltinLinkPolicy::PortableRuntime,
        execution_stack: ExecutionStackRequirement::Any,
        artifact_dependencies: &[],
    },
    bindings: &REQUIRED_DEFAULT_BINDING,
    extensions: &[],
    integer_capabilities: &MAX_INTEGER_CAPABILITIES,
    integer_audit: None,
    suppress_auto_output: false,
};

pub(super) const ENTRIES: &[&BuiltinCatalogEntry] = &[&MAX_CATALOG_ENTRY];

/// Legacy omit-NaN maximum has the same admitted shape/class/output grammar.
pub(super) fn infer_omit_nan(
    request: &runmat_types::CallRequest,
    entry: &crate::BuiltinCatalogEntry,
) -> runmat_types::CallInference {
    inference::infer(request, entry)
}
