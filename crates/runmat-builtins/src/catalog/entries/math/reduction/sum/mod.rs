mod contract;
mod documentation;
mod inference;
#[cfg(test)]
mod tests;

use crate::*;
use runmat_types::ExecutionStackRequirement;

pub use contract::{
    SUM_DESCRIPTOR, SUM_ERROR_INTERNAL, SUM_ERROR_INVALID_ARGUMENT, SUM_ERROR_INVALID_INPUT,
    SUM_INTEGER_CAPABILITIES,
};

const SUM_PREPARATION_FORMS: [BuiltinPreparationForm; 1] = [BuiltinPreparationForm {
    id: BuiltinPreparationFormId(0),
    binding_variant: "default",
    signatures: &[8],
    data_arguments: &[0],
    operation: BuiltinPreparationOperation::Reduction(BuiltinPreparationReduction::Sum),
    provider_requirement: BuiltinPreparationProviderRequirement::InputSupported,
}];

pub const SUM_CATALOG_ENTRY: BuiltinCatalogEntry = BuiltinCatalogEntry {
    provenance: BuiltinCatalogProvenance::new(file!(), module_path!()),
    identity: BuiltinCatalogIdentity { name: "sum" },
    category: "math/reduction",
    documentation: documentation::DOCUMENTATION,
    descriptor: &SUM_DESCRIPTOR,
    contract: BuiltinContractDeclaration {
        maturity: BuiltinContractMaturity::Incomplete,
        inference_rule: BuiltinInferenceRule::Identity(IdentityInferenceRule::new(
            inference::infer,
        )),
        compatibility: BuiltinCompatibility::Matlab,
        async_behavior: BuiltinAsyncBehavior::MaySuspend,
        purity: BuiltinPurity::Pure,
        semantic_kind: BuiltinSemanticKind::Reduction,
        workspace_effect: None,
        environment_effect: None,
        effects: &[
            runmat_types::EffectKind::MaySuspend,
            runmat_types::EffectKind::MayThrow,
        ],
        capabilities: &[],
    },
    placement: BuiltinPlacementContract {
        preparation: BuiltinOperandPreparation::Forms(&SUM_PREPARATION_FORMS),
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
    integer_capabilities: &SUM_INTEGER_CAPABILITIES,
    integer_audit: None,
    suppress_auto_output: false,
};

pub(super) const ENTRIES: &[&BuiltinCatalogEntry] = &[&SUM_CATALOG_ENTRY];
