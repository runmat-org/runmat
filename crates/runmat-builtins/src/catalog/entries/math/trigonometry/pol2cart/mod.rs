mod contract;
mod documentation;
mod inference;
#[cfg(test)]
mod tests;
use crate::*;
pub use contract::*;
use runmat_types::{EffectKind, ExecutionStackRequirement};

pub const POL2CART_CATALOG_ENTRY: BuiltinCatalogEntry = BuiltinCatalogEntry {
    audience: crate::BuiltinCatalogAudience::Public,
    provenance: BuiltinCatalogProvenance::new(file!(), module_path!()),
    identity: BuiltinCatalogIdentity { name: "pol2cart" },
    category: "math/trigonometry",
    documentation: documentation::DOCUMENTATION,
    descriptor: &POL2CART_DESCRIPTOR,
    contract: BuiltinContractDeclaration {
        maturity: BuiltinContractMaturity::Complete,
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
        preparation: BuiltinOperandPreparation::Forms(&[
            BuiltinPreparationForm {
                id: BuiltinPreparationFormId(0),
                binding_variant: "default",
                signatures: &[0],
                data_arguments: &[0, 1],
                operation: BuiltinPreparationOperation::ElementwiseNary { data_arity: 2 },
                provider_requirement: BuiltinPreparationProviderRequirement::InputSupported,
            },
            BuiltinPreparationForm {
                id: BuiltinPreparationFormId(1),
                binding_variant: "default",
                signatures: &[1],
                data_arguments: &[0, 1, 2],
                operation: BuiltinPreparationOperation::ElementwiseNary { data_arity: 3 },
                provider_requirement: BuiltinPreparationProviderRequirement::InputSupported,
            },
        ]),
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
    extensions: &POL2CART_EXTENSIONS,
    integer_capabilities: &POL2CART_INTEGER_CAPABILITIES,
    integer_audit: None,
    suppress_auto_output: false,
};
pub(super) const ENTRIES: &[&BuiltinCatalogEntry] = &[&POL2CART_CATALOG_ENTRY];
