mod descriptor;
mod documentation;

pub use descriptor::*;

use crate::*;

fn infer(
    request: &runmat_types::CallRequest,
    entry: &BuiltinCatalogEntry,
) -> runmat_types::CallInference {
    super::rgb_input::infer(request, entry, super::rgb_input::RgbConversion::Gray)
}

pub const RGB2GRAY_CATALOG_ENTRY: BuiltinCatalogEntry = BuiltinCatalogEntry {
    audience: crate::BuiltinCatalogAudience::Public,
    provenance: BuiltinCatalogProvenance::new(file!(), module_path!()),
    identity: BuiltinCatalogIdentity { name: "rgb2gray" },
    category: "image/color",
    documentation: documentation::DOCUMENTATION,
    descriptor: &RGB2GRAY_DESCRIPTOR,
    contract: BuiltinContractDeclaration {
        maturity: BuiltinContractMaturity::Incomplete,
        inference_rule: BuiltinInferenceRule::Identity(IdentityInferenceRule::new(infer)),
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
        preparation: BuiltinOperandPreparation::Forms(&[BuiltinPreparationForm {
            id: BuiltinPreparationFormId(0),
            binding_variant: "default",
            signatures: &[0],
            data_arguments: &[0],
            operation: BuiltinPreparationOperation::Sink(BuiltinSinkPreparation::Gather),
            provider_requirement: BuiltinPreparationProviderRequirement::InputSupported,
        }]),
        portability: BuiltinPortability::NativeAndWasm,
        accelerator: BuiltinAcceleratorPolicy::Optional,
        residency: BuiltinResidencyPolicy::Dynamic,
        fusion: BuiltinFusionPolicy::Boundary,
        distributed: BuiltinDistributedPolicy::MaterializeArguments,
    },
    link: BuiltinLinkContract {
        reachability: BuiltinReachability::Always,
        policy: BuiltinLinkPolicy::PortableRuntime,
        execution_stack: runmat_types::ExecutionStackRequirement::Any,
        artifact_dependencies: &[],
    },
    bindings: &REQUIRED_DEFAULT_BINDING,
    extensions: &[],
    integer_capabilities: &RGB2GRAY_INTEGER_CAPABILITIES,
    integer_audit: None,
    suppress_auto_output: false,
};

pub(super) const ENTRIES: &[&BuiltinCatalogEntry] = &[&RGB2GRAY_CATALOG_ENTRY];
