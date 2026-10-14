use super::*;
use crate::*;
use runmat_types::{CallInference, CallRequest, EffectKind, ExecutionStackRequirement};

pub const CVPARTITION_CATALOG_ENTRY: BuiltinCatalogEntry = BuiltinCatalogEntry {
    audience: crate::BuiltinCatalogAudience::Public,
    provenance: BuiltinCatalogProvenance::new(file!(), module_path!()),
    identity: BuiltinCatalogIdentity {
        name: "cvpartition",
    },
    category: "stats/ml",
    documentation: documentation::CONSTRUCTOR,
    descriptor: &CVPARTITION_DESCRIPTOR,
    contract: BuiltinContractDeclaration {
        maturity: BuiltinContractMaturity::Incomplete,
        inference_rule: BuiltinInferenceRule::Identity(IdentityInferenceRule::new(
            inference::infer_constructor as fn(&CallRequest, &BuiltinCatalogEntry) -> CallInference,
        )),
        compatibility: BuiltinCompatibility::Matlab,
        async_behavior: BuiltinAsyncBehavior::MaySuspend,
        purity: BuiltinPurity::Impure,
        semantic_kind: BuiltinSemanticKind::General,
        workspace_effect: None,
        environment_effect: None,
        effects: &[
            EffectKind::Randomness,
            EffectKind::MaySuspend,
            EffectKind::MayThrow,
        ],
        capabilities: &[],
    },
    placement: BuiltinPlacementContract {
        preparation: BuiltinOperandPreparation::RuntimeOwned,
        portability: BuiltinPortability::NativeAndWasm,
        accelerator: BuiltinAcceleratorPolicy::Forbidden,
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
    extensions: &CVPARTITION_EXTENSIONS,
    integer_capabilities: &CVPARTITION_INTEGER_CAPABILITIES,
    integer_audit: None,
    suppress_auto_output: false,
};

pub const TEST_CATALOG_ENTRY: BuiltinCatalogEntry = mask_entry(
    "test",
    documentation::TEST,
    &TEST_DESCRIPTOR,
    &TEST_EXTENSIONS,
    &TEST_INTEGER_CAPABILITIES,
    inference::infer_test,
);

pub const TRAINING_CATALOG_ENTRY: BuiltinCatalogEntry = mask_entry(
    "training",
    documentation::TRAINING,
    &TRAINING_DESCRIPTOR,
    &TRAINING_EXTENSIONS,
    &TRAINING_INTEGER_CAPABILITIES,
    inference::infer_training,
);

const fn mask_entry(
    name: &'static str,
    documentation: BuiltinDocumentation,
    descriptor: &'static BuiltinDescriptor,
    extensions: &'static [BuiltinExtensionDescriptor],
    integer_capabilities: &'static [BuiltinIntegerCapabilityDescriptor],
    infer: fn(&CallRequest, &BuiltinCatalogEntry) -> CallInference,
) -> BuiltinCatalogEntry {
    BuiltinCatalogEntry {
        audience: crate::BuiltinCatalogAudience::Public,
        provenance: BuiltinCatalogProvenance::new(file!(), module_path!()),
        identity: BuiltinCatalogIdentity { name },
        category: "stats/ml",
        documentation,
        descriptor,
        contract: BuiltinContractDeclaration {
            maturity: BuiltinContractMaturity::Incomplete,
            inference_rule: BuiltinInferenceRule::Identity(IdentityInferenceRule::new(infer)),
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
        extensions,
        integer_capabilities,
        integer_audit: None,
        suppress_auto_output: false,
    }
}
