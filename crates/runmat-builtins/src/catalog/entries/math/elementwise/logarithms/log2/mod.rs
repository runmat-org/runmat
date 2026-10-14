mod descriptor;
mod documentation;
mod extensions;
mod integer;

pub use descriptor::{
    LOG2_DESCRIPTOR, LOG2_ERROR_COMPLEX_DISSECTION, LOG2_ERROR_GPU_COMPLEX_INPUT,
    LOG2_ERROR_GPU_DISSECTION, LOG2_ERROR_INTERNAL, LOG2_ERROR_INVALID_INPUT,
    LOG2_ERROR_PROVIDER_OWNERSHIP, LOG2_ERROR_TOO_MANY_OUTPUTS,
};
use documentation::LOG2_DOCUMENTATION;
pub use extensions::{
    LOG2_CHARACTER_EXTENSION, LOG2_EXTENSIONS, LOG2_INTEGER_EXTENSION, LOG2_LOGICAL_EXTENSION,
};
pub use integer::LOG2_INTEGER_CAPABILITIES;

use crate::{
    BuiltinAcceleratorPolicy, BuiltinAsyncBehavior, BuiltinBindingDeclaration, BuiltinCatalogEntry,
    BuiltinCatalogIdentity, BuiltinCompatibility, BuiltinContractDeclaration,
    BuiltinContractMaturity, BuiltinDistributedPolicy, BuiltinFusionPolicy, BuiltinInferenceRule,
    BuiltinLinkContract, BuiltinLinkPolicy, BuiltinPlacementContract, BuiltinPortability,
    BuiltinPurity, BuiltinReachability, BuiltinResidencyPolicy, BuiltinSemanticKind, LogarithmKind,
    MathInferenceRule,
};
use runmat_types::{EffectKind, ExecutionStackRequirement};

const BINDINGS: [BuiltinBindingDeclaration; 1] = crate::REQUIRED_DEFAULT_BINDING;
const EFFECTS: [EffectKind; 2] = [EffectKind::MaySuspend, EffectKind::MayThrow];

pub const LOG2_CATALOG_ENTRY: BuiltinCatalogEntry = BuiltinCatalogEntry {
    audience: crate::BuiltinCatalogAudience::Public,
    provenance: crate::BuiltinCatalogProvenance::new(file!(), module_path!()),
    identity: BuiltinCatalogIdentity { name: "log2" },
    category: "math/elementwise",
    documentation: LOG2_DOCUMENTATION,
    descriptor: &LOG2_DESCRIPTOR,
    contract: BuiltinContractDeclaration {
        maturity: BuiltinContractMaturity::Complete,
        inference_rule: BuiltinInferenceRule::Math(MathInferenceRule::Logarithm(
            LogarithmKind::Binary,
        )),
        compatibility: BuiltinCompatibility::Matlab,
        async_behavior: BuiltinAsyncBehavior::MaySuspend,
        purity: BuiltinPurity::Pure,
        semantic_kind: BuiltinSemanticKind::General,
        workspace_effect: None,
        environment_effect: None,
        effects: &EFFECTS,
        capabilities: &[],
    },
    placement: BuiltinPlacementContract {
        preparation: crate::BuiltinOperandPreparation::Forms(&[crate::BuiltinPreparationForm {
            id: crate::BuiltinPreparationFormId(0),
            binding_variant: "default",
            signatures: &[0],
            data_arguments: &[0],
            operation: crate::BuiltinPreparationOperation::RealUnary(
                runmat_types::RealUnaryPredicate::Nonnegative,
            ),
            provider_requirement: crate::BuiltinPreparationProviderRequirement::InputSupported,
        }]),
        portability: BuiltinPortability::NativeAndWasm,
        accelerator: BuiltinAcceleratorPolicy::Optional,
        residency: BuiltinResidencyPolicy::Dynamic,
        fusion: BuiltinFusionPolicy::Never,
        distributed: BuiltinDistributedPolicy::MapUnary,
    },
    link: BuiltinLinkContract {
        reachability: BuiltinReachability::Always,
        policy: BuiltinLinkPolicy::PortableRuntime,
        execution_stack: ExecutionStackRequirement::Any,
        artifact_dependencies: &[],
    },
    bindings: &BINDINGS,
    extensions: &LOG2_EXTENSIONS,
    integer_capabilities: &LOG2_INTEGER_CAPABILITIES,
    integer_audit: None,
    suppress_auto_output: false,
};
