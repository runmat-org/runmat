mod documentation;

use crate::{
    BuiltinAcceleratorPolicy, BuiltinAsyncBehavior, BuiltinBindingDeclaration, BuiltinCatalogEntry,
    BuiltinCatalogIdentity, BuiltinCompatibility, BuiltinCompletionPolicy,
    BuiltinContractDeclaration, BuiltinContractMaturity, BuiltinDescriptor, BuiltinErrorDescriptor,
    BuiltinExtensionDescriptor, BuiltinExtensionMode, BuiltinFusionPolicy, BuiltinInferenceRule,
    BuiltinIntegerBackendRule, BuiltinIntegerCapabilityDescriptor, BuiltinIntegerComputationDomain,
    BuiltinIntegerInputAvailability, BuiltinIntegerInputCapability, BuiltinIntegerOutputClassRule,
    BuiltinIntegerOverflowRule, BuiltinIntegerOverloadKind, BuiltinIntegerScalarDoubleRule,
    BuiltinLinkContract, BuiltinLinkPolicy, BuiltinOutputMode, BuiltinParamArity,
    BuiltinParamDescriptor, BuiltinParamType, BuiltinPlacementContract, BuiltinPortability,
    BuiltinPurity, BuiltinReachability, BuiltinResidencyPolicy, BuiltinSemanticKind,
    BuiltinSignatureDescriptor, MathInferenceRule, ALL_INTEGER_CLASSES,
};
use runmat_types::{EffectKind, ExecutionStackRequirement};

use documentation::DOCUMENTATION;

const OUTPUTS: [BuiltinParamDescriptor; 1] = [BuiltinParamDescriptor {
    name: "Y",
    ty: BuiltinParamType::NumericArray,
    arity: BuiltinParamArity::Required,
    default: None,
    description: "Elementwise step values with the input shape.",
}];
const INPUTS: [BuiltinParamDescriptor; 1] = [BuiltinParamDescriptor {
    name: "X",
    ty: BuiltinParamType::Any,
    arity: BuiltinParamArity::Required,
    default: None,
    description: "Real numeric, logical, character, or symbolic input.",
}];
const SIGNATURES: [BuiltinSignatureDescriptor; 1] = [BuiltinSignatureDescriptor {
    label: "Y = heaviside(X)",
    inputs: &INPUTS,
    outputs: &OUTPUTS,
}];

pub const HEAVISIDE_ERROR_INVALID_INPUT: BuiltinErrorDescriptor = BuiltinErrorDescriptor {
    code: "RM.HEAVISIDE.INVALID_INPUT",
    identifier: Some("RunMat:heaviside:InvalidInput"),
    when: "Input is not a supported real numeric, logical, character, or symbolic value.",
    message: "heaviside: invalid input",
};
pub const HEAVISIDE_ERROR_PROVIDER_FAILED: BuiltinErrorDescriptor = BuiltinErrorDescriptor {
    code: "RM.HEAVISIDE.PROVIDER_FAILED",
    identifier: Some("RunMat:heaviside:ProviderFailed"),
    when: "The owning provider reports a terminal unary_heaviside execution failure.",
    message: "heaviside: GPU provider unary_heaviside failed",
};
pub const HEAVISIDE_ERROR_INTERNAL: BuiltinErrorDescriptor = BuiltinErrorDescriptor {
    code: "RM.HEAVISIDE.INTERNAL",
    identifier: Some("RunMat:heaviside:Internal"),
    when: "Internal tensor construction, provider validation, gather, or restoration fails.",
    message: "heaviside: internal error",
};
const ERRORS: [BuiltinErrorDescriptor; 3] = [
    HEAVISIDE_ERROR_INVALID_INPUT,
    HEAVISIDE_ERROR_PROVIDER_FAILED,
    HEAVISIDE_ERROR_INTERNAL,
];
pub const HEAVISIDE_DESCRIPTOR: BuiltinDescriptor = BuiltinDescriptor {
    signatures: &SIGNATURES,
    output_mode: BuiltinOutputMode::Fixed,
    completion_policy: BuiltinCompletionPolicy::Public,
    errors: &ERRORS,
};

pub const HEAVISIDE_INTEGER_INPUT_EXTENSION: BuiltinExtensionDescriptor =
    BuiltinExtensionDescriptor {
        id: "heaviside-integer-input",
        mode: BuiltinExtensionMode::RunMatOnly,
        description: "heaviside with an integer input is a RunMat extension",
        error_identifier: Some("RunMat:compatibility:HeavisideIntegerInputExtension"),
    };
pub const HEAVISIDE_LOGICAL_INPUT_EXTENSION: BuiltinExtensionDescriptor =
    BuiltinExtensionDescriptor {
        id: "heaviside-logical-input",
        mode: BuiltinExtensionMode::RunMatOnly,
        description: "heaviside with a logical input is a RunMat extension",
        error_identifier: Some("RunMat:compatibility:HeavisideLogicalInputExtension"),
    };
pub const HEAVISIDE_CHARACTER_INPUT_EXTENSION: BuiltinExtensionDescriptor =
    BuiltinExtensionDescriptor {
        id: "heaviside-character-input",
        mode: BuiltinExtensionMode::RunMatOnly,
        description: "heaviside with a character input is a RunMat extension",
        error_identifier: Some("RunMat:compatibility:HeavisideCharacterInputExtension"),
    };
pub const HEAVISIDE_GPU_INPUT_EXTENSION: BuiltinExtensionDescriptor = BuiltinExtensionDescriptor {
    id: "heaviside-gpu-input",
    mode: BuiltinExtensionMode::RunMatOnly,
    description: "heaviside with a gpuArray input is a RunMat extension",
    error_identifier: Some("RunMat:compatibility:HeavisideGpuInputExtension"),
};
pub const HEAVISIDE_EXTENSIONS: [BuiltinExtensionDescriptor; 4] = [
    HEAVISIDE_INTEGER_INPUT_EXTENSION,
    HEAVISIDE_LOGICAL_INPUT_EXTENSION,
    HEAVISIDE_CHARACTER_INPUT_EXTENSION,
    HEAVISIDE_GPU_INPUT_EXTENSION,
];

const INTEGER_INPUTS: [BuiltinIntegerInputCapability; 1] = [BuiltinIntegerInputCapability {
    name: "X",
    classes: &ALL_INTEGER_CLASSES,
    availability: BuiltinIntegerInputAvailability::RunMatOnly,
    scalar_double: BuiltinIntegerScalarDoubleRule::NotApplicable,
    notes:
        "All eight real integer classes are classified directly from authoritative integer storage.",
}];
pub const HEAVISIDE_INTEGER_CAPABILITIES: [BuiltinIntegerCapabilityDescriptor; 1] =
    [BuiltinIntegerCapabilityDescriptor {
        form: "Y = heaviside(integer_X)",
        inputs: &INTEGER_INPUTS,
        computation_domain: BuiltinIntegerComputationDomain::ExactInteger,
        output_class: BuiltinIntegerOutputClassRule::Double,
        overflow: BuiltinIntegerOverflowRule::NotApplicable,
        backend: BuiltinIntegerBackendRule::GatherFallback,
        overload: BuiltinIntegerOverloadKind::ElementwiseShapePreserving,
        notes: "Integer sign classification is exact and returns double values 0, 0.5, or 1. Resident integer input is gathered through its exact owner before classification.",
    }];

const BINDINGS: [BuiltinBindingDeclaration; 1] = crate::REQUIRED_DEFAULT_BINDING;
const EFFECTS: [EffectKind; 1] = [EffectKind::MayThrow];

pub const HEAVISIDE_CATALOG_ENTRY: BuiltinCatalogEntry = BuiltinCatalogEntry {
    provenance: crate::BuiltinCatalogProvenance::new(file!(), module_path!()),
    identity: BuiltinCatalogIdentity { name: "heaviside" },
    category: "math/elementwise",
    documentation: DOCUMENTATION,
    descriptor: &HEAVISIDE_DESCRIPTOR,
    contract: BuiltinContractDeclaration {
        maturity: BuiltinContractMaturity::Complete,
        inference_rule: BuiltinInferenceRule::Math(MathInferenceRule::Heaviside),
        compatibility: BuiltinCompatibility::Matlab,
        async_behavior: BuiltinAsyncBehavior::NeverSuspends,
        purity: BuiltinPurity::Pure,
        semantic_kind: BuiltinSemanticKind::General,
        workspace_effect: None,
        environment_effect: None,
        effects: &EFFECTS,
        capabilities: &[],
    },
    placement: BuiltinPlacementContract {
        portability: BuiltinPortability::NativeAndWasm,
        accelerator: BuiltinAcceleratorPolicy::Optional,
        residency: BuiltinResidencyPolicy::Dynamic,
        fusion: BuiltinFusionPolicy::Candidate,
        distributed: crate::BuiltinDistributedPolicy::MapUnary,
    },
    link: BuiltinLinkContract {
        reachability: BuiltinReachability::Always,
        policy: BuiltinLinkPolicy::PortableRuntime,
        execution_stack: ExecutionStackRequirement::Any,
        artifact_dependencies: &[],
    },
    bindings: &BINDINGS,
    extensions: &HEAVISIDE_EXTENSIONS,
    integer_capabilities: &HEAVISIDE_INTEGER_CAPABILITIES,
    integer_audit: None,
    suppress_auto_output: false,
};

pub(super) const ENTRIES: &[&BuiltinCatalogEntry] = &[&HEAVISIDE_CATALOG_ENTRY];
