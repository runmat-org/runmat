use crate::{
    BuiltinAcceleratorPolicy, BuiltinAsyncBehavior, BuiltinBindingDeclaration, BuiltinCatalogEntry,
    BuiltinCatalogIdentity, BuiltinCompatibility, BuiltinCompletionPolicy,
    BuiltinContractDeclaration, BuiltinContractMaturity, BuiltinDescriptor,
    BuiltinDistributedPolicy, BuiltinErrorDescriptor, BuiltinExtensionDescriptor,
    BuiltinExtensionMode, BuiltinFusionPolicy, BuiltinInferenceRule, BuiltinIntegerBackendRule,
    BuiltinIntegerCapabilityDescriptor, BuiltinIntegerComputationDomain,
    BuiltinIntegerInputAvailability, BuiltinIntegerInputCapability, BuiltinIntegerOutputClassRule,
    BuiltinIntegerOverflowRule, BuiltinIntegerOverloadKind, BuiltinIntegerScalarDoubleRule,
    BuiltinLinkContract, BuiltinLinkPolicy, BuiltinOutputMode, BuiltinParamArity,
    BuiltinParamDescriptor, BuiltinParamType, BuiltinPlacementContract, BuiltinPortability,
    BuiltinPurity, BuiltinReachability, BuiltinResidencyPolicy, BuiltinSemanticKind,
    BuiltinSignatureDescriptor, InverseTrigonometricFunction, MathInferenceRule,
    ALL_INTEGER_CLASSES,
};
use runmat_types::{EffectKind, ExecutionStackRequirement};

use super::documentation::ACOS_DOCUMENTATION;

const OUTPUTS: [BuiltinParamDescriptor; 1] = [BuiltinParamDescriptor {
    name: "Y",
    ty: BuiltinParamType::NumericArray,
    arity: BuiltinParamArity::Required,
    default: None,
    description: "Elementwise principal inverse-cosine result in radians.",
}];
const INPUTS: [BuiltinParamDescriptor; 1] = [BuiltinParamDescriptor {
    name: "X",
    ty: BuiltinParamType::Any,
    arity: BuiltinParamArity::Required,
    default: None,
    description: "Single or double real/complex input; integer, logical, and character forms are RunMat-only extensions.",
}];
const SIGNATURES: [BuiltinSignatureDescriptor; 1] = [BuiltinSignatureDescriptor {
    label: "Y = acos(X)",
    inputs: &INPUTS,
    outputs: &OUTPUTS,
}];

pub const ACOS_ERROR_INVALID_INPUT: BuiltinErrorDescriptor = BuiltinErrorDescriptor {
    code: "RM.ACOS.INVALID_INPUT",
    identifier: Some("RunMat:acos:InvalidInput"),
    when: "Input is not a supported numeric, logical, or character value.",
    message: "acos: invalid input",
};
pub const ACOS_ERROR_INTERNAL: BuiltinErrorDescriptor = BuiltinErrorDescriptor {
    code: "RM.ACOS.INTERNAL",
    identifier: Some("RunMat:acos:Internal"),
    when: "Internal reduction, gather, conversion, allocation, or provider restoration failed.",
    message: "acos: internal error",
};
pub const ACOS_ERROR_TOO_MANY_OUTPUTS: BuiltinErrorDescriptor = BuiltinErrorDescriptor {
    code: "RM.ACOS.TOO_MANY_OUTPUTS",
    identifier: Some("RunMat:acos:TooManyOutputs"),
    when: "More than one output is requested.",
    message: "acos: too many output arguments",
};
const ERRORS: [BuiltinErrorDescriptor; 3] = [
    ACOS_ERROR_INVALID_INPUT,
    ACOS_ERROR_INTERNAL,
    ACOS_ERROR_TOO_MANY_OUTPUTS,
];
pub const ACOS_DESCRIPTOR: BuiltinDescriptor = BuiltinDescriptor {
    signatures: &SIGNATURES,
    output_mode: BuiltinOutputMode::Fixed,
    completion_policy: BuiltinCompletionPolicy::Public,
    errors: &ERRORS,
};

pub const ACOS_INTEGER_INPUT_EXTENSION: BuiltinExtensionDescriptor = BuiltinExtensionDescriptor {
    id: "acos-integer-input",
    mode: BuiltinExtensionMode::RunMatOnly,
    description: "acos with typed-integer input is a RunMat extension",
    error_identifier: Some("RunMat:compatibility:AcosIntegerInputExtension"),
};
pub const ACOS_LOGICAL_INPUT_EXTENSION: BuiltinExtensionDescriptor = BuiltinExtensionDescriptor {
    id: "acos-logical-input",
    mode: BuiltinExtensionMode::RunMatOnly,
    description: "acos with logical input is a RunMat extension",
    error_identifier: Some("RunMat:compatibility:AcosLogicalInputExtension"),
};
pub const ACOS_CHARACTER_INPUT_EXTENSION: BuiltinExtensionDescriptor = BuiltinExtensionDescriptor {
    id: "acos-character-input",
    mode: BuiltinExtensionMode::RunMatOnly,
    description: "acos with character input is a RunMat extension",
    error_identifier: Some("RunMat:compatibility:AcosCharacterInputExtension"),
};
pub const ACOS_GPU_REAL_COMPLEX_EXTENSION: BuiltinExtensionDescriptor =
    BuiltinExtensionDescriptor {
        id: "acos-gpu-real-complex-promotion",
        mode: BuiltinExtensionMode::RunMatOnly,
        description: "acos resident real input that requires complex output is a RunMat extension",
        error_identifier: Some("RunMat:compatibility:AcosGpuRealComplexPromotionExtension"),
    };
pub const ACOS_EXTENSIONS: [BuiltinExtensionDescriptor; 4] = [
    ACOS_INTEGER_INPUT_EXTENSION,
    ACOS_LOGICAL_INPUT_EXTENSION,
    ACOS_CHARACTER_INPUT_EXTENSION,
    ACOS_GPU_REAL_COMPLEX_EXTENSION,
];

const INTEGER_INPUTS: [BuiltinIntegerInputCapability; 1] = [BuiltinIntegerInputCapability {
    name: "X",
    classes: &ALL_INTEGER_CLASSES,
    availability: BuiltinIntegerInputAvailability::RunMatOnly,
    scalar_double: BuiltinIntegerScalarDoubleRule::NotApplicable,
    notes: "All eight real integer classes enter the declared binary64 inverse-cosine computation boundary.",
}];
pub const ACOS_INTEGER_CAPABILITIES: [BuiltinIntegerCapabilityDescriptor; 1] =
    [BuiltinIntegerCapabilityDescriptor {
        form: "Y = acos(integer_X)",
        inputs: &INTEGER_INPUTS,
        computation_domain: BuiltinIntegerComputationDomain::FloatingPoint,
        output_class: BuiltinIntegerOutputClassRule::Double,
        overflow: BuiltinIntegerOverflowRule::NotApplicable,
        backend: BuiltinIntegerBackendRule::GatherFallback,
        overload: BuiltinIntegerOverloadKind::ElementwiseShapePreserving,
        notes: "Authoritative integer storage converts explicitly at the binary64 algorithm boundary; values outside [-1, 1] produce complex double, and resident results return to the owner.",
    }];

const BINDINGS: [BuiltinBindingDeclaration; 1] = crate::REQUIRED_DEFAULT_BINDING;
const EFFECTS: [EffectKind; 1] = [EffectKind::MayThrow];
pub const ACOS_CATALOG_ENTRY: BuiltinCatalogEntry = BuiltinCatalogEntry {
    identity: BuiltinCatalogIdentity { name: "acos" },
    category: "math/trigonometry",
    documentation: ACOS_DOCUMENTATION,
    descriptor: &ACOS_DESCRIPTOR,
    contract: BuiltinContractDeclaration {
        maturity: BuiltinContractMaturity::Complete,
        inference_rule: BuiltinInferenceRule::Math(MathInferenceRule::InverseTrigonometric(
            InverseTrigonometricFunction::Cosine,
        )),
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
        residency: BuiltinResidencyPolicy::PreserveInputs,
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
    extensions: &ACOS_EXTENSIONS,
    integer_capabilities: &ACOS_INTEGER_CAPABILITIES,
    integer_audit: None,
    suppress_auto_output: false,
};
