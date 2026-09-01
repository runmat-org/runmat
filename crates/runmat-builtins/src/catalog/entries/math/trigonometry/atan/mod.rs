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

mod documentation;

use documentation::ATAN_DOCUMENTATION;

const OUTPUTS: [BuiltinParamDescriptor; 1] = [BuiltinParamDescriptor {
    name: "Y",
    ty: BuiltinParamType::NumericArray,
    arity: BuiltinParamArity::Required,
    default: None,
    description: "Elementwise principal inverse-tangent result in radians.",
}];
const INPUTS: [BuiltinParamDescriptor; 1] = [BuiltinParamDescriptor {
    name: "X",
    ty: BuiltinParamType::Any,
    arity: BuiltinParamArity::Required,
    default: None,
    description: "Single or double real/complex input; integer, logical, and character forms are RunMat-only extensions.",
}];
const LIKE_INPUTS: [BuiltinParamDescriptor; 3] = [
    BuiltinParamDescriptor {
        name: "X",
        ty: BuiltinParamType::Any,
        arity: BuiltinParamArity::Required,
        default: None,
        description: "Single or double real/complex input; integer, logical, and character forms are RunMat-only extensions.",
    },
    BuiltinParamDescriptor {
        name: "like",
        ty: BuiltinParamType::StringScalar,
        arity: BuiltinParamArity::Required,
        default: Some("\"like\""),
        description: "RunMat-only output-template selector.",
    },
    BuiltinParamDescriptor {
        name: "P",
        ty: BuiltinParamType::LikePrototype,
        arity: BuiltinParamArity::Required,
        default: None,
        description: "RunMat-only prototype selecting host/provider residency and real/complex representation.",
    },
];
const SIGNATURES: [BuiltinSignatureDescriptor; 2] = [
    BuiltinSignatureDescriptor {
        label: "Y = atan(X)",
        inputs: &INPUTS,
        outputs: &OUTPUTS,
    },
    BuiltinSignatureDescriptor {
        label: "Y = atan(X, \"like\", P)",
        inputs: &LIKE_INPUTS,
        outputs: &OUTPUTS,
    },
];

pub const ATAN_ERROR_INVALID_INPUT: BuiltinErrorDescriptor = BuiltinErrorDescriptor {
    code: "RM.ATAN.INVALID_INPUT",
    identifier: Some("RunMat:atan:InvalidInput"),
    when: "Input is not a supported numeric, logical, or character value.",
    message: "atan: invalid input",
};
pub const ATAN_ERROR_INVALID_OPTION: BuiltinErrorDescriptor = BuiltinErrorDescriptor {
    code: "RM.ATAN.INVALID_OPTION",
    identifier: Some("RunMat:atan:InvalidOption"),
    when: "Optional arguments after X are malformed or unsupported.",
    message: "atan: invalid option",
};
pub const ATAN_ERROR_ARG_COUNT: BuiltinErrorDescriptor = BuiltinErrorDescriptor {
    code: "RM.ATAN.ARG_COUNT",
    identifier: Some("RunMat:atan:ArgCount"),
    when: "The call supplies an unsupported number of inputs.",
    message: "atan: too many input arguments",
};
pub const ATAN_ERROR_LIKE_PROTOTYPE: BuiltinErrorDescriptor = BuiltinErrorDescriptor {
    code: "RM.ATAN.LIKE_PROTOTYPE",
    identifier: Some("RunMat:atan:LikePrototype"),
    when: "The output prototype or requested representation is unsupported.",
    message: "atan: invalid \"like\" prototype",
};
pub const ATAN_ERROR_GPU_UNAVAILABLE: BuiltinErrorDescriptor = BuiltinErrorDescriptor {
    code: "RM.ATAN.GPU_UNAVAILABLE",
    identifier: Some("RunMat:atan:GpuUnavailable"),
    when: "Provider output is requested through \"like\" without an active provider.",
    message: "atan: GPU provider unavailable",
};
pub const ATAN_ERROR_INTERNAL: BuiltinErrorDescriptor = BuiltinErrorDescriptor {
    code: "RM.ATAN.INTERNAL",
    identifier: Some("RunMat:atan:Internal"),
    when: "Internal gather, conversion, allocation, or provider restoration failed.",
    message: "atan: internal error",
};
pub const ATAN_ERROR_TOO_MANY_OUTPUTS: BuiltinErrorDescriptor = BuiltinErrorDescriptor {
    code: "RM.ATAN.TOO_MANY_OUTPUTS",
    identifier: Some("RunMat:atan:TooManyOutputs"),
    when: "More than one output is requested.",
    message: "atan: too many output arguments",
};
const ERRORS: [BuiltinErrorDescriptor; 7] = [
    ATAN_ERROR_INVALID_INPUT,
    ATAN_ERROR_INVALID_OPTION,
    ATAN_ERROR_ARG_COUNT,
    ATAN_ERROR_LIKE_PROTOTYPE,
    ATAN_ERROR_GPU_UNAVAILABLE,
    ATAN_ERROR_INTERNAL,
    ATAN_ERROR_TOO_MANY_OUTPUTS,
];
pub const ATAN_DESCRIPTOR: BuiltinDescriptor = BuiltinDescriptor {
    signatures: &SIGNATURES,
    output_mode: BuiltinOutputMode::Fixed,
    completion_policy: BuiltinCompletionPolicy::Public,
    errors: &ERRORS,
};

pub const ATAN_INTEGER_INPUT_EXTENSION: BuiltinExtensionDescriptor = BuiltinExtensionDescriptor {
    id: "atan-integer-input",
    mode: BuiltinExtensionMode::RunMatOnly,
    description: "atan with typed-integer input is a RunMat extension",
    error_identifier: Some("RunMat:compatibility:AtanIntegerInputExtension"),
};
pub const ATAN_LOGICAL_INPUT_EXTENSION: BuiltinExtensionDescriptor = BuiltinExtensionDescriptor {
    id: "atan-logical-input",
    mode: BuiltinExtensionMode::RunMatOnly,
    description: "atan with logical input is a RunMat extension",
    error_identifier: Some("RunMat:compatibility:AtanLogicalInputExtension"),
};
pub const ATAN_CHARACTER_INPUT_EXTENSION: BuiltinExtensionDescriptor = BuiltinExtensionDescriptor {
    id: "atan-character-input",
    mode: BuiltinExtensionMode::RunMatOnly,
    description: "atan with character input is a RunMat extension",
    error_identifier: Some("RunMat:compatibility:AtanCharacterInputExtension"),
};
pub const ATAN_LIKE_EXTENSION: BuiltinExtensionDescriptor = BuiltinExtensionDescriptor {
    id: "atan-like-output-template",
    mode: BuiltinExtensionMode::RunMatOnly,
    description: "atan output templating with the \"like\" syntax is a RunMat extension",
    error_identifier: Some("RunMat:compatibility:AtanLikeOutputTemplateExtension"),
};
pub const ATAN_EXTENSIONS: [BuiltinExtensionDescriptor; 4] = [
    ATAN_INTEGER_INPUT_EXTENSION,
    ATAN_LOGICAL_INPUT_EXTENSION,
    ATAN_CHARACTER_INPUT_EXTENSION,
    ATAN_LIKE_EXTENSION,
];

const INTEGER_INPUTS: [BuiltinIntegerInputCapability; 1] = [BuiltinIntegerInputCapability {
    name: "X",
    classes: &ALL_INTEGER_CLASSES,
    availability: BuiltinIntegerInputAvailability::RunMatOnly,
    scalar_double: BuiltinIntegerScalarDoubleRule::NotApplicable,
    notes: "All eight real integer classes enter the declared binary64 inverse-tangent computation boundary.",
}];
pub const ATAN_INTEGER_CAPABILITIES: [BuiltinIntegerCapabilityDescriptor; 1] =
    [BuiltinIntegerCapabilityDescriptor {
        form: "Y = atan(integer_X)",
        inputs: &INTEGER_INPUTS,
        computation_domain: BuiltinIntegerComputationDomain::FloatingPoint,
        output_class: BuiltinIntegerOutputClassRule::Double,
        overflow: BuiltinIntegerOverflowRule::NotApplicable,
        backend: BuiltinIntegerBackendRule::GatherFallback,
        overload: BuiltinIntegerOverloadKind::ElementwiseShapePreserving,
        notes: "Authoritative integer storage converts explicitly at the binary64 algorithm boundary; the real double result preserves shape and resident results return to their owner before any independently gated output template is applied.",
    }];

const BINDINGS: [BuiltinBindingDeclaration; 1] = crate::REQUIRED_DEFAULT_BINDING;
const EFFECTS: [EffectKind; 2] = [EffectKind::MaySuspend, EffectKind::MayThrow];
pub const ATAN_CATALOG_ENTRY: BuiltinCatalogEntry = BuiltinCatalogEntry {
    identity: BuiltinCatalogIdentity { name: "atan" },
    category: "math/trigonometry",
    documentation: ATAN_DOCUMENTATION,
    descriptor: &ATAN_DESCRIPTOR,
    contract: BuiltinContractDeclaration {
        maturity: BuiltinContractMaturity::Complete,
        inference_rule: BuiltinInferenceRule::Math(MathInferenceRule::InverseTrigonometric(
            InverseTrigonometricFunction::Tangent,
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
        portability: BuiltinPortability::NativeAndWasm,
        accelerator: BuiltinAcceleratorPolicy::Optional,
        residency: BuiltinResidencyPolicy::PreserveInputs,
        fusion: BuiltinFusionPolicy::Candidate,
        distributed: BuiltinDistributedPolicy::MapUnary,
    },
    link: BuiltinLinkContract {
        reachability: BuiltinReachability::Always,
        policy: BuiltinLinkPolicy::PortableRuntime,
        execution_stack: ExecutionStackRequirement::Any,
        artifact_dependencies: &[],
    },
    bindings: &BINDINGS,
    extensions: &ATAN_EXTENSIONS,
    integer_capabilities: &ATAN_INTEGER_CAPABILITIES,
    integer_audit: None,
    suppress_auto_output: false,
};
