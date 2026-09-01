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
    BuiltinSignatureDescriptor, MathInferenceRule, TrigonometricFunction, ALL_INTEGER_CLASSES,
};
use runmat_types::{EffectKind, ExecutionStackRequirement};

use super::documentation::TAN_DOCUMENTATION;

const TAN_OUTPUT: [BuiltinParamDescriptor; 1] = [BuiltinParamDescriptor {
    name: "Y",
    ty: BuiltinParamType::NumericArray,
    arity: BuiltinParamArity::Required,
    default: None,
    description: "Elementwise tangent result.",
}];
const TAN_INPUT: BuiltinParamDescriptor = BuiltinParamDescriptor {
    name: "X",
    ty: BuiltinParamType::Any,
    arity: BuiltinParamArity::Required,
    default: None,
    description: "Single or double real/complex input; integer, logical, and character forms are RunMat-only extensions.",
};
const TAN_INPUTS: [BuiltinParamDescriptor; 1] = [TAN_INPUT];
const TAN_LIKE_INPUTS: [BuiltinParamDescriptor; 3] = [
    TAN_INPUT,
    BuiltinParamDescriptor {
        name: "like",
        ty: BuiltinParamType::StringScalar,
        arity: BuiltinParamArity::Required,
        default: Some("\"like\""),
        description: "RunMat output-representation selector.",
    },
    BuiltinParamDescriptor {
        name: "P",
        ty: BuiltinParamType::LikePrototype,
        arity: BuiltinParamArity::Required,
        default: None,
        description:
            "Prototype selecting host or provider residency and real or complex representation.",
    },
];
const TAN_SIGNATURES: [BuiltinSignatureDescriptor; 2] = [
    BuiltinSignatureDescriptor {
        label: "Y = tan(X)",
        inputs: &TAN_INPUTS,
        outputs: &TAN_OUTPUT,
    },
    BuiltinSignatureDescriptor {
        label: "Y = tan(X, \"like\", P)",
        inputs: &TAN_LIKE_INPUTS,
        outputs: &TAN_OUTPUT,
    },
];

pub const TAN_ERROR_INVALID_INPUT: BuiltinErrorDescriptor = BuiltinErrorDescriptor {
    code: "RM.TAN.INVALID_INPUT",
    identifier: Some("RunMat:tan:InvalidInput"),
    when: "Input cannot be interpreted as supported numeric, logical, character, or complex data.",
    message: "tan: invalid input",
};
pub const TAN_ERROR_INVALID_OPTION: BuiltinErrorDescriptor = BuiltinErrorDescriptor {
    code: "RM.TAN.INVALID_OPTION",
    identifier: Some("RunMat:tan:InvalidOption"),
    when: "Optional arguments after X are malformed or unsupported.",
    message: "tan: invalid option",
};
pub const TAN_ERROR_ARG_COUNT: BuiltinErrorDescriptor = BuiltinErrorDescriptor {
    code: "RM.TAN.ARG_COUNT",
    identifier: Some("RunMat:tan:ArgCount"),
    when: "Too many input arguments were supplied.",
    message: "tan: too many input arguments",
};
pub const TAN_ERROR_LIKE_PROTOTYPE: BuiltinErrorDescriptor = BuiltinErrorDescriptor {
    code: "RM.TAN.LIKE_PROTOTYPE",
    identifier: Some("RunMat:tan:LikePrototype"),
    when: "The \"like\" prototype is unsupported for this output conversion path.",
    message: "tan: invalid \"like\" prototype",
};
pub const TAN_ERROR_GPU_UNAVAILABLE: BuiltinErrorDescriptor = BuiltinErrorDescriptor {
    code: "RM.TAN.GPU_UNAVAILABLE",
    identifier: Some("RunMat:tan:GpuUnavailable"),
    when: "Provider output was requested through \"like\" but no active provider is available.",
    message: "tan: GPU provider unavailable",
};
pub const TAN_ERROR_INTERNAL: BuiltinErrorDescriptor = BuiltinErrorDescriptor {
    code: "RM.TAN.INTERNAL",
    identifier: Some("RunMat:tan:Internal"),
    when: "Internal tensor conversion, allocation, or provider execution failed.",
    message: "tan: internal error",
};
const TAN_ERRORS: [BuiltinErrorDescriptor; 6] = [
    TAN_ERROR_INVALID_INPUT,
    TAN_ERROR_INVALID_OPTION,
    TAN_ERROR_ARG_COUNT,
    TAN_ERROR_LIKE_PROTOTYPE,
    TAN_ERROR_GPU_UNAVAILABLE,
    TAN_ERROR_INTERNAL,
];
pub const TAN_DESCRIPTOR: BuiltinDescriptor = BuiltinDescriptor {
    signatures: &TAN_SIGNATURES,
    output_mode: BuiltinOutputMode::Fixed,
    completion_policy: BuiltinCompletionPolicy::Public,
    errors: &TAN_ERRORS,
};

pub const TAN_INTEGER_INPUT_EXTENSION: BuiltinExtensionDescriptor = BuiltinExtensionDescriptor {
    id: "tan-integer-input",
    mode: BuiltinExtensionMode::RunMatOnly,
    description: "tan with typed-integer input is a RunMat extension",
    error_identifier: Some("RunMat:compatibility:TanIntegerInputExtension"),
};
pub const TAN_LOGICAL_INPUT_EXTENSION: BuiltinExtensionDescriptor = BuiltinExtensionDescriptor {
    id: "tan-logical-input",
    mode: BuiltinExtensionMode::RunMatOnly,
    description: "tan with logical input is a RunMat extension",
    error_identifier: Some("RunMat:compatibility:TanLogicalInputExtension"),
};
pub const TAN_CHARACTER_INPUT_EXTENSION: BuiltinExtensionDescriptor = BuiltinExtensionDescriptor {
    id: "tan-character-input",
    mode: BuiltinExtensionMode::RunMatOnly,
    description: "tan with character input is a RunMat extension",
    error_identifier: Some("RunMat:compatibility:TanCharacterInputExtension"),
};
pub const TAN_LIKE_OUTPUT_EXTENSION: BuiltinExtensionDescriptor = BuiltinExtensionDescriptor {
    id: "tan-like-output",
    mode: BuiltinExtensionMode::RunMatOnly,
    description: "tan with a like output prototype is a RunMat extension",
    error_identifier: Some("RunMat:compatibility:TanLikeOutputExtension"),
};
pub const TAN_EXTENSIONS: [BuiltinExtensionDescriptor; 4] = [
    TAN_INTEGER_INPUT_EXTENSION,
    TAN_LOGICAL_INPUT_EXTENSION,
    TAN_CHARACTER_INPUT_EXTENSION,
    TAN_LIKE_OUTPUT_EXTENSION,
];

const TAN_INTEGER_INPUT: [BuiltinIntegerInputCapability; 1] = [BuiltinIntegerInputCapability {
    name: "X",
    classes: &ALL_INTEGER_CLASSES,
    availability: BuiltinIntegerInputAvailability::RunMatOnly,
    scalar_double: BuiltinIntegerScalarDoubleRule::NotApplicable,
    notes: "All eight real integer classes are admitted only when every value is exactly representable at the binary64 transcendental boundary.",
}];
pub const TAN_INTEGER_CAPABILITIES: [BuiltinIntegerCapabilityDescriptor; 1] =
    [BuiltinIntegerCapabilityDescriptor {
        form: "Y = tan(integer_X)",
        inputs: &TAN_INTEGER_INPUT,
        computation_domain: BuiltinIntegerComputationDomain::FloatingPoint,
        output_class: BuiltinIntegerOutputClassRule::Double,
        overflow: BuiltinIntegerOverflowRule::Error,
        backend: BuiltinIntegerBackendRule::GatherFallback,
        overload: BuiltinIntegerOverloadKind::ElementwiseShapePreserving,
        notes: "RunMat mode validates authoritative integer storage before conversion; provider-resident input may gather and follows the declared output placement policy.",
    }];

const TAN_BINDINGS: [BuiltinBindingDeclaration; 1] = crate::REQUIRED_DEFAULT_BINDING;
const TAN_EFFECTS: [EffectKind; 1] = [EffectKind::MayThrow];
pub const TAN_CATALOG_ENTRY: BuiltinCatalogEntry = BuiltinCatalogEntry {
    identity: BuiltinCatalogIdentity { name: "tan" },
    category: "math/trigonometry",
    documentation: TAN_DOCUMENTATION,
    descriptor: &TAN_DESCRIPTOR,
    contract: BuiltinContractDeclaration {
        maturity: BuiltinContractMaturity::Complete,
        inference_rule: BuiltinInferenceRule::Math(MathInferenceRule::Trigonometric(
            TrigonometricFunction::Tan,
        )),
        compatibility: BuiltinCompatibility::Matlab,
        async_behavior: BuiltinAsyncBehavior::NeverSuspends,
        purity: BuiltinPurity::Pure,
        semantic_kind: BuiltinSemanticKind::General,
        workspace_effect: None,
        environment_effect: None,
        effects: &TAN_EFFECTS,
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
    bindings: &TAN_BINDINGS,
    extensions: &TAN_EXTENSIONS,
    integer_capabilities: &TAN_INTEGER_CAPABILITIES,
    integer_audit: None,
    suppress_auto_output: false,
};
