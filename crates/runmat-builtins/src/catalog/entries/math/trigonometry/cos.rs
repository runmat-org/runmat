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

use super::documentation::COS_DOCUMENTATION;

const COS_OUTPUT: [BuiltinParamDescriptor; 1] = [BuiltinParamDescriptor {
    name: "Y",
    ty: BuiltinParamType::NumericArray,
    arity: BuiltinParamArity::Required,
    default: None,
    description: "Elementwise cosine result.",
}];
const COS_INPUT: BuiltinParamDescriptor = BuiltinParamDescriptor {
    name: "X",
    ty: BuiltinParamType::Any,
    arity: BuiltinParamArity::Required,
    default: None,
    description: "Single or double real/complex input; integer, logical, and character forms are RunMat-only extensions.",
};
const COS_INPUTS: [BuiltinParamDescriptor; 1] = [COS_INPUT];
const COS_LIKE_INPUTS: [BuiltinParamDescriptor; 3] = [
    COS_INPUT,
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
const COS_SIGNATURES: [BuiltinSignatureDescriptor; 2] = [
    BuiltinSignatureDescriptor {
        label: "Y = cos(X)",
        inputs: &COS_INPUTS,
        outputs: &COS_OUTPUT,
    },
    BuiltinSignatureDescriptor {
        label: "Y = cos(X, \"like\", P)",
        inputs: &COS_LIKE_INPUTS,
        outputs: &COS_OUTPUT,
    },
];

pub const COS_ERROR_INVALID_INPUT: BuiltinErrorDescriptor = BuiltinErrorDescriptor {
    code: "RM.COS.INVALID_INPUT",
    identifier: Some("RunMat:cos:InvalidInput"),
    when: "Input cannot be interpreted as supported numeric, logical, character, or complex data.",
    message: "cos: invalid input",
};
pub const COS_ERROR_INVALID_OPTION: BuiltinErrorDescriptor = BuiltinErrorDescriptor {
    code: "RM.COS.INVALID_OPTION",
    identifier: Some("RunMat:cos:InvalidOption"),
    when: "Optional arguments after X are malformed or unsupported.",
    message: "cos: invalid option",
};
pub const COS_ERROR_ARG_COUNT: BuiltinErrorDescriptor = BuiltinErrorDescriptor {
    code: "RM.COS.ARG_COUNT",
    identifier: Some("RunMat:cos:ArgCount"),
    when: "Too many input arguments were supplied.",
    message: "cos: too many input arguments",
};
pub const COS_ERROR_LIKE_PROTOTYPE: BuiltinErrorDescriptor = BuiltinErrorDescriptor {
    code: "RM.COS.LIKE_PROTOTYPE",
    identifier: Some("RunMat:cos:LikePrototype"),
    when: "The \"like\" prototype is unsupported for this output conversion path.",
    message: "cos: invalid \"like\" prototype",
};
pub const COS_ERROR_GPU_UNAVAILABLE: BuiltinErrorDescriptor = BuiltinErrorDescriptor {
    code: "RM.COS.GPU_UNAVAILABLE",
    identifier: Some("RunMat:cos:GpuUnavailable"),
    when: "Provider output was requested through \"like\" but no active provider is available.",
    message: "cos: GPU provider unavailable",
};
pub const COS_ERROR_INTERNAL: BuiltinErrorDescriptor = BuiltinErrorDescriptor {
    code: "RM.COS.INTERNAL",
    identifier: Some("RunMat:cos:Internal"),
    when: "Internal tensor conversion, allocation, or provider execution failed.",
    message: "cos: internal error",
};
const COS_ERRORS: [BuiltinErrorDescriptor; 6] = [
    COS_ERROR_INVALID_INPUT,
    COS_ERROR_INVALID_OPTION,
    COS_ERROR_ARG_COUNT,
    COS_ERROR_LIKE_PROTOTYPE,
    COS_ERROR_GPU_UNAVAILABLE,
    COS_ERROR_INTERNAL,
];
pub const COS_DESCRIPTOR: BuiltinDescriptor = BuiltinDescriptor {
    signatures: &COS_SIGNATURES,
    output_mode: BuiltinOutputMode::Fixed,
    completion_policy: BuiltinCompletionPolicy::Public,
    errors: &COS_ERRORS,
};

pub const COS_INTEGER_INPUT_EXTENSION: BuiltinExtensionDescriptor = BuiltinExtensionDescriptor {
    id: "cos-integer-input",
    mode: BuiltinExtensionMode::RunMatOnly,
    description: "cos with typed-integer input is a RunMat extension",
    error_identifier: Some("RunMat:compatibility:CosIntegerInputExtension"),
};
pub const COS_LOGICAL_INPUT_EXTENSION: BuiltinExtensionDescriptor = BuiltinExtensionDescriptor {
    id: "cos-logical-input",
    mode: BuiltinExtensionMode::RunMatOnly,
    description: "cos with logical input is a RunMat extension",
    error_identifier: Some("RunMat:compatibility:CosLogicalInputExtension"),
};
pub const COS_CHARACTER_INPUT_EXTENSION: BuiltinExtensionDescriptor = BuiltinExtensionDescriptor {
    id: "cos-character-input",
    mode: BuiltinExtensionMode::RunMatOnly,
    description: "cos with character input is a RunMat extension",
    error_identifier: Some("RunMat:compatibility:CosCharacterInputExtension"),
};
pub const COS_LIKE_OUTPUT_EXTENSION: BuiltinExtensionDescriptor = BuiltinExtensionDescriptor {
    id: "cos-like-output",
    mode: BuiltinExtensionMode::RunMatOnly,
    description: "cos with a like output prototype is a RunMat extension",
    error_identifier: Some("RunMat:compatibility:CosLikeOutputExtension"),
};
pub const COS_EXTENSIONS: [BuiltinExtensionDescriptor; 4] = [
    COS_INTEGER_INPUT_EXTENSION,
    COS_LOGICAL_INPUT_EXTENSION,
    COS_CHARACTER_INPUT_EXTENSION,
    COS_LIKE_OUTPUT_EXTENSION,
];

const COS_INTEGER_INPUT: [BuiltinIntegerInputCapability; 1] = [BuiltinIntegerInputCapability {
    name: "X",
    classes: &ALL_INTEGER_CLASSES,
    availability: BuiltinIntegerInputAvailability::RunMatOnly,
    scalar_double: BuiltinIntegerScalarDoubleRule::NotApplicable,
    notes: "All eight real integer classes are admitted only when every value is exactly representable at the binary64 transcendental boundary.",
}];
pub const COS_INTEGER_CAPABILITIES: [BuiltinIntegerCapabilityDescriptor; 1] =
    [BuiltinIntegerCapabilityDescriptor {
        form: "Y = cos(integer_X)",
        inputs: &COS_INTEGER_INPUT,
        computation_domain: BuiltinIntegerComputationDomain::FloatingPoint,
        output_class: BuiltinIntegerOutputClassRule::Double,
        overflow: BuiltinIntegerOverflowRule::Error,
        backend: BuiltinIntegerBackendRule::GatherFallback,
        overload: BuiltinIntegerOverloadKind::ElementwiseShapePreserving,
        notes: "RunMat mode validates authoritative integer storage before conversion; provider-resident input may gather and follows the declared output placement policy.",
    }];

const COS_BINDINGS: [BuiltinBindingDeclaration; 1] = crate::REQUIRED_DEFAULT_BINDING;
const COS_EFFECTS: [EffectKind; 1] = [EffectKind::MayThrow];
pub const COS_CATALOG_ENTRY: BuiltinCatalogEntry = BuiltinCatalogEntry {
    identity: BuiltinCatalogIdentity { name: "cos" },
    category: "math/trigonometry",
    documentation: COS_DOCUMENTATION,
    descriptor: &COS_DESCRIPTOR,
    contract: BuiltinContractDeclaration {
        maturity: BuiltinContractMaturity::Complete,
        inference_rule: BuiltinInferenceRule::Math(MathInferenceRule::Trigonometric(
            TrigonometricFunction::Cos,
        )),
        compatibility: BuiltinCompatibility::Matlab,
        async_behavior: BuiltinAsyncBehavior::NeverSuspends,
        purity: BuiltinPurity::Pure,
        semantic_kind: BuiltinSemanticKind::General,
        workspace_effect: None,
        environment_effect: None,
        effects: &COS_EFFECTS,
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
    bindings: &COS_BINDINGS,
    extensions: &COS_EXTENSIONS,
    integer_capabilities: &COS_INTEGER_CAPABILITIES,
    integer_audit: None,
    suppress_auto_output: false,
};
