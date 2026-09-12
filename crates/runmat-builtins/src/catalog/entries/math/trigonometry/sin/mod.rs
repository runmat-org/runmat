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

mod documentation;

use documentation::SIN_DOCUMENTATION;

const SIN_OUTPUT: [BuiltinParamDescriptor; 1] = [BuiltinParamDescriptor {
    name: "Y",
    ty: BuiltinParamType::NumericArray,
    arity: BuiltinParamArity::Required,
    default: None,
    description: "Elementwise sine result.",
}];
const SIN_INPUT: BuiltinParamDescriptor = BuiltinParamDescriptor {
    name: "X",
    ty: BuiltinParamType::Any,
    arity: BuiltinParamArity::Required,
    default: None,
    description: "Single or double real/complex input; integer, logical, and character forms are RunMat-only extensions.",
};
const SIN_INPUTS: [BuiltinParamDescriptor; 1] = [SIN_INPUT];
const SIN_LIKE_INPUTS: [BuiltinParamDescriptor; 3] = [
    SIN_INPUT,
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
const SIN_SIGNATURES: [BuiltinSignatureDescriptor; 2] = [
    BuiltinSignatureDescriptor {
        label: "Y = sin(X)",
        inputs: &SIN_INPUTS,
        outputs: &SIN_OUTPUT,
    },
    BuiltinSignatureDescriptor {
        label: "Y = sin(X, \"like\", P)",
        inputs: &SIN_LIKE_INPUTS,
        outputs: &SIN_OUTPUT,
    },
];

pub const SIN_ERROR_INVALID_INPUT: BuiltinErrorDescriptor = BuiltinErrorDescriptor {
    code: "RM.SIN.INVALID_INPUT",
    identifier: Some("RunMat:sin:InvalidInput"),
    when: "Input cannot be interpreted as supported numeric, logical, character, or complex data.",
    message: "sin: invalid input",
};
pub const SIN_ERROR_INVALID_OPTION: BuiltinErrorDescriptor = BuiltinErrorDescriptor {
    code: "RM.SIN.INVALID_OPTION",
    identifier: Some("RunMat:sin:InvalidOption"),
    when: "Optional arguments after X are malformed or unsupported.",
    message: "sin: invalid option",
};
pub const SIN_ERROR_ARG_COUNT: BuiltinErrorDescriptor = BuiltinErrorDescriptor {
    code: "RM.SIN.ARG_COUNT",
    identifier: Some("RunMat:sin:ArgCount"),
    when: "Too many input arguments were supplied.",
    message: "sin: too many input arguments",
};
pub const SIN_ERROR_LIKE_PROTOTYPE: BuiltinErrorDescriptor = BuiltinErrorDescriptor {
    code: "RM.SIN.LIKE_PROTOTYPE",
    identifier: Some("RunMat:sin:LikePrototype"),
    when: "The \"like\" prototype is unsupported for this output conversion path.",
    message: "sin: invalid \"like\" prototype",
};
pub const SIN_ERROR_GPU_UNAVAILABLE: BuiltinErrorDescriptor = BuiltinErrorDescriptor {
    code: "RM.SIN.GPU_UNAVAILABLE",
    identifier: Some("RunMat:sin:GpuUnavailable"),
    when: "Provider output was requested through \"like\" but no active provider is available.",
    message: "sin: GPU provider unavailable",
};
pub const SIN_ERROR_INTERNAL: BuiltinErrorDescriptor = BuiltinErrorDescriptor {
    code: "RM.SIN.INTERNAL",
    identifier: Some("RunMat:sin:Internal"),
    when: "Internal tensor conversion, allocation, or provider execution failed.",
    message: "sin: internal error",
};
const SIN_ERRORS: [BuiltinErrorDescriptor; 6] = [
    SIN_ERROR_INVALID_INPUT,
    SIN_ERROR_INVALID_OPTION,
    SIN_ERROR_ARG_COUNT,
    SIN_ERROR_LIKE_PROTOTYPE,
    SIN_ERROR_GPU_UNAVAILABLE,
    SIN_ERROR_INTERNAL,
];
pub const SIN_DESCRIPTOR: BuiltinDescriptor = BuiltinDescriptor {
    signatures: &SIN_SIGNATURES,
    output_mode: BuiltinOutputMode::Fixed,
    completion_policy: BuiltinCompletionPolicy::Public,
    errors: &SIN_ERRORS,
};

pub const SIN_INTEGER_INPUT_EXTENSION: BuiltinExtensionDescriptor = BuiltinExtensionDescriptor {
    id: "sin-integer-input",
    mode: BuiltinExtensionMode::RunMatOnly,
    description: "sin with typed-integer input is a RunMat extension",
    error_identifier: Some("RunMat:compatibility:SinIntegerInputExtension"),
};
pub const SIN_LOGICAL_INPUT_EXTENSION: BuiltinExtensionDescriptor = BuiltinExtensionDescriptor {
    id: "sin-logical-input",
    mode: BuiltinExtensionMode::RunMatOnly,
    description: "sin with logical input is a RunMat extension",
    error_identifier: Some("RunMat:compatibility:SinLogicalInputExtension"),
};
pub const SIN_CHARACTER_INPUT_EXTENSION: BuiltinExtensionDescriptor = BuiltinExtensionDescriptor {
    id: "sin-character-input",
    mode: BuiltinExtensionMode::RunMatOnly,
    description: "sin with character input is a RunMat extension",
    error_identifier: Some("RunMat:compatibility:SinCharacterInputExtension"),
};
pub const SIN_LIKE_OUTPUT_EXTENSION: BuiltinExtensionDescriptor = BuiltinExtensionDescriptor {
    id: "sin-like-output",
    mode: BuiltinExtensionMode::RunMatOnly,
    description: "sin with a like output prototype is a RunMat extension",
    error_identifier: Some("RunMat:compatibility:SinLikeOutputExtension"),
};
pub const SIN_EXTENSIONS: [BuiltinExtensionDescriptor; 4] = [
    SIN_INTEGER_INPUT_EXTENSION,
    SIN_LOGICAL_INPUT_EXTENSION,
    SIN_CHARACTER_INPUT_EXTENSION,
    SIN_LIKE_OUTPUT_EXTENSION,
];

const SIN_INTEGER_INPUT: [BuiltinIntegerInputCapability; 1] = [BuiltinIntegerInputCapability {
    name: "X",
    classes: &ALL_INTEGER_CLASSES,
    availability: BuiltinIntegerInputAvailability::RunMatOnly,
    scalar_double: BuiltinIntegerScalarDoubleRule::NotApplicable,
    notes: "All eight real integer classes are admitted only when every value is exactly representable at the binary64 transcendental boundary.",
}];
pub const SIN_INTEGER_CAPABILITIES: [BuiltinIntegerCapabilityDescriptor; 1] =
    [BuiltinIntegerCapabilityDescriptor {
        form: "Y = sin(integer_X)",
        inputs: &SIN_INTEGER_INPUT,
        computation_domain: BuiltinIntegerComputationDomain::FloatingPoint,
        output_class: BuiltinIntegerOutputClassRule::Double,
        overflow: BuiltinIntegerOverflowRule::Error,
        backend: BuiltinIntegerBackendRule::GatherFallback,
        overload: BuiltinIntegerOverloadKind::ElementwiseShapePreserving,
        notes: "RunMat mode validates authoritative integer storage before conversion; provider-resident input may gather and follows the declared output placement policy.",
    }];

const SIN_BINDINGS: [BuiltinBindingDeclaration; 1] = crate::REQUIRED_DEFAULT_BINDING;
const SIN_EFFECTS: [EffectKind; 1] = [EffectKind::MayThrow];
pub const SIN_CATALOG_ENTRY: BuiltinCatalogEntry = BuiltinCatalogEntry {
    provenance: crate::BuiltinCatalogProvenance::new(file!(), module_path!()),
    identity: BuiltinCatalogIdentity { name: "sin" },
    category: "math/trigonometry",
    documentation: SIN_DOCUMENTATION,
    descriptor: &SIN_DESCRIPTOR,
    contract: BuiltinContractDeclaration {
        maturity: BuiltinContractMaturity::Complete,
        inference_rule: BuiltinInferenceRule::Math(MathInferenceRule::Trigonometric(
            TrigonometricFunction::Sin,
        )),
        compatibility: BuiltinCompatibility::Matlab,
        async_behavior: BuiltinAsyncBehavior::NeverSuspends,
        purity: BuiltinPurity::Pure,
        semantic_kind: BuiltinSemanticKind::General,
        workspace_effect: None,
        environment_effect: None,
        effects: &SIN_EFFECTS,
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
    bindings: &SIN_BINDINGS,
    extensions: &SIN_EXTENSIONS,
    integer_capabilities: &SIN_INTEGER_CAPABILITIES,
    integer_audit: None,
    suppress_auto_output: false,
};

pub(super) const ENTRIES: &[&BuiltinCatalogEntry] = &[&SIN_CATALOG_ENTRY];
