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
    BuiltinSignatureDescriptor, InverseHyperbolicFunction, MathInferenceRule, ALL_INTEGER_CLASSES,
};
use runmat_types::{EffectKind, ExecutionStackRequirement};

mod documentation;

use documentation::ATANH_DOCUMENTATION;

const OUTPUTS: [BuiltinParamDescriptor; 1] = [BuiltinParamDescriptor {
    name: "Y",
    ty: BuiltinParamType::NumericArray,
    arity: BuiltinParamArity::Required,
    default: None,
    description: "Elementwise principal inverse-hyperbolic-tangent result.",
}];
const INPUTS: [BuiltinParamDescriptor; 1] = [BuiltinParamDescriptor {
    name: "X",
    ty: BuiltinParamType::Any,
    arity: BuiltinParamArity::Required,
    default: None,
    description: "Single or double real/complex input; integer, logical, and character forms are RunMat-only extensions.",
}];
const SIGNATURES: [BuiltinSignatureDescriptor; 1] = [BuiltinSignatureDescriptor {
    label: "Y = atanh(X)",
    inputs: &INPUTS,
    outputs: &OUTPUTS,
}];

pub const ATANH_ERROR_INVALID_INPUT: BuiltinErrorDescriptor = BuiltinErrorDescriptor {
    code: "RM.ATANH.INVALID_INPUT",
    identifier: Some("RunMat:atanh:InvalidInput"),
    when: "Input is not a supported numeric, logical, or character value.",
    message: "atanh: invalid input",
};
pub const ATANH_ERROR_INTERNAL: BuiltinErrorDescriptor = BuiltinErrorDescriptor {
    code: "RM.ATANH.INTERNAL",
    identifier: Some("RunMat:atanh:Internal"),
    when: "Internal gather, reduction, conversion, allocation, provider execution, or restoration fails.",
    message: "atanh: internal error",
};
pub const ATANH_ERROR_TOO_MANY_OUTPUTS: BuiltinErrorDescriptor = BuiltinErrorDescriptor {
    code: "RM.ATANH.TOO_MANY_OUTPUTS",
    identifier: Some("RunMat:atanh:TooManyOutputs"),
    when: "More than one output is requested.",
    message: "atanh: too many output arguments",
};
const ERRORS: [BuiltinErrorDescriptor; 3] = [
    ATANH_ERROR_INVALID_INPUT,
    ATANH_ERROR_INTERNAL,
    ATANH_ERROR_TOO_MANY_OUTPUTS,
];
pub const ATANH_DESCRIPTOR: BuiltinDescriptor = BuiltinDescriptor {
    signatures: &SIGNATURES,
    output_mode: BuiltinOutputMode::Fixed,
    completion_policy: BuiltinCompletionPolicy::Public,
    errors: &ERRORS,
};

pub const ATANH_INTEGER_INPUT_EXTENSION: BuiltinExtensionDescriptor = BuiltinExtensionDescriptor {
    id: "atanh-integer-input",
    mode: BuiltinExtensionMode::RunMatOnly,
    description: "atanh with typed-integer input is a RunMat extension",
    error_identifier: Some("RunMat:compatibility:AtanhIntegerInputExtension"),
};
pub const ATANH_LOGICAL_INPUT_EXTENSION: BuiltinExtensionDescriptor = BuiltinExtensionDescriptor {
    id: "atanh-logical-input",
    mode: BuiltinExtensionMode::RunMatOnly,
    description: "atanh with logical input is a RunMat extension",
    error_identifier: Some("RunMat:compatibility:AtanhLogicalInputExtension"),
};
pub const ATANH_CHARACTER_INPUT_EXTENSION: BuiltinExtensionDescriptor =
    BuiltinExtensionDescriptor {
        id: "atanh-character-input",
        mode: BuiltinExtensionMode::RunMatOnly,
        description: "atanh with character input is a RunMat extension",
        error_identifier: Some("RunMat:compatibility:AtanhCharacterInputExtension"),
    };
pub const ATANH_GPU_REAL_COMPLEX_EXTENSION: BuiltinExtensionDescriptor =
    BuiltinExtensionDescriptor {
        id: "atanh-gpu-real-complex-promotion",
        mode: BuiltinExtensionMode::RunMatOnly,
        description: "atanh resident real input that requires complex output is a RunMat extension",
        error_identifier: Some("RunMat:compatibility:AtanhGpuRealComplexPromotionExtension"),
    };
pub const ATANH_EXTENSIONS: [BuiltinExtensionDescriptor; 4] = [
    ATANH_INTEGER_INPUT_EXTENSION,
    ATANH_LOGICAL_INPUT_EXTENSION,
    ATANH_CHARACTER_INPUT_EXTENSION,
    ATANH_GPU_REAL_COMPLEX_EXTENSION,
];

const INTEGER_INPUTS: [BuiltinIntegerInputCapability; 1] = [BuiltinIntegerInputCapability {
    name: "X",
    classes: &ALL_INTEGER_CLASSES,
    availability: BuiltinIntegerInputAvailability::RunMatOnly,
    scalar_double: BuiltinIntegerScalarDoubleRule::NotApplicable,
    notes: "All eight real integer classes enter the declared binary64 inverse-hyperbolic-tangent computation boundary.",
}];
pub const ATANH_INTEGER_CAPABILITIES: [BuiltinIntegerCapabilityDescriptor; 1] =
    [BuiltinIntegerCapabilityDescriptor {
        form: "Y = atanh(integer_X)",
        inputs: &INTEGER_INPUTS,
        computation_domain: BuiltinIntegerComputationDomain::FloatingPoint,
        output_class: BuiltinIntegerOutputClassRule::Double,
        overflow: BuiltinIntegerOverflowRule::NotApplicable,
        backend: BuiltinIntegerBackendRule::GatherFallback,
        overload: BuiltinIntegerOverloadKind::ElementwiseShapePreserving,
        notes: "Authoritative integer storage converts explicitly at the binary64 algorithm boundary. Resident values gather and the real or complex-double result restores through their owning provider.",
    }];

const BINDINGS: [BuiltinBindingDeclaration; 1] = crate::REQUIRED_DEFAULT_BINDING;
const EFFECTS: [EffectKind; 2] = [EffectKind::MaySuspend, EffectKind::MayThrow];
pub const ATANH_CATALOG_ENTRY: BuiltinCatalogEntry = BuiltinCatalogEntry {
    identity: BuiltinCatalogIdentity { name: "atanh" },
    category: "math/trigonometry",
    documentation: ATANH_DOCUMENTATION,
    descriptor: &ATANH_DESCRIPTOR,
    contract: BuiltinContractDeclaration {
        maturity: BuiltinContractMaturity::Complete,
        inference_rule: BuiltinInferenceRule::Math(MathInferenceRule::InverseHyperbolic(
            InverseHyperbolicFunction::Tangent,
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
    extensions: &ATANH_EXTENSIONS,
    integer_capabilities: &ATANH_INTEGER_CAPABILITIES,
    integer_audit: None,
    suppress_auto_output: false,
};
