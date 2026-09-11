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

use documentation::ASIN_DOCUMENTATION;

const OUTPUTS: [BuiltinParamDescriptor; 1] = [BuiltinParamDescriptor {
    name: "Y",
    ty: BuiltinParamType::NumericArray,
    arity: BuiltinParamArity::Required,
    default: None,
    description: "Elementwise principal inverse-sine result in radians.",
}];
const INPUTS: [BuiltinParamDescriptor; 1] = [BuiltinParamDescriptor {
    name: "X",
    ty: BuiltinParamType::Any,
    arity: BuiltinParamArity::Required,
    default: None,
    description: "Single or double real/complex input; integer, logical, and character forms are RunMat-only extensions.",
}];
const SIGNATURES: [BuiltinSignatureDescriptor; 1] = [BuiltinSignatureDescriptor {
    label: "Y = asin(X)",
    inputs: &INPUTS,
    outputs: &OUTPUTS,
}];

pub const ASIN_ERROR_INVALID_INPUT: BuiltinErrorDescriptor = BuiltinErrorDescriptor {
    code: "RM.ASIN.INVALID_INPUT",
    identifier: Some("RunMat:asin:InvalidInput"),
    when: "Input is not a supported numeric, logical, or character value.",
    message: "asin: invalid input",
};
pub const ASIN_ERROR_INTERNAL: BuiltinErrorDescriptor = BuiltinErrorDescriptor {
    code: "RM.ASIN.INTERNAL",
    identifier: Some("RunMat:asin:Internal"),
    when: "Internal reduction, gather, conversion, allocation, or provider restoration failed.",
    message: "asin: internal error",
};
pub const ASIN_ERROR_TOO_MANY_OUTPUTS: BuiltinErrorDescriptor = BuiltinErrorDescriptor {
    code: "RM.ASIN.TOO_MANY_OUTPUTS",
    identifier: Some("RunMat:asin:TooManyOutputs"),
    when: "More than one output is requested.",
    message: "asin: too many output arguments",
};
const ERRORS: [BuiltinErrorDescriptor; 3] = [
    ASIN_ERROR_INVALID_INPUT,
    ASIN_ERROR_INTERNAL,
    ASIN_ERROR_TOO_MANY_OUTPUTS,
];
pub const ASIN_DESCRIPTOR: BuiltinDescriptor = BuiltinDescriptor {
    signatures: &SIGNATURES,
    output_mode: BuiltinOutputMode::Fixed,
    completion_policy: BuiltinCompletionPolicy::Public,
    errors: &ERRORS,
};

pub const ASIN_INTEGER_INPUT_EXTENSION: BuiltinExtensionDescriptor = BuiltinExtensionDescriptor {
    id: "asin-integer-input",
    mode: BuiltinExtensionMode::RunMatOnly,
    description: "asin with typed-integer input is a RunMat extension",
    error_identifier: Some("RunMat:compatibility:AsinIntegerInputExtension"),
};
pub const ASIN_LOGICAL_INPUT_EXTENSION: BuiltinExtensionDescriptor = BuiltinExtensionDescriptor {
    id: "asin-logical-input",
    mode: BuiltinExtensionMode::RunMatOnly,
    description: "asin with logical input is a RunMat extension",
    error_identifier: Some("RunMat:compatibility:AsinLogicalInputExtension"),
};
pub const ASIN_CHARACTER_INPUT_EXTENSION: BuiltinExtensionDescriptor = BuiltinExtensionDescriptor {
    id: "asin-character-input",
    mode: BuiltinExtensionMode::RunMatOnly,
    description: "asin with character input is a RunMat extension",
    error_identifier: Some("RunMat:compatibility:AsinCharacterInputExtension"),
};
pub const ASIN_GPU_REAL_COMPLEX_EXTENSION: BuiltinExtensionDescriptor =
    BuiltinExtensionDescriptor {
        id: "asin-gpu-real-complex-promotion",
        mode: BuiltinExtensionMode::RunMatOnly,
        description: "asin resident real input that requires complex output is a RunMat extension",
        error_identifier: Some("RunMat:compatibility:AsinGpuRealComplexPromotionExtension"),
    };
pub const ASIN_EXTENSIONS: [BuiltinExtensionDescriptor; 4] = [
    ASIN_INTEGER_INPUT_EXTENSION,
    ASIN_LOGICAL_INPUT_EXTENSION,
    ASIN_CHARACTER_INPUT_EXTENSION,
    ASIN_GPU_REAL_COMPLEX_EXTENSION,
];

const INTEGER_INPUTS: [BuiltinIntegerInputCapability; 1] = [BuiltinIntegerInputCapability {
    name: "X",
    classes: &ALL_INTEGER_CLASSES,
    availability: BuiltinIntegerInputAvailability::RunMatOnly,
    scalar_double: BuiltinIntegerScalarDoubleRule::NotApplicable,
    notes: "All eight real integer classes enter the declared binary64 inverse-sine computation boundary.",
}];
pub const ASIN_INTEGER_CAPABILITIES: [BuiltinIntegerCapabilityDescriptor; 1] =
    [BuiltinIntegerCapabilityDescriptor {
        form: "Y = asin(integer_X)",
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
pub const ASIN_CATALOG_ENTRY: BuiltinCatalogEntry = BuiltinCatalogEntry {
    provenance: crate::BuiltinCatalogProvenance::new(file!(), module_path!()),
    identity: BuiltinCatalogIdentity { name: "asin" },
    category: "math/trigonometry",
    documentation: ASIN_DOCUMENTATION,
    descriptor: &ASIN_DESCRIPTOR,
    contract: BuiltinContractDeclaration {
        maturity: BuiltinContractMaturity::Complete,
        inference_rule: BuiltinInferenceRule::Math(MathInferenceRule::InverseTrigonometric(
            InverseTrigonometricFunction::Sine,
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
    extensions: &ASIN_EXTENSIONS,
    integer_capabilities: &ASIN_INTEGER_CAPABILITIES,
    integer_audit: None,
    suppress_auto_output: false,
};
