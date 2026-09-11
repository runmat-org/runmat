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
    BuiltinSignatureDescriptor, HyperbolicFunction, MathInferenceRule, ALL_INTEGER_CLASSES,
};
use runmat_types::{EffectKind, ExecutionStackRequirement};

mod documentation;
use documentation::TANH_DOCUMENTATION;

const OUTPUTS: [BuiltinParamDescriptor; 1] = [BuiltinParamDescriptor {
    name: "Y",
    ty: BuiltinParamType::NumericArray,
    arity: BuiltinParamArity::Required,
    default: None,
    description: "Elementwise hyperbolic-tangent result.",
}];
const INPUTS: [BuiltinParamDescriptor; 1] = [BuiltinParamDescriptor {
    name: "X",
    ty: BuiltinParamType::Any,
    arity: BuiltinParamArity::Required,
    default: None,
    description: "Single or double real/complex input; integer, logical, and character forms are RunMat-only extensions.",
}];
const SIGNATURES: [BuiltinSignatureDescriptor; 1] = [BuiltinSignatureDescriptor {
    label: "Y = tanh(X)",
    inputs: &INPUTS,
    outputs: &OUTPUTS,
}];

pub const TANH_ERROR_INVALID_INPUT: BuiltinErrorDescriptor = BuiltinErrorDescriptor {
    code: "RM.TANH.INVALID_INPUT",
    identifier: Some("RunMat:tanh:InvalidInput"),
    when: "Input is not a supported numeric, logical, or character value.",
    message: "tanh: invalid input",
};
pub const TANH_ERROR_INTERNAL: BuiltinErrorDescriptor = BuiltinErrorDescriptor {
    code: "RM.TANH.INTERNAL",
    identifier: Some("RunMat:tanh:Internal"),
    when: "Internal gather, conversion, allocation, provider execution, or restoration fails.",
    message: "tanh: internal error",
};
pub const TANH_ERROR_TOO_MANY_OUTPUTS: BuiltinErrorDescriptor = BuiltinErrorDescriptor {
    code: "RM.TANH.TOO_MANY_OUTPUTS",
    identifier: Some("RunMat:tanh:TooManyOutputs"),
    when: "More than one output is requested.",
    message: "tanh: too many output arguments",
};
pub const TANH_ERROR_INEXACT_INTEGER: BuiltinErrorDescriptor = BuiltinErrorDescriptor {
    code: "RM.TANH.INEXACT_INTEGER",
    identifier: None,
    when: "An integer input cannot be represented exactly at the binary64 computation boundary.",
    message: "tanh: integer X values must be exactly representable as double",
};
const ERRORS: [BuiltinErrorDescriptor; 4] = [
    TANH_ERROR_INVALID_INPUT,
    TANH_ERROR_INTERNAL,
    TANH_ERROR_TOO_MANY_OUTPUTS,
    TANH_ERROR_INEXACT_INTEGER,
];
pub const TANH_DESCRIPTOR: BuiltinDescriptor = BuiltinDescriptor {
    signatures: &SIGNATURES,
    output_mode: BuiltinOutputMode::Fixed,
    completion_policy: BuiltinCompletionPolicy::Public,
    errors: &ERRORS,
};

pub const TANH_INTEGER_INPUT_EXTENSION: BuiltinExtensionDescriptor = BuiltinExtensionDescriptor {
    id: "tanh-integer-input",
    mode: BuiltinExtensionMode::RunMatOnly,
    description: "tanh with typed-integer input is a RunMat extension",
    error_identifier: Some("RunMat:compatibility:TanhIntegerInputExtension"),
};
pub const TANH_LOGICAL_INPUT_EXTENSION: BuiltinExtensionDescriptor = BuiltinExtensionDescriptor {
    id: "tanh-logical-input",
    mode: BuiltinExtensionMode::RunMatOnly,
    description: "tanh with logical input is a RunMat extension",
    error_identifier: Some("RunMat:compatibility:TanhLogicalInputExtension"),
};
pub const TANH_CHARACTER_INPUT_EXTENSION: BuiltinExtensionDescriptor = BuiltinExtensionDescriptor {
    id: "tanh-character-input",
    mode: BuiltinExtensionMode::RunMatOnly,
    description: "tanh with character input is a RunMat extension",
    error_identifier: Some("RunMat:compatibility:TanhCharacterInputExtension"),
};
pub const TANH_EXTENSIONS: [BuiltinExtensionDescriptor; 3] = [
    TANH_INTEGER_INPUT_EXTENSION,
    TANH_LOGICAL_INPUT_EXTENSION,
    TANH_CHARACTER_INPUT_EXTENSION,
];
const INTEGER_INPUTS: [BuiltinIntegerInputCapability; 1] = [BuiltinIntegerInputCapability {
    name: "X",
    classes: &ALL_INTEGER_CLASSES,
    availability: BuiltinIntegerInputAvailability::RunMatOnly,
    scalar_double: BuiltinIntegerScalarDoubleRule::NotApplicable,
    notes: "All eight real integer classes require exact binary64 representability at the hyperbolic-tangent computation boundary.",
}];
pub const TANH_INTEGER_CAPABILITIES: [BuiltinIntegerCapabilityDescriptor; 1] =
    [BuiltinIntegerCapabilityDescriptor {
        form: "Y = tanh(integer_X)",
        inputs: &INTEGER_INPUTS,
        computation_domain: BuiltinIntegerComputationDomain::FloatingPoint,
        output_class: BuiltinIntegerOutputClassRule::Double,
        overflow: BuiltinIntegerOverflowRule::NotApplicable,
        backend: BuiltinIntegerBackendRule::GatherFallback,
        overload: BuiltinIntegerOverloadKind::ElementwiseShapePreserving,
        notes: "Authoritative integer storage converts explicitly at the binary64 algorithm boundary; finite results approach unit magnitude.",
    }];

const BINDINGS: [BuiltinBindingDeclaration; 1] = crate::REQUIRED_DEFAULT_BINDING;
const EFFECTS: [EffectKind; 2] = [EffectKind::MaySuspend, EffectKind::MayThrow];
pub const TANH_CATALOG_ENTRY: BuiltinCatalogEntry = BuiltinCatalogEntry {
    provenance: crate::BuiltinCatalogProvenance::new(file!(), module_path!()),
    identity: BuiltinCatalogIdentity { name: "tanh" },
    category: "math/trigonometry",
    documentation: TANH_DOCUMENTATION,
    descriptor: &TANH_DESCRIPTOR,
    contract: BuiltinContractDeclaration {
        maturity: BuiltinContractMaturity::Complete,
        inference_rule: BuiltinInferenceRule::Math(MathInferenceRule::Hyperbolic(
            HyperbolicFunction::Tangent,
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
    extensions: &TANH_EXTENSIONS,
    integer_capabilities: &TANH_INTEGER_CAPABILITIES,
    integer_audit: None,
    suppress_auto_output: false,
};
