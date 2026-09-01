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

use documentation::SINH_DOCUMENTATION;

const OUTPUTS: [BuiltinParamDescriptor; 1] = [BuiltinParamDescriptor {
    name: "Y",
    ty: BuiltinParamType::NumericArray,
    arity: BuiltinParamArity::Required,
    default: None,
    description: "Elementwise hyperbolic-sine result.",
}];
const INPUTS: [BuiltinParamDescriptor; 1] = [BuiltinParamDescriptor {
    name: "X",
    ty: BuiltinParamType::Any,
    arity: BuiltinParamArity::Required,
    default: None,
    description: "Single or double real/complex input; integer, logical, and character forms are RunMat-only extensions.",
}];
const SIGNATURES: [BuiltinSignatureDescriptor; 1] = [BuiltinSignatureDescriptor {
    label: "Y = sinh(X)",
    inputs: &INPUTS,
    outputs: &OUTPUTS,
}];

pub const SINH_ERROR_INVALID_INPUT: BuiltinErrorDescriptor = BuiltinErrorDescriptor {
    code: "RM.SINH.INVALID_INPUT",
    identifier: Some("RunMat:sinh:InvalidInput"),
    when: "Input is not a supported numeric, logical, or character value.",
    message: "sinh: invalid input",
};
pub const SINH_ERROR_INTERNAL: BuiltinErrorDescriptor = BuiltinErrorDescriptor {
    code: "RM.SINH.INTERNAL",
    identifier: Some("RunMat:sinh:Internal"),
    when: "Internal gather, conversion, allocation, provider execution, or restoration fails.",
    message: "sinh: internal error",
};
pub const SINH_ERROR_TOO_MANY_OUTPUTS: BuiltinErrorDescriptor = BuiltinErrorDescriptor {
    code: "RM.SINH.TOO_MANY_OUTPUTS",
    identifier: Some("RunMat:sinh:TooManyOutputs"),
    when: "More than one output is requested.",
    message: "sinh: too many output arguments",
};
pub const SINH_ERROR_INEXACT_INTEGER: BuiltinErrorDescriptor = BuiltinErrorDescriptor {
    code: "RM.SINH.INEXACT_INTEGER",
    identifier: None,
    when: "An integer input cannot be represented exactly at the binary64 computation boundary.",
    message: "sinh: integer X values must be exactly representable as double",
};
const ERRORS: [BuiltinErrorDescriptor; 4] = [
    SINH_ERROR_INVALID_INPUT,
    SINH_ERROR_INTERNAL,
    SINH_ERROR_TOO_MANY_OUTPUTS,
    SINH_ERROR_INEXACT_INTEGER,
];
pub const SINH_DESCRIPTOR: BuiltinDescriptor = BuiltinDescriptor {
    signatures: &SIGNATURES,
    output_mode: BuiltinOutputMode::Fixed,
    completion_policy: BuiltinCompletionPolicy::Public,
    errors: &ERRORS,
};

pub const SINH_INTEGER_INPUT_EXTENSION: BuiltinExtensionDescriptor = BuiltinExtensionDescriptor {
    id: "sinh-integer-input",
    mode: BuiltinExtensionMode::RunMatOnly,
    description: "sinh with typed-integer input is a RunMat extension",
    error_identifier: Some("RunMat:compatibility:SinhIntegerInputExtension"),
};
pub const SINH_LOGICAL_INPUT_EXTENSION: BuiltinExtensionDescriptor = BuiltinExtensionDescriptor {
    id: "sinh-logical-input",
    mode: BuiltinExtensionMode::RunMatOnly,
    description: "sinh with logical input is a RunMat extension",
    error_identifier: Some("RunMat:compatibility:SinhLogicalInputExtension"),
};
pub const SINH_CHARACTER_INPUT_EXTENSION: BuiltinExtensionDescriptor = BuiltinExtensionDescriptor {
    id: "sinh-character-input",
    mode: BuiltinExtensionMode::RunMatOnly,
    description: "sinh with character input is a RunMat extension",
    error_identifier: Some("RunMat:compatibility:SinhCharacterInputExtension"),
};
pub const SINH_EXTENSIONS: [BuiltinExtensionDescriptor; 3] = [
    SINH_INTEGER_INPUT_EXTENSION,
    SINH_LOGICAL_INPUT_EXTENSION,
    SINH_CHARACTER_INPUT_EXTENSION,
];

const INTEGER_INPUTS: [BuiltinIntegerInputCapability; 1] = [BuiltinIntegerInputCapability {
    name: "X",
    classes: &ALL_INTEGER_CLASSES,
    availability: BuiltinIntegerInputAvailability::RunMatOnly,
    scalar_double: BuiltinIntegerScalarDoubleRule::NotApplicable,
    notes: "All eight real integer classes require exact binary64 representability at the hyperbolic-sine computation boundary.",
}];
pub const SINH_INTEGER_CAPABILITIES: [BuiltinIntegerCapabilityDescriptor; 1] =
    [BuiltinIntegerCapabilityDescriptor {
        form: "Y = sinh(integer_X)",
        inputs: &INTEGER_INPUTS,
        computation_domain: BuiltinIntegerComputationDomain::FloatingPoint,
        output_class: BuiltinIntegerOutputClassRule::Double,
        overflow: BuiltinIntegerOverflowRule::NotApplicable,
        backend: BuiltinIntegerBackendRule::GatherFallback,
        overload: BuiltinIntegerOverloadKind::ElementwiseShapePreserving,
        notes: "Authoritative integer storage converts explicitly at the binary64 algorithm boundary. Large finite values may overflow naturally to infinity.",
    }];

const BINDINGS: [BuiltinBindingDeclaration; 1] = crate::REQUIRED_DEFAULT_BINDING;
const EFFECTS: [EffectKind; 2] = [EffectKind::MaySuspend, EffectKind::MayThrow];
pub const SINH_CATALOG_ENTRY: BuiltinCatalogEntry = BuiltinCatalogEntry {
    identity: BuiltinCatalogIdentity { name: "sinh" },
    category: "math/trigonometry",
    documentation: SINH_DOCUMENTATION,
    descriptor: &SINH_DESCRIPTOR,
    contract: BuiltinContractDeclaration {
        maturity: BuiltinContractMaturity::Complete,
        inference_rule: BuiltinInferenceRule::Math(MathInferenceRule::Hyperbolic(
            HyperbolicFunction::Sine,
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
    extensions: &SINH_EXTENSIONS,
    integer_capabilities: &SINH_INTEGER_CAPABILITIES,
    integer_audit: None,
    suppress_auto_output: false,
};
