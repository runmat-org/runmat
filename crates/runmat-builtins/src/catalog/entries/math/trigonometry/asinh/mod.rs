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

use documentation::ASINH_DOCUMENTATION;

const OUTPUTS: [BuiltinParamDescriptor; 1] = [BuiltinParamDescriptor {
    name: "Y",
    ty: BuiltinParamType::NumericArray,
    arity: BuiltinParamArity::Required,
    default: None,
    description: "Elementwise principal inverse-hyperbolic-sine result.",
}];
const INPUTS: [BuiltinParamDescriptor; 1] = [BuiltinParamDescriptor {
    name: "X",
    ty: BuiltinParamType::Any,
    arity: BuiltinParamArity::Required,
    default: None,
    description: "Single or double real/complex input; integer, logical, and character forms are RunMat-only extensions.",
}];
const SIGNATURES: [BuiltinSignatureDescriptor; 1] = [BuiltinSignatureDescriptor {
    label: "Y = asinh(X)",
    inputs: &INPUTS,
    outputs: &OUTPUTS,
}];

pub const ASINH_ERROR_INVALID_INPUT: BuiltinErrorDescriptor = BuiltinErrorDescriptor {
    code: "RM.ASINH.INVALID_INPUT",
    identifier: Some("RunMat:asinh:InvalidInput"),
    when: "Input is not a supported numeric, logical, or character value.",
    message: "asinh: invalid input",
};
pub const ASINH_ERROR_INTERNAL: BuiltinErrorDescriptor = BuiltinErrorDescriptor {
    code: "RM.ASINH.INTERNAL",
    identifier: Some("RunMat:asinh:Internal"),
    when: "Internal gather, conversion, allocation, provider execution, or restoration fails.",
    message: "asinh: internal error",
};
pub const ASINH_ERROR_TOO_MANY_OUTPUTS: BuiltinErrorDescriptor = BuiltinErrorDescriptor {
    code: "RM.ASINH.TOO_MANY_OUTPUTS",
    identifier: Some("RunMat:asinh:TooManyOutputs"),
    when: "More than one output is requested.",
    message: "asinh: too many output arguments",
};
const ERRORS: [BuiltinErrorDescriptor; 3] = [
    ASINH_ERROR_INVALID_INPUT,
    ASINH_ERROR_INTERNAL,
    ASINH_ERROR_TOO_MANY_OUTPUTS,
];
pub const ASINH_DESCRIPTOR: BuiltinDescriptor = BuiltinDescriptor {
    signatures: &SIGNATURES,
    output_mode: BuiltinOutputMode::Fixed,
    completion_policy: BuiltinCompletionPolicy::Public,
    errors: &ERRORS,
};

pub const ASINH_INTEGER_INPUT_EXTENSION: BuiltinExtensionDescriptor = BuiltinExtensionDescriptor {
    id: "asinh-integer-input",
    mode: BuiltinExtensionMode::RunMatOnly,
    description: "asinh with typed-integer input is a RunMat extension",
    error_identifier: Some("RunMat:compatibility:AsinhIntegerInputExtension"),
};
pub const ASINH_LOGICAL_INPUT_EXTENSION: BuiltinExtensionDescriptor = BuiltinExtensionDescriptor {
    id: "asinh-logical-input",
    mode: BuiltinExtensionMode::RunMatOnly,
    description: "asinh with logical input is a RunMat extension",
    error_identifier: Some("RunMat:compatibility:AsinhLogicalInputExtension"),
};
pub const ASINH_CHARACTER_INPUT_EXTENSION: BuiltinExtensionDescriptor =
    BuiltinExtensionDescriptor {
        id: "asinh-character-input",
        mode: BuiltinExtensionMode::RunMatOnly,
        description: "asinh with character input is a RunMat extension",
        error_identifier: Some("RunMat:compatibility:AsinhCharacterInputExtension"),
    };
pub const ASINH_EXTENSIONS: [BuiltinExtensionDescriptor; 3] = [
    ASINH_INTEGER_INPUT_EXTENSION,
    ASINH_LOGICAL_INPUT_EXTENSION,
    ASINH_CHARACTER_INPUT_EXTENSION,
];

const INTEGER_INPUTS: [BuiltinIntegerInputCapability; 1] = [BuiltinIntegerInputCapability {
    name: "X",
    classes: &ALL_INTEGER_CLASSES,
    availability: BuiltinIntegerInputAvailability::RunMatOnly,
    scalar_double: BuiltinIntegerScalarDoubleRule::NotApplicable,
    notes: "All eight real integer classes enter the declared binary64 inverse-hyperbolic-sine computation boundary.",
}];
pub const ASINH_INTEGER_CAPABILITIES: [BuiltinIntegerCapabilityDescriptor; 1] =
    [BuiltinIntegerCapabilityDescriptor {
        form: "Y = asinh(integer_X)",
        inputs: &INTEGER_INPUTS,
        computation_domain: BuiltinIntegerComputationDomain::FloatingPoint,
        output_class: BuiltinIntegerOutputClassRule::Double,
        overflow: BuiltinIntegerOverflowRule::NotApplicable,
        backend: BuiltinIntegerBackendRule::GatherFallback,
        overload: BuiltinIntegerOverloadKind::ElementwiseShapePreserving,
        notes: "Authoritative integer storage converts explicitly at the binary64 algorithm boundary. Resident values gather and the real double result restores through their owning provider.",
    }];

const BINDINGS: [BuiltinBindingDeclaration; 1] = crate::REQUIRED_DEFAULT_BINDING;
const EFFECTS: [EffectKind; 2] = [EffectKind::MaySuspend, EffectKind::MayThrow];
pub const ASINH_CATALOG_ENTRY: BuiltinCatalogEntry = BuiltinCatalogEntry {
    provenance: crate::BuiltinCatalogProvenance::new(file!(), module_path!()),
    identity: BuiltinCatalogIdentity { name: "asinh" },
    category: "math/trigonometry",
    documentation: ASINH_DOCUMENTATION,
    descriptor: &ASINH_DESCRIPTOR,
    contract: BuiltinContractDeclaration {
        maturity: BuiltinContractMaturity::Complete,
        inference_rule: BuiltinInferenceRule::Math(MathInferenceRule::InverseHyperbolic(
            InverseHyperbolicFunction::Sine,
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
    extensions: &ASINH_EXTENSIONS,
    integer_capabilities: &ASINH_INTEGER_CAPABILITIES,
    integer_audit: None,
    suppress_auto_output: false,
};

pub(super) const ENTRIES: &[&BuiltinCatalogEntry] = &[&ASINH_CATALOG_ENTRY];
