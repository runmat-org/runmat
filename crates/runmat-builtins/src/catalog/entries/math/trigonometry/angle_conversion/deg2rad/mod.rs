mod documentation;

use crate::{
    AngleConversionInferenceRule, BuiltinAcceleratorPolicy, BuiltinAsyncBehavior,
    BuiltinBindingDeclaration, BuiltinCatalogEntry, BuiltinCatalogIdentity, BuiltinCompatibility,
    BuiltinCompletionPolicy, BuiltinContractDeclaration, BuiltinContractMaturity,
    BuiltinDescriptor, BuiltinDistributedPolicy, BuiltinErrorDescriptor,
    BuiltinExtensionDescriptor, BuiltinExtensionMode, BuiltinFusionPolicy, BuiltinInferenceRule,
    BuiltinIntegerBackendRule, BuiltinIntegerCapabilityDescriptor, BuiltinIntegerComputationDomain,
    BuiltinIntegerInputAvailability, BuiltinIntegerInputCapability, BuiltinIntegerOutputClassRule,
    BuiltinIntegerOverflowRule, BuiltinIntegerOverloadKind, BuiltinIntegerScalarDoubleRule,
    BuiltinLinkContract, BuiltinLinkPolicy, BuiltinOutputMode, BuiltinParamArity,
    BuiltinParamDescriptor, BuiltinParamType, BuiltinPlacementContract, BuiltinPortability,
    BuiltinPurity, BuiltinReachability, BuiltinResidencyPolicy, BuiltinSemanticKind,
    BuiltinSignatureDescriptor, MathInferenceRule, ALL_INTEGER_CLASSES,
};
use documentation::DEG2RAD_DOCUMENTATION;
use runmat_types::{EffectKind, ExecutionStackRequirement};

const OUTPUTS: [BuiltinParamDescriptor; 1] = [BuiltinParamDescriptor {
    name: "R",
    ty: BuiltinParamType::NumericArray,
    arity: BuiltinParamArity::Required,
    default: None,
    description: "Angles in radians with the input shape and floating-point precision.",
}];
const INPUTS: [BuiltinParamDescriptor; 1] = [BuiltinParamDescriptor {
    name: "D",
    ty: BuiltinParamType::Any,
    arity: BuiltinParamArity::Required,
    default: None,
    description: "Single or double real or complex angles in degrees; integer and logical values are RunMat extensions.",
}];
const SIGNATURES: [BuiltinSignatureDescriptor; 1] = [BuiltinSignatureDescriptor {
    label: "R = deg2rad(D)",
    inputs: &INPUTS,
    outputs: &OUTPUTS,
}];

pub const DEG2RAD_ERROR_INVALID_INPUT: BuiltinErrorDescriptor = BuiltinErrorDescriptor {
    code: "RM.DEG2RAD.INVALID_INPUT",
    identifier: Some("RunMat:deg2rad:InvalidInput"),
    when: "Input is not a supported real or complex numeric value.",
    message: "deg2rad: invalid input",
};
pub const DEG2RAD_ERROR_INTERNAL: BuiltinErrorDescriptor = BuiltinErrorDescriptor {
    code: "RM.DEG2RAD.INTERNAL",
    identifier: Some("RunMat:deg2rad:Internal"),
    when: "Internal gather, conversion, allocation, or residency restoration fails.",
    message: "deg2rad: internal error",
};
pub const DEG2RAD_ERROR_TOO_MANY_OUTPUTS: BuiltinErrorDescriptor = BuiltinErrorDescriptor {
    code: "RM.DEG2RAD.TOO_MANY_OUTPUTS",
    identifier: Some("RunMat:deg2rad:TooManyOutputs"),
    when: "More than one output is requested.",
    message: "deg2rad: too many output arguments",
};
const ERRORS: [BuiltinErrorDescriptor; 3] = [
    DEG2RAD_ERROR_INVALID_INPUT,
    DEG2RAD_ERROR_INTERNAL,
    DEG2RAD_ERROR_TOO_MANY_OUTPUTS,
];
pub const DEG2RAD_DESCRIPTOR: BuiltinDescriptor = BuiltinDescriptor {
    signatures: &SIGNATURES,
    output_mode: BuiltinOutputMode::Fixed,
    completion_policy: BuiltinCompletionPolicy::Public,
    errors: &ERRORS,
};

pub const DEG2RAD_INTEGER_INPUT_EXTENSION: BuiltinExtensionDescriptor =
    BuiltinExtensionDescriptor {
        id: "deg2rad-integer-input",
        mode: BuiltinExtensionMode::RunMatOnly,
        description: "deg2rad with fixed-width integer input is a RunMat extension",
        error_identifier: Some("RunMat:compatibility:Deg2radIntegerInputExtension"),
    };
pub const DEG2RAD_LOGICAL_INPUT_EXTENSION: BuiltinExtensionDescriptor =
    BuiltinExtensionDescriptor {
        id: "deg2rad-logical-input",
        mode: BuiltinExtensionMode::RunMatOnly,
        description: "deg2rad with logical input is a RunMat extension",
        error_identifier: Some("RunMat:compatibility:Deg2radLogicalInputExtension"),
    };
const EXTENSIONS: [BuiltinExtensionDescriptor; 2] = [
    DEG2RAD_INTEGER_INPUT_EXTENSION,
    DEG2RAD_LOGICAL_INPUT_EXTENSION,
];
const INTEGER_INPUTS: [BuiltinIntegerInputCapability; 1] = [BuiltinIntegerInputCapability {
    name: "D",
    classes: &ALL_INTEGER_CLASSES,
    availability: BuiltinIntegerInputAvailability::RunMatOnly,
    scalar_double: BuiltinIntegerScalarDoubleRule::NotApplicable,
    notes: "Every value must be exactly representable at the binary64 conversion boundary.",
}];
pub const DEG2RAD_INTEGER_CAPABILITIES: [BuiltinIntegerCapabilityDescriptor; 1] =
    [BuiltinIntegerCapabilityDescriptor {
        form: "R = deg2rad(integer_D)",
        inputs: &INTEGER_INPUTS,
        computation_domain: BuiltinIntegerComputationDomain::FloatingPoint,
        output_class: BuiltinIntegerOutputClassRule::Double,
        overflow: BuiltinIntegerOverflowRule::Error,
        backend: BuiltinIntegerBackendRule::GatherFallback,
        overload: BuiltinIntegerOverloadKind::ElementwiseShapePreserving,
        notes: "Native integer storage is validated before explicit binary64 conversion; resident output is restored through the input's owner.",
    }];

const BINDINGS: [BuiltinBindingDeclaration; 1] = crate::REQUIRED_DEFAULT_BINDING;
const EFFECTS: [EffectKind; 2] = [EffectKind::MaySuspend, EffectKind::MayThrow];
pub const DEG2RAD_CATALOG_ENTRY: BuiltinCatalogEntry = BuiltinCatalogEntry {
    provenance: crate::BuiltinCatalogProvenance::new(file!(), module_path!()),
    identity: BuiltinCatalogIdentity { name: "deg2rad" },
    category: "math/trigonometry",
    documentation: DEG2RAD_DOCUMENTATION,
    descriptor: &DEG2RAD_DESCRIPTOR,
    contract: BuiltinContractDeclaration {
        maturity: BuiltinContractMaturity::Complete,
        inference_rule: BuiltinInferenceRule::Math(MathInferenceRule::AngleConversion(
            AngleConversionInferenceRule::DegreesToRadians,
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
    extensions: &EXTENSIONS,
    integer_capabilities: &DEG2RAD_INTEGER_CAPABILITIES,
    integer_audit: None,
    suppress_auto_output: false,
};
