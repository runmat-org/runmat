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
use documentation::RAD2DEG_DOCUMENTATION;
use runmat_types::{EffectKind, ExecutionStackRequirement};

const OUTPUTS: [BuiltinParamDescriptor; 1] = [BuiltinParamDescriptor {
    name: "D",
    ty: BuiltinParamType::NumericArray,
    arity: BuiltinParamArity::Required,
    default: None,
    description: "Angles in degrees with the input shape and floating-point precision.",
}];
const INPUTS: [BuiltinParamDescriptor; 1] = [BuiltinParamDescriptor {
    name: "R",
    ty: BuiltinParamType::Any,
    arity: BuiltinParamArity::Required,
    default: None,
    description: "Single or double real or complex angles in radians; integer and logical values are RunMat extensions.",
}];
const SIGNATURES: [BuiltinSignatureDescriptor; 1] = [BuiltinSignatureDescriptor {
    label: "D = rad2deg(R)",
    inputs: &INPUTS,
    outputs: &OUTPUTS,
}];

pub const RAD2DEG_ERROR_INVALID_INPUT: BuiltinErrorDescriptor = BuiltinErrorDescriptor {
    code: "RM.RAD2DEG.INVALID_INPUT",
    identifier: Some("RunMat:rad2deg:InvalidInput"),
    when: "Input is not a supported real or complex numeric value.",
    message: "rad2deg: invalid input",
};
pub const RAD2DEG_ERROR_INTERNAL: BuiltinErrorDescriptor = BuiltinErrorDescriptor {
    code: "RM.RAD2DEG.INTERNAL",
    identifier: Some("RunMat:rad2deg:Internal"),
    when: "Internal gather, conversion, allocation, or residency restoration fails.",
    message: "rad2deg: internal error",
};
pub const RAD2DEG_ERROR_TOO_MANY_OUTPUTS: BuiltinErrorDescriptor = BuiltinErrorDescriptor {
    code: "RM.RAD2DEG.TOO_MANY_OUTPUTS",
    identifier: Some("RunMat:rad2deg:TooManyOutputs"),
    when: "More than one output is requested.",
    message: "rad2deg: too many output arguments",
};
const ERRORS: [BuiltinErrorDescriptor; 3] = [
    RAD2DEG_ERROR_INVALID_INPUT,
    RAD2DEG_ERROR_INTERNAL,
    RAD2DEG_ERROR_TOO_MANY_OUTPUTS,
];
pub const RAD2DEG_DESCRIPTOR: BuiltinDescriptor = BuiltinDescriptor {
    signatures: &SIGNATURES,
    output_mode: BuiltinOutputMode::Fixed,
    completion_policy: BuiltinCompletionPolicy::Public,
    errors: &ERRORS,
};

pub const RAD2DEG_INTEGER_INPUT_EXTENSION: BuiltinExtensionDescriptor =
    BuiltinExtensionDescriptor {
        id: "rad2deg-integer-input",
        mode: BuiltinExtensionMode::RunMatOnly,
        description: "rad2deg with fixed-width integer input is a RunMat extension",
        error_identifier: Some("RunMat:compatibility:Rad2degIntegerInputExtension"),
    };
pub const RAD2DEG_LOGICAL_INPUT_EXTENSION: BuiltinExtensionDescriptor =
    BuiltinExtensionDescriptor {
        id: "rad2deg-logical-input",
        mode: BuiltinExtensionMode::RunMatOnly,
        description: "rad2deg with logical input is a RunMat extension",
        error_identifier: Some("RunMat:compatibility:Rad2degLogicalInputExtension"),
    };
const EXTENSIONS: [BuiltinExtensionDescriptor; 2] = [
    RAD2DEG_INTEGER_INPUT_EXTENSION,
    RAD2DEG_LOGICAL_INPUT_EXTENSION,
];
const INTEGER_INPUTS: [BuiltinIntegerInputCapability; 1] = [BuiltinIntegerInputCapability {
    name: "R",
    classes: &ALL_INTEGER_CLASSES,
    availability: BuiltinIntegerInputAvailability::RunMatOnly,
    scalar_double: BuiltinIntegerScalarDoubleRule::NotApplicable,
    notes: "Every value must be exactly representable at the binary64 conversion boundary.",
}];
pub const RAD2DEG_INTEGER_CAPABILITIES: [BuiltinIntegerCapabilityDescriptor; 1] =
    [BuiltinIntegerCapabilityDescriptor {
        form: "D = rad2deg(integer_R)",
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
pub const RAD2DEG_CATALOG_ENTRY: BuiltinCatalogEntry = BuiltinCatalogEntry {
    provenance: crate::BuiltinCatalogProvenance::new(file!(), module_path!()),
    identity: BuiltinCatalogIdentity { name: "rad2deg" },
    category: "math/trigonometry",
    documentation: RAD2DEG_DOCUMENTATION,
    descriptor: &RAD2DEG_DESCRIPTOR,
    contract: BuiltinContractDeclaration {
        maturity: BuiltinContractMaturity::Complete,
        inference_rule: BuiltinInferenceRule::Math(MathInferenceRule::AngleConversion(
            AngleConversionInferenceRule::RadiansToDegrees,
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
    integer_capabilities: &RAD2DEG_INTEGER_CAPABILITIES,
    integer_audit: None,
    suppress_auto_output: false,
};
