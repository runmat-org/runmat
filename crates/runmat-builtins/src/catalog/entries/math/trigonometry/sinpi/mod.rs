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
    BuiltinSignatureDescriptor, MathInferenceRule, PiScaledTrigonometricFunction,
    ALL_INTEGER_CLASSES,
};
use runmat_types::{EffectKind, ExecutionStackRequirement};

mod documentation;

use documentation::SINPI_DOCUMENTATION;

const SINPI_OUTPUTS: [BuiltinParamDescriptor; 1] = [BuiltinParamDescriptor {
    name: "Y",
    ty: BuiltinParamType::NumericArray,
    arity: BuiltinParamArity::Required,
    default: None,
    description: "Elementwise sin(X*pi) result with exact integer and half-integer handling.",
}];
const SINPI_INPUTS: [BuiltinParamDescriptor; 1] = [BuiltinParamDescriptor {
    name: "X",
    ty: BuiltinParamType::Any,
    arity: BuiltinParamArity::Required,
    default: None,
    description: "Single or double real/complex input; integer, logical, and character forms are RunMat-only extensions.",
}];
const SINPI_SIGNATURES: [BuiltinSignatureDescriptor; 1] = [BuiltinSignatureDescriptor {
    label: "Y = sinpi(X)",
    inputs: &SINPI_INPUTS,
    outputs: &SINPI_OUTPUTS,
}];

pub const SINPI_ERROR_INVALID_INPUT: BuiltinErrorDescriptor = BuiltinErrorDescriptor {
    code: "RM.SINPI.INVALID_INPUT",
    identifier: Some("RunMat:sinpi:InvalidInput"),
    when: "Input cannot be interpreted as supported numeric, logical, character, or floating-complex data.",
    message: "sinpi: invalid input",
};
pub const SINPI_ERROR_INTERNAL: BuiltinErrorDescriptor = BuiltinErrorDescriptor {
    code: "RM.SINPI.INTERNAL",
    identifier: Some("RunMat:sinpi:Internal"),
    when: "Internal gather, conversion, or allocation failed.",
    message: "sinpi: internal error",
};
const SINPI_ERRORS: [BuiltinErrorDescriptor; 2] = [SINPI_ERROR_INVALID_INPUT, SINPI_ERROR_INTERNAL];
pub const SINPI_DESCRIPTOR: BuiltinDescriptor = BuiltinDescriptor {
    signatures: &SINPI_SIGNATURES,
    output_mode: BuiltinOutputMode::Fixed,
    completion_policy: BuiltinCompletionPolicy::Public,
    errors: &SINPI_ERRORS,
};

pub const SINPI_INTEGER_INPUT_EXTENSION: BuiltinExtensionDescriptor = BuiltinExtensionDescriptor {
    id: "sinpi-integer-input",
    mode: BuiltinExtensionMode::RunMatOnly,
    description: "sinpi with typed-integer input is a RunMat extension",
    error_identifier: Some("RunMat:compatibility:SinpiIntegerInputExtension"),
};
pub const SINPI_LOGICAL_INPUT_EXTENSION: BuiltinExtensionDescriptor = BuiltinExtensionDescriptor {
    id: "sinpi-logical-input",
    mode: BuiltinExtensionMode::RunMatOnly,
    description: "sinpi with logical input is a RunMat extension",
    error_identifier: Some("RunMat:compatibility:SinpiLogicalInputExtension"),
};
pub const SINPI_CHARACTER_INPUT_EXTENSION: BuiltinExtensionDescriptor =
    BuiltinExtensionDescriptor {
        id: "sinpi-character-input",
        mode: BuiltinExtensionMode::RunMatOnly,
        description: "sinpi with character input is a RunMat extension",
        error_identifier: Some("RunMat:compatibility:SinpiCharacterInputExtension"),
    };
pub const SINPI_EXTENSIONS: [BuiltinExtensionDescriptor; 3] = [
    SINPI_INTEGER_INPUT_EXTENSION,
    SINPI_LOGICAL_INPUT_EXTENSION,
    SINPI_CHARACTER_INPUT_EXTENSION,
];

const SINPI_INTEGER_INPUT: [BuiltinIntegerInputCapability; 1] =
    [BuiltinIntegerInputCapability {
        name: "X",
        classes: &ALL_INTEGER_CLASSES,
        availability: BuiltinIntegerInputAvailability::RunMatOnly,
        scalar_double: BuiltinIntegerScalarDoubleRule::NotApplicable,
        notes: "The exact identity sinpi(n)=0 admits every full-width integer value without conversion through binary64.",
    }];
pub const SINPI_INTEGER_CAPABILITIES: [BuiltinIntegerCapabilityDescriptor; 1] =
    [BuiltinIntegerCapabilityDescriptor {
        form: "Y = sinpi(integer_X)",
        inputs: &SINPI_INTEGER_INPUT,
        computation_domain: BuiltinIntegerComputationDomain::FunctionSpecific,
        output_class: BuiltinIntegerOutputClassRule::Double,
        overflow: BuiltinIntegerOverflowRule::NotApplicable,
        backend: BuiltinIntegerBackendRule::GatherFallback,
        overload: BuiltinIntegerOverloadKind::ElementwiseShapePreserving,
        notes: "RunMat mode constructs exact double zeros from integer class and shape, including int64 and uint64 values above flintmax; resident input gathers authoritatively.",
    }];

const SINPI_BINDINGS: [BuiltinBindingDeclaration; 1] = crate::REQUIRED_DEFAULT_BINDING;
const SINPI_EFFECTS: [EffectKind; 1] = [EffectKind::MayThrow];
pub const SINPI_CATALOG_ENTRY: BuiltinCatalogEntry = BuiltinCatalogEntry {
    identity: BuiltinCatalogIdentity { name: "sinpi" },
    category: "math/trigonometry",
    documentation: SINPI_DOCUMENTATION,
    descriptor: &SINPI_DESCRIPTOR,
    contract: BuiltinContractDeclaration {
        maturity: BuiltinContractMaturity::Complete,
        inference_rule: BuiltinInferenceRule::Math(MathInferenceRule::PiScaledTrigonometric(
            PiScaledTrigonometricFunction::Sin,
        )),
        compatibility: BuiltinCompatibility::Matlab,
        async_behavior: BuiltinAsyncBehavior::NeverSuspends,
        purity: BuiltinPurity::Pure,
        semantic_kind: BuiltinSemanticKind::General,
        workspace_effect: None,
        environment_effect: None,
        effects: &SINPI_EFFECTS,
        capabilities: &[],
    },
    placement: BuiltinPlacementContract {
        portability: BuiltinPortability::NativeAndWasm,
        accelerator: BuiltinAcceleratorPolicy::Optional,
        residency: BuiltinResidencyPolicy::GatherToHost,
        fusion: BuiltinFusionPolicy::Boundary,
        distributed: crate::BuiltinDistributedPolicy::MapUnary,
    },
    link: BuiltinLinkContract {
        reachability: BuiltinReachability::Always,
        policy: BuiltinLinkPolicy::PortableRuntime,
        execution_stack: ExecutionStackRequirement::Any,
        artifact_dependencies: &[],
    },
    bindings: &SINPI_BINDINGS,
    extensions: &SINPI_EXTENSIONS,
    integer_capabilities: &SINPI_INTEGER_CAPABILITIES,
    integer_audit: None,
    suppress_auto_output: false,
};
