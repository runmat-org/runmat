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

use super::documentation::COSPI_DOCUMENTATION;

const COSPI_OUTPUTS: [BuiltinParamDescriptor; 1] = [BuiltinParamDescriptor {
    name: "Y",
    ty: BuiltinParamType::NumericArray,
    arity: BuiltinParamArity::Required,
    default: None,
    description: "Elementwise cos(X*pi) result with exact integer and half-integer handling.",
}];
const COSPI_INPUTS: [BuiltinParamDescriptor; 1] = [BuiltinParamDescriptor {
    name: "X",
    ty: BuiltinParamType::Any,
    arity: BuiltinParamArity::Required,
    default: None,
    description: "Single or double real/complex input; integer, logical, and character forms are RunMat-only extensions.",
}];
const COSPI_SIGNATURES: [BuiltinSignatureDescriptor; 1] = [BuiltinSignatureDescriptor {
    label: "Y = cospi(X)",
    inputs: &COSPI_INPUTS,
    outputs: &COSPI_OUTPUTS,
}];

pub const COSPI_ERROR_INVALID_INPUT: BuiltinErrorDescriptor = BuiltinErrorDescriptor {
    code: "RM.COSPI.INVALID_INPUT",
    identifier: Some("RunMat:cospi:InvalidInput"),
    when: "Input cannot be interpreted as supported numeric, logical, character, or floating-complex data.",
    message: "cospi: invalid input",
};
pub const COSPI_ERROR_INTERNAL: BuiltinErrorDescriptor = BuiltinErrorDescriptor {
    code: "RM.COSPI.INTERNAL",
    identifier: Some("RunMat:cospi:Internal"),
    when: "Internal gather, conversion, allocation, or provider restoration failed.",
    message: "cospi: internal error",
};
const COSPI_ERRORS: [BuiltinErrorDescriptor; 2] = [COSPI_ERROR_INVALID_INPUT, COSPI_ERROR_INTERNAL];
pub const COSPI_DESCRIPTOR: BuiltinDescriptor = BuiltinDescriptor {
    signatures: &COSPI_SIGNATURES,
    output_mode: BuiltinOutputMode::Fixed,
    completion_policy: BuiltinCompletionPolicy::Public,
    errors: &COSPI_ERRORS,
};

pub const COSPI_INTEGER_INPUT_EXTENSION: BuiltinExtensionDescriptor = BuiltinExtensionDescriptor {
    id: "cospi-integer-input",
    mode: BuiltinExtensionMode::RunMatOnly,
    description: "cospi with typed-integer input is a RunMat extension",
    error_identifier: Some("RunMat:compatibility:CospiIntegerInputExtension"),
};
pub const COSPI_LOGICAL_INPUT_EXTENSION: BuiltinExtensionDescriptor = BuiltinExtensionDescriptor {
    id: "cospi-logical-input",
    mode: BuiltinExtensionMode::RunMatOnly,
    description: "cospi with logical input is a RunMat extension",
    error_identifier: Some("RunMat:compatibility:CospiLogicalInputExtension"),
};
pub const COSPI_CHARACTER_INPUT_EXTENSION: BuiltinExtensionDescriptor =
    BuiltinExtensionDescriptor {
        id: "cospi-character-input",
        mode: BuiltinExtensionMode::RunMatOnly,
        description: "cospi with character input is a RunMat extension",
        error_identifier: Some("RunMat:compatibility:CospiCharacterInputExtension"),
    };
pub const COSPI_EXTENSIONS: [BuiltinExtensionDescriptor; 3] = [
    COSPI_INTEGER_INPUT_EXTENSION,
    COSPI_LOGICAL_INPUT_EXTENSION,
    COSPI_CHARACTER_INPUT_EXTENSION,
];

const COSPI_INTEGER_INPUT: [BuiltinIntegerInputCapability; 1] = [BuiltinIntegerInputCapability {
    name: "X",
    classes: &ALL_INTEGER_CLASSES,
    availability: BuiltinIntegerInputAvailability::RunMatOnly,
    scalar_double: BuiltinIntegerScalarDoubleRule::NotApplicable,
    notes: "Native integer parity determines the exact result without conversion through binary64.",
}];
pub const COSPI_INTEGER_CAPABILITIES: [BuiltinIntegerCapabilityDescriptor; 1] =
    [BuiltinIntegerCapabilityDescriptor {
        form: "Y = cospi(integer_X)",
        inputs: &COSPI_INTEGER_INPUT,
        computation_domain: BuiltinIntegerComputationDomain::ExactInteger,
        output_class: BuiltinIntegerOutputClassRule::Double,
        overflow: BuiltinIntegerOverflowRule::NotApplicable,
        backend: BuiltinIntegerBackendRule::GatherFallback,
        overload: BuiltinIntegerOverloadKind::ElementwiseShapePreserving,
        notes: "RunMat mode computes exact positive or negative one directly from integer parity, including int64 and uint64 values above flintmax; resident input gathers authoritatively and the result returns to its owner.",
    }];

const COSPI_BINDINGS: [BuiltinBindingDeclaration; 1] = crate::REQUIRED_DEFAULT_BINDING;
const COSPI_EFFECTS: [EffectKind; 1] = [EffectKind::MayThrow];
pub const COSPI_CATALOG_ENTRY: BuiltinCatalogEntry = BuiltinCatalogEntry {
    identity: BuiltinCatalogIdentity { name: "cospi" },
    category: "math/trigonometry",
    documentation: COSPI_DOCUMENTATION,
    descriptor: &COSPI_DESCRIPTOR,
    contract: BuiltinContractDeclaration {
        maturity: BuiltinContractMaturity::Complete,
        inference_rule: BuiltinInferenceRule::Math(MathInferenceRule::PiScaledTrigonometric(
            PiScaledTrigonometricFunction::Cos,
        )),
        compatibility: BuiltinCompatibility::Matlab,
        async_behavior: BuiltinAsyncBehavior::NeverSuspends,
        purity: BuiltinPurity::Pure,
        semantic_kind: BuiltinSemanticKind::General,
        workspace_effect: None,
        environment_effect: None,
        effects: &COSPI_EFFECTS,
        capabilities: &[],
    },
    placement: BuiltinPlacementContract {
        portability: BuiltinPortability::NativeAndWasm,
        accelerator: BuiltinAcceleratorPolicy::Optional,
        residency: BuiltinResidencyPolicy::PreserveInputs,
        fusion: BuiltinFusionPolicy::Boundary,
        distributed: crate::BuiltinDistributedPolicy::MapUnary,
    },
    link: BuiltinLinkContract {
        reachability: BuiltinReachability::Always,
        policy: BuiltinLinkPolicy::PortableRuntime,
        execution_stack: ExecutionStackRequirement::Any,
        artifact_dependencies: &[],
    },
    bindings: &COSPI_BINDINGS,
    extensions: &COSPI_EXTENSIONS,
    integer_capabilities: &COSPI_INTEGER_CAPABILITIES,
    integer_audit: None,
    suppress_auto_output: false,
};
