use crate::{
    BuiltinAcceleratorPolicy, BuiltinAsyncBehavior, BuiltinBindingAvailability,
    BuiltinBindingDeclaration, BuiltinBindingIdentity, BuiltinCatalogEntry, BuiltinCatalogIdentity,
    BuiltinCompatibility, BuiltinCompletionPolicy, BuiltinContractDeclaration,
    BuiltinContractMaturity, BuiltinDescriptor, BuiltinDocumentation, BuiltinErrorDescriptor,
    BuiltinExtensionDescriptor, BuiltinExtensionMode, BuiltinFusionPolicy, BuiltinInferenceRuleId,
    BuiltinIntegerBackendRule, BuiltinIntegerCapabilityDescriptor, BuiltinIntegerComputationDomain,
    BuiltinIntegerInputAvailability, BuiltinIntegerInputCapability, BuiltinIntegerOutputClassRule,
    BuiltinIntegerOverflowRule, BuiltinIntegerOverloadKind, BuiltinIntegerScalarDoubleRule,
    BuiltinLinkContract, BuiltinLinkPolicy, BuiltinOutputMode, BuiltinParamArity,
    BuiltinParamDescriptor, BuiltinParamType, BuiltinPlacementContract, BuiltinPortability,
    BuiltinPurity, BuiltinReachability, BuiltinResidencyPolicy, BuiltinSemanticKind,
    BuiltinSignatureDescriptor, ALL_INTEGER_CLASSES, SPARSE_INTEGER_EXTENSION,
};
use runmat_types::EffectKind;

const ABS_OUTPUT: [BuiltinParamDescriptor; 1] = [BuiltinParamDescriptor {
    name: "Y",
    ty: BuiltinParamType::NumericArray,
    arity: BuiltinParamArity::Required,
    default: None,
    description: "Absolute value or magnitude.",
}];
const ABS_INPUTS: [BuiltinParamDescriptor; 1] = [BuiltinParamDescriptor {
    name: "X", ty: BuiltinParamType::Any, arity: BuiltinParamArity::Required, default: None,
    description: "Real numeric input or floating complex input; logical and character forms are RunMat-only extensions.",
}];
const ABS_SIGNATURES: [BuiltinSignatureDescriptor; 1] = [BuiltinSignatureDescriptor {
    label: "Y = abs(X)",
    inputs: &ABS_INPUTS,
    outputs: &ABS_OUTPUT,
}];
pub const ABS_ERROR_INVALID_INPUT: BuiltinErrorDescriptor = BuiltinErrorDescriptor {
    code: "RM.ABS.INVALID_INPUT",
    identifier: Some("RunMat:abs:InvalidInput"),
    when: "Input is not supported real numeric, floating complex, or declared extension data.",
    message: "abs: invalid input",
};
pub const ABS_ERROR_INTERNAL: BuiltinErrorDescriptor = BuiltinErrorDescriptor {
    code: "RM.ABS.INTERNAL",
    identifier: Some("RunMat:abs:Internal"),
    when: "Internal tensor conversion/allocation/provider interaction failed.",
    message: "abs: internal error",
};
pub const ABS_ERROR_TOO_MANY_OUTPUTS: BuiltinErrorDescriptor = BuiltinErrorDescriptor {
    code: "RM.ABS.TOO_MANY_OUTPUTS",
    identifier: Some("RunMat:abs:TooManyOutputs"),
    when: "More than one output is requested.",
    message: "abs: too many output arguments",
};
const ABS_ERRORS: [BuiltinErrorDescriptor; 3] = [
    ABS_ERROR_INVALID_INPUT,
    ABS_ERROR_INTERNAL,
    ABS_ERROR_TOO_MANY_OUTPUTS,
];
pub const ABS_LOGICAL_INPUT_EXTENSION: BuiltinExtensionDescriptor = BuiltinExtensionDescriptor {
    id: "abs-logical-input",
    mode: BuiltinExtensionMode::RunMatOnly,
    description: "abs with logical input is a RunMat extension",
    error_identifier: Some("RunMat:compatibility:AbsLogicalInputExtension"),
};
pub const ABS_CHARACTER_INPUT_EXTENSION: BuiltinExtensionDescriptor = BuiltinExtensionDescriptor {
    id: "abs-character-input",
    mode: BuiltinExtensionMode::RunMatOnly,
    description: "abs with character input is a RunMat extension",
    error_identifier: Some("RunMat:compatibility:AbsCharacterInputExtension"),
};
pub const ABS_EXTENSIONS: [BuiltinExtensionDescriptor; 3] = [
    ABS_LOGICAL_INPUT_EXTENSION,
    ABS_CHARACTER_INPUT_EXTENSION,
    SPARSE_INTEGER_EXTENSION,
];
const ABS_INTEGER_INPUT: [BuiltinIntegerInputCapability; 1] = [BuiltinIntegerInputCapability {
    name: "X", classes: &ALL_INTEGER_CLASSES,
    availability: BuiltinIntegerInputAvailability::Documented,
    scalar_double: BuiltinIntegerScalarDoubleRule::NotApplicable,
    notes: "Real full scalars and arrays accept every built-in integer class and preserve size and class.",
}];
const ABS_SPARSE_INTEGER_INPUT: [BuiltinIntegerInputCapability; 1] = [BuiltinIntegerInputCapability {
    name: "X", classes: &ALL_INTEGER_CLASSES,
    availability: BuiltinIntegerInputAvailability::RunMatOnly,
    scalar_double: BuiltinIntegerScalarDoubleRule::NotApplicable,
    notes: "MATLAB sparse numeric values are single or double; RunMat mode additionally preserves exact integer CSC values.",
}];
pub const ABS_INTEGER_CAPABILITIES: [BuiltinIntegerCapabilityDescriptor; 2] = [
    BuiltinIntegerCapabilityDescriptor {
        form: "Y = abs(integer_X)", inputs: &ABS_INTEGER_INPUT,
        computation_domain: BuiltinIntegerComputationDomain::ExactInteger,
        output_class: BuiltinIntegerOutputClassRule::PreserveInput,
        overflow: BuiltinIntegerOverflowRule::Saturate,
        backend: BuiltinIntegerBackendRule::HostAndGpu,
        overload: BuiltinIntegerOverloadKind::ElementwiseShapePreserving,
        notes: "Unsigned values are unchanged; signed values negate exactly and intmin saturates to intmax. Resident input returns to its owning provider.",
    },
    BuiltinIntegerCapabilityDescriptor {
        form: "S = abs(sparse(integer_X))", inputs: &ABS_SPARSE_INTEGER_INPUT,
        computation_domain: BuiltinIntegerComputationDomain::ExactInteger,
        output_class: BuiltinIntegerOutputClassRule::PreserveInput,
        overflow: BuiltinIntegerOverflowRule::Saturate,
        backend: BuiltinIntegerBackendRule::HostOnly,
        overload: BuiltinIntegerOverloadKind::ElementwiseShapePreserving,
        notes: "RunMat-only typed sparse storage retains CSC structure and exact class while applying signed saturation.",
    },
];
pub const ABS_DESCRIPTOR: BuiltinDescriptor = BuiltinDescriptor {
    signatures: &ABS_SIGNATURES,
    output_mode: BuiltinOutputMode::Fixed,
    completion_policy: BuiltinCompletionPolicy::Public,
    errors: &ABS_ERRORS,
};
const ABS_BINDINGS: [BuiltinBindingDeclaration; 1] = [BuiltinBindingDeclaration {
    identity: BuiltinBindingIdentity {
        builtin: BuiltinCatalogIdentity { name: "abs" },
        variant: "default",
    },
    availability: BuiltinBindingAvailability::Required,
}];
const ABS_EFFECTS: [EffectKind; 1] = [EffectKind::MayThrow];
pub const ABS_CATALOG_ENTRY: BuiltinCatalogEntry = BuiltinCatalogEntry {
    identity: BuiltinCatalogIdentity { name: "abs" },
    category: "math/elementwise",
    documentation: BuiltinDocumentation {
        summary: "Absolute value and complex magnitude for scalars and arrays.",
        keywords: &["abs", "absolute value", "complex", "gpu", "magnitude"],
        related: &[],
        introduced: None,
        status: None,
        examples: &[],
    },
    descriptor: &ABS_DESCRIPTOR,
    contract: BuiltinContractDeclaration {
        maturity: BuiltinContractMaturity::Complete,
        inference_rule: BuiltinInferenceRuleId("math.abs"),
        compatibility: BuiltinCompatibility::Matlab,
        async_behavior: BuiltinAsyncBehavior::NeverSuspends,
        purity: BuiltinPurity::Pure,
        semantic_kind: BuiltinSemanticKind::General,
        workspace_effect: None,
        environment_effect: None,
        effects: &ABS_EFFECTS,
        capabilities: &[],
    },
    placement: BuiltinPlacementContract {
        portability: BuiltinPortability::NativeAndWasm,
        accelerator: BuiltinAcceleratorPolicy::Optional,
        residency: BuiltinResidencyPolicy::PreserveInputs,
        fusion: BuiltinFusionPolicy::Candidate,
    },
    link: BuiltinLinkContract {
        reachability: BuiltinReachability::Always,
        policy: BuiltinLinkPolicy::PortableRuntime,
        artifact_dependencies: &[],
    },
    bindings: &ABS_BINDINGS,
    extensions: &ABS_EXTENSIONS,
    integer_capabilities: &ABS_INTEGER_CAPABILITIES,
    integer_audit: None,
    suppress_auto_output: false,
};
