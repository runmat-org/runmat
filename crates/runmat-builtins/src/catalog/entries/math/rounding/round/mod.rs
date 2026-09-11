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
    BuiltinSignatureDescriptor, MathInferenceRule, ALL_INTEGER_CLASSES,
};
use runmat_types::{EffectKind, ExecutionStackRequirement};

mod documentation;
use documentation::ROUND_DOCUMENTATION;

const OUTPUTS: [BuiltinParamDescriptor; 1] = [BuiltinParamDescriptor {
    name: "Y",
    ty: BuiltinParamType::NumericArray,
    arity: BuiltinParamArity::Required,
    default: None,
    description: "Rounded values with the class and shape determined by X.",
}];
const INPUT_X: [BuiltinParamDescriptor; 1] = [BuiltinParamDescriptor {
    name: "X",
    ty: BuiltinParamType::Any,
    arity: BuiltinParamArity::Required,
    default: None,
    description: "Numeric, logical, character, or complex input values.",
}];
const INPUT_X_N: [BuiltinParamDescriptor; 2] = [
    BuiltinParamDescriptor {
        name: "X",
        ty: BuiltinParamType::Any,
        arity: BuiltinParamArity::Required,
        default: None,
        description: "Numeric, logical, character, or complex input values.",
    },
    BuiltinParamDescriptor {
        name: "N",
        ty: BuiltinParamType::NumericScalar,
        arity: BuiltinParamArity::Required,
        default: None,
        description: "Finite integer number of decimal digits.",
    },
];
const INPUT_X_N_MODE: [BuiltinParamDescriptor; 3] = [
    BuiltinParamDescriptor {
        name: "X",
        ty: BuiltinParamType::Any,
        arity: BuiltinParamArity::Required,
        default: None,
        description: "Numeric, logical, character, or complex input values.",
    },
    BuiltinParamDescriptor {
        name: "N",
        ty: BuiltinParamType::NumericScalar,
        arity: BuiltinParamArity::Required,
        default: None,
        description: "Finite integer number of digits.",
    },
    BuiltinParamDescriptor {
        name: "mode",
        ty: BuiltinParamType::StringScalar,
        arity: BuiltinParamArity::Required,
        default: Some("\"decimals\""),
        description: "Rounding interpretation: \"decimals\" or \"significant\".",
    },
];
const SIGNATURES: [BuiltinSignatureDescriptor; 3] = [
    BuiltinSignatureDescriptor {
        label: "Y = round(X)",
        inputs: &INPUT_X,
        outputs: &OUTPUTS,
    },
    BuiltinSignatureDescriptor {
        label: "Y = round(X, N)",
        inputs: &INPUT_X_N,
        outputs: &OUTPUTS,
    },
    BuiltinSignatureDescriptor {
        label: "Y = round(X, N, mode)",
        inputs: &INPUT_X_N_MODE,
        outputs: &OUTPUTS,
    },
];

pub const ROUND_ERROR_INVALID_INPUT: BuiltinErrorDescriptor = BuiltinErrorDescriptor {
    code: "RM.ROUND.INVALID_INPUT",
    identifier: Some("RunMat:round:InvalidInput"),
    when: "X is unsupported or a typed integer X is used with a multi-input form.",
    message: "round: invalid input",
};
pub const ROUND_ERROR_INVALID_ARGUMENT: BuiltinErrorDescriptor = BuiltinErrorDescriptor {
    code: "RM.ROUND.INVALID_ARGUMENT",
    identifier: Some("RunMat:round:InvalidArgument"),
    when: "The invocation does not contain one to three inputs.",
    message: "round: invalid argument",
};
pub const ROUND_ERROR_INVALID_DIGITS: BuiltinErrorDescriptor = BuiltinErrorDescriptor {
    code: "RM.ROUND.INVALID_DIGITS",
    identifier: Some("RunMat:round:InvalidDigits"),
    when: "N is not a finite integer scalar in range, or is non-positive in significant mode.",
    message: "round: invalid digits argument",
};
pub const ROUND_ERROR_INVALID_MODE: BuiltinErrorDescriptor = BuiltinErrorDescriptor {
    code: "RM.ROUND.INVALID_MODE",
    identifier: Some("RunMat:round:InvalidMode"),
    when: "mode is not a supported scalar text token.",
    message: "round: invalid mode",
};
pub const ROUND_ERROR_TOO_MANY_OUTPUTS: BuiltinErrorDescriptor = BuiltinErrorDescriptor {
    code: "RM.ROUND.TOO_MANY_OUTPUTS",
    identifier: Some("RunMat:round:TooManyOutputs"),
    when: "More than one output is requested.",
    message: "round: too many output arguments",
};
pub const ROUND_ERROR_INTERNAL: BuiltinErrorDescriptor = BuiltinErrorDescriptor {
    code: "RM.ROUND.INTERNAL",
    identifier: Some("RunMat:round:Internal"),
    when: "Internal conversion, allocation, provider execution, or residency restoration fails.",
    message: "round: internal error",
};
const ERRORS: [BuiltinErrorDescriptor; 6] = [
    ROUND_ERROR_INVALID_INPUT,
    ROUND_ERROR_INVALID_ARGUMENT,
    ROUND_ERROR_INVALID_DIGITS,
    ROUND_ERROR_INVALID_MODE,
    ROUND_ERROR_TOO_MANY_OUTPUTS,
    ROUND_ERROR_INTERNAL,
];
pub const ROUND_DESCRIPTOR: BuiltinDescriptor = BuiltinDescriptor {
    signatures: &SIGNATURES,
    output_mode: BuiltinOutputMode::Fixed,
    completion_policy: BuiltinCompletionPolicy::Public,
    errors: &ERRORS,
};

pub const ROUND_TYPED_INTEGER_DIGITS_EXTENSION: BuiltinExtensionDescriptor =
    BuiltinExtensionDescriptor {
        id: "round-typed-integer-digits",
        mode: BuiltinExtensionMode::RunMatOnly,
        description: "round accepts an exact typed-integer N control",
        error_identifier: Some("RunMat:compatibility:RoundTypedIntegerDigitsExtension"),
    };
pub const ROUND_DECIMAL_MODE_ALIAS_EXTENSION: BuiltinExtensionDescriptor =
    BuiltinExtensionDescriptor {
        id: "round-decimal-mode-alias",
        mode: BuiltinExtensionMode::RunMatOnly,
        description: "round accepts \"decimal\" as an alias for \"decimals\"",
        error_identifier: Some("RunMat:compatibility:RoundDecimalModeAliasExtension"),
    };
pub const ROUND_EXTENSIONS: [BuiltinExtensionDescriptor; 2] = [
    ROUND_TYPED_INTEGER_DIGITS_EXTENSION,
    ROUND_DECIMAL_MODE_ALIAS_EXTENSION,
];

const INTEGER_DATA_INPUTS: [BuiltinIntegerInputCapability; 1] = [BuiltinIntegerInputCapability {
    name: "X",
    classes: &ALL_INTEGER_CLASSES,
    availability: BuiltinIntegerInputAvailability::Documented,
    scalar_double: BuiltinIntegerScalarDoubleRule::NotApplicable,
    notes:
        "The one-input form is an exact class-, shape-, bits-, and residency-preserving identity.",
}];
const INTEGER_DIGITS_INPUTS: [BuiltinIntegerInputCapability; 1] = [BuiltinIntegerInputCapability {
    name: "N",
    classes: &ALL_INTEGER_CLASSES,
    availability: BuiltinIntegerInputAvailability::RunMatOnly,
    scalar_double: BuiltinIntegerScalarDoubleRule::NotApplicable,
    notes: "RunMat mode decodes one exact native integer element and requires it to fit i32.",
}];
pub const ROUND_INTEGER_CAPABILITIES: [BuiltinIntegerCapabilityDescriptor; 2] = [
    BuiltinIntegerCapabilityDescriptor {
        form: "Y = round(integer_X)",
        inputs: &INTEGER_DATA_INPUTS,
        computation_domain: BuiltinIntegerComputationDomain::Structural,
        output_class: BuiltinIntegerOutputClassRule::PreserveInput,
        overflow: BuiltinIntegerOverflowRule::NotApplicable,
        backend: BuiltinIntegerBackendRule::HostAndGpu,
        overload: BuiltinIntegerOverloadKind::ElementwiseShapePreserving,
        notes: "Host storage and resident handles are returned unchanged; multi-input integer-X forms are rejected.",
    },
    BuiltinIntegerCapabilityDescriptor {
        form: "Y = round(floating_X, integer_N [, mode])",
        inputs: &INTEGER_DIGITS_INPUTS,
        computation_domain: BuiltinIntegerComputationDomain::Structural,
        output_class: BuiltinIntegerOutputClassRule::FunctionSpecific,
        overflow: BuiltinIntegerOverflowRule::Error,
        backend: BuiltinIntegerBackendRule::HostAndGpu,
        overload: BuiltinIntegerOverloadKind::ScalarOnly,
        notes: "The control value is never routed through binary64; output follows floating X.",
    },
];

const BINDINGS: [BuiltinBindingDeclaration; 1] = crate::REQUIRED_DEFAULT_BINDING;
const EFFECTS: [EffectKind; 2] = [EffectKind::MaySuspend, EffectKind::MayThrow];
pub const ROUND_CATALOG_ENTRY: BuiltinCatalogEntry = BuiltinCatalogEntry {
    provenance: crate::BuiltinCatalogProvenance::new(file!(), module_path!()),
    identity: BuiltinCatalogIdentity { name: "round" },
    category: "math/rounding",
    documentation: ROUND_DOCUMENTATION,
    descriptor: &ROUND_DESCRIPTOR,
    contract: BuiltinContractDeclaration {
        maturity: BuiltinContractMaturity::Complete,
        inference_rule: BuiltinInferenceRule::Math(MathInferenceRule::Round),
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
    extensions: &ROUND_EXTENSIONS,
    integer_capabilities: &ROUND_INTEGER_CAPABILITIES,
    integer_audit: None,
    suppress_auto_output: false,
};
