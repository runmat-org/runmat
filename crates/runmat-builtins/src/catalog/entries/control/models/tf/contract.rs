use crate::*;

pub const INTEGER_COEFFICIENT_EXTENSION: BuiltinExtensionDescriptor = BuiltinExtensionDescriptor {
    id: "tf-integer-coefficients",
    mode: BuiltinExtensionMode::RunMatOnly,
    description: "tf with typed-integer coefficients is a RunMat extension",
    error_identifier: Some("RunMat:compatibility:TfIntegerCoefficientsExtension"),
};
pub const LOGICAL_COEFFICIENT_EXTENSION: BuiltinExtensionDescriptor = BuiltinExtensionDescriptor {
    id: "tf-logical-coefficients",
    mode: BuiltinExtensionMode::RunMatOnly,
    description: "tf with logical coefficients is a RunMat extension",
    error_identifier: Some("RunMat:compatibility:TfLogicalCoefficientsExtension"),
};
pub const INTEGER_SAMPLE_TIME_EXTENSION: BuiltinExtensionDescriptor = BuiltinExtensionDescriptor {
    id: "tf-integer-sample-time",
    mode: BuiltinExtensionMode::RunMatOnly,
    description: "tf with a typed-integer sample time is a RunMat extension",
    error_identifier: Some("RunMat:compatibility:TfIntegerSampleTimeExtension"),
};
pub const VARIABLE_SHORTHAND_EXTENSION: BuiltinExtensionDescriptor = BuiltinExtensionDescriptor {
    id: "tf-variable-shorthand-alias",
    mode: BuiltinExtensionMode::RunMatOnly,
    description: "tf variable shorthand other than documented s and z forms is a RunMat extension",
    error_identifier: Some("RunMat:compatibility:TfVariableShorthandExtension"),
};
pub const DEFAULT_DISCRETE_SAMPLE_TIME_EXTENSION: BuiltinExtensionDescriptor =
    BuiltinExtensionDescriptor {
        id: "tf-default-discrete-sample-time",
        mode: BuiltinExtensionMode::RunMatOnly,
        description:
            "tf discrete-variable shorthand without an explicit sample time is a RunMat extension",
        error_identifier: Some("RunMat:compatibility:TfDefaultDiscreteSampleTimeExtension"),
    };
pub const EXPLICIT_GPU_INPUT_EXTENSION: BuiltinExtensionDescriptor = BuiltinExtensionDescriptor {
    id: "tf-explicit-gpu-input",
    mode: BuiltinExtensionMode::RunMatOnly,
    description: "tf with explicit gpuArray input is a RunMat extension",
    error_identifier: Some("RunMat:compatibility:TfExplicitGpuInputExtension"),
};
pub const TF_EXTENSIONS: [BuiltinExtensionDescriptor; 6] = [
    INTEGER_COEFFICIENT_EXTENSION,
    LOGICAL_COEFFICIENT_EXTENSION,
    INTEGER_SAMPLE_TIME_EXTENSION,
    VARIABLE_SHORTHAND_EXTENSION,
    DEFAULT_DISCRETE_SAMPLE_TIME_EXTENSION,
    EXPLICIT_GPU_INPUT_EXTENSION,
];

// These are the constructor gates also reached when another control builtin
// consumes mutable tf model properties. Shorthand and explicit-input gates
// belong only to construction, not to model-property admission.
pub const TF_CONSUMED_PROPERTY_EXTENSIONS: [BuiltinExtensionDescriptor; 3] = [
    INTEGER_COEFFICIENT_EXTENSION,
    LOGICAL_COEFFICIENT_EXTENSION,
    INTEGER_SAMPLE_TIME_EXTENSION,
];

// Mixed tf/ss consumers reach both parsers. Keep this inventory at the model
// boundary so analysis entries can share the exact transitive gate set.
pub const TF_OR_SS_CONSUMED_PROPERTY_EXTENSIONS: [BuiltinExtensionDescriptor; 5] = [
    INTEGER_COEFFICIENT_EXTENSION,
    LOGICAL_COEFFICIENT_EXTENSION,
    INTEGER_SAMPLE_TIME_EXTENSION,
    super::super::ss::SS_INTEGER_MATRIX_EXTENSION,
    super::super::ss::SS_INTEGER_SAMPLE_TIME_EXTENSION,
];

const INTEGER_COEFFICIENT_INPUTS: [BuiltinIntegerInputCapability; 2] = [
    BuiltinIntegerInputCapability {
        name: "numerator",
        classes: &ALL_INTEGER_CLASSES,
        availability: BuiltinIntegerInputAvailability::RunMatOnly,
        scalar_double: BuiltinIntegerScalarDoubleRule::Allowed,
        notes: "The public page describes numeric real or complex coefficient arrays without enumerating typed integer classes. RunMat mode admits them only at a checked binary64 model boundary.",
    },
    BuiltinIntegerInputCapability {
        name: "denominator",
        classes: &ALL_INTEGER_CLASSES,
        availability: BuiltinIntegerInputAvailability::RunMatOnly,
        scalar_double: BuiltinIntegerScalarDoubleRule::Allowed,
        notes: "Typed denominator coefficients are independently read from authoritative storage and must be exactly representable in the model's double coefficient domain.",
    },
];
const INTEGER_SAMPLE_TIME_INPUT: [BuiltinIntegerInputCapability; 1] =
    [BuiltinIntegerInputCapability {
        name: "Ts",
        classes: &ALL_INTEGER_CLASSES,
        availability: BuiltinIntegerInputAvailability::RunMatOnly,
        scalar_double: BuiltinIntegerScalarDoubleRule::Allowed,
        notes: "Typed sample time is gated independently and crosses one exact binary64 metadata boundary; documented double values include positive periods and -1 for unspecified discrete time.",
    }];
pub const TF_INTEGER_CAPABILITIES: [BuiltinIntegerCapabilityDescriptor; 2] = [
    BuiltinIntegerCapabilityDescriptor {
        form: "sys = tf(integer_numerator, integer_denominator, ...)",
        inputs: &INTEGER_COEFFICIENT_INPUTS,
        computation_domain: BuiltinIntegerComputationDomain::FloatingPoint,
        output_class: BuiltinIntegerOutputClassRule::Double,
        overflow: BuiltinIntegerOverflowRule::Error,
        backend: BuiltinIntegerBackendRule::GatherFallback,
        overload: BuiltinIntegerOverloadKind::Multiple,
        notes: "Each integer coefficient role is gated before provider access, checked for exact binary64 conversion, and stored as host double or complex-double model metadata.",
    },
    BuiltinIntegerCapabilityDescriptor {
        form: "sys = tf(..., integer_Ts)",
        inputs: &INTEGER_SAMPLE_TIME_INPUT,
        computation_domain: BuiltinIntegerComputationDomain::FloatingPoint,
        output_class: BuiltinIntegerOutputClassRule::FunctionSpecific,
        overflow: BuiltinIntegerOverflowRule::Error,
        backend: BuiltinIntegerBackendRule::GatherFallback,
        overload: BuiltinIntegerOverloadKind::ScalarOnly,
        notes: "Typed Ts is checked before conversion and does not determine coefficient or object class.",
    },
];

const TF_OUTPUT_SYS: [BuiltinParamDescriptor; 1] = [BuiltinParamDescriptor {
    name: "sys",
    ty: BuiltinParamType::Any,
    arity: BuiltinParamArity::Required,
    default: None,
    description: "SISO transfer-function object.",
}];
const TF_PARAM_NUMERATOR: BuiltinParamDescriptor = BuiltinParamDescriptor {
    name: "numerator",
    ty: BuiltinParamType::Any,
    arity: BuiltinParamArity::Required,
    default: None,
    description: "Numerator coefficient vector.",
};
const TF_PARAM_DENOMINATOR: BuiltinParamDescriptor = BuiltinParamDescriptor {
    name: "denominator",
    ty: BuiltinParamType::Any,
    arity: BuiltinParamArity::Required,
    default: None,
    description: "Denominator coefficient vector.",
};
const TF_PARAM_VARIABLE_SYMBOL: BuiltinParamDescriptor = BuiltinParamDescriptor {
    name: "variable",
    ty: BuiltinParamType::StringScalar,
    arity: BuiltinParamArity::Required,
    default: None,
    description: "Transfer-function indeterminate ('s', 'p', 'z', 'q', 'z^-1', or 'q^-1').",
};
const TF_PARAM_TS: BuiltinParamDescriptor = BuiltinParamDescriptor {
    name: "Ts",
    ty: BuiltinParamType::NumericScalar,
    arity: BuiltinParamArity::Optional,
    default: Some("0.0"),
    description: "Sample time (0 for continuous-time model).",
};
const TF_INPUTS_VARIABLE: [BuiltinParamDescriptor; 1] = [TF_PARAM_VARIABLE_SYMBOL];
const TF_INPUTS_VARIABLE_TS: [BuiltinParamDescriptor; 2] = [TF_PARAM_VARIABLE_SYMBOL, TF_PARAM_TS];
const TF_INPUTS_NUM_DEN: [BuiltinParamDescriptor; 2] = [TF_PARAM_NUMERATOR, TF_PARAM_DENOMINATOR];
const TF_INPUTS_NUM_DEN_TS: [BuiltinParamDescriptor; 3] =
    [TF_PARAM_NUMERATOR, TF_PARAM_DENOMINATOR, TF_PARAM_TS];
const TF_INPUTS_NUM_DEN_NAMEVALUE: [BuiltinParamDescriptor; 4] = [
    TF_PARAM_NUMERATOR,
    TF_PARAM_DENOMINATOR,
    BuiltinParamDescriptor {
        name: "name",
        ty: BuiltinParamType::StringScalar,
        arity: BuiltinParamArity::Variadic,
        default: None,
        description: "Option name ('Variable' or 'Ts').",
    },
    BuiltinParamDescriptor {
        name: "value",
        ty: BuiltinParamType::Any,
        arity: BuiltinParamArity::Variadic,
        default: None,
        description: "Option value.",
    },
];
const TF_SIGNATURES: [BuiltinSignatureDescriptor; 6] = [
    BuiltinSignatureDescriptor {
        label: "s = tf('s')",
        inputs: &TF_INPUTS_VARIABLE,
        outputs: &TF_OUTPUT_SYS,
    },
    BuiltinSignatureDescriptor {
        label: "z = tf('z', Ts)",
        inputs: &TF_INPUTS_VARIABLE_TS,
        outputs: &TF_OUTPUT_SYS,
    },
    BuiltinSignatureDescriptor {
        label: "sys = tf(numerator, denominator)",
        inputs: &TF_INPUTS_NUM_DEN,
        outputs: &TF_OUTPUT_SYS,
    },
    BuiltinSignatureDescriptor {
        label: "sys = tf(numerator, denominator, Ts)",
        inputs: &TF_INPUTS_NUM_DEN_TS,
        outputs: &TF_OUTPUT_SYS,
    },
    BuiltinSignatureDescriptor {
        label: "sys = tf(numerator, denominator, \"Variable\", variableName)",
        inputs: &TF_INPUTS_NUM_DEN_NAMEVALUE,
        outputs: &TF_OUTPUT_SYS,
    },
    BuiltinSignatureDescriptor {
        label: "sys = tf(numerator, denominator, name, value, ...)",
        inputs: &TF_INPUTS_NUM_DEN_NAMEVALUE,
        outputs: &TF_OUTPUT_SYS,
    },
];
pub const TF_ERROR_INVALID_ARGUMENT: BuiltinErrorDescriptor = BuiltinErrorDescriptor {
    code: "RM.TF.INVALID_ARGUMENT",
    identifier: Some("RunMat:tf:InvalidArgument"),
    when: "Arguments do not match supported tf invocation forms.",
    message: "tf: invalid argument",
};
pub const TF_ERROR_INVALID_OPTION: BuiltinErrorDescriptor = BuiltinErrorDescriptor {
    code: "RM.TF.INVALID_OPTION",
    identifier: Some("RunMat:tf:InvalidOption"),
    when: "A name/value option token is unsupported or malformed.",
    message: "tf: invalid option",
};
pub const TF_ERROR_INVALID_SAMPLE_TIME: BuiltinErrorDescriptor = BuiltinErrorDescriptor {
    code: "RM.TF.INVALID_SAMPLE_TIME",
    identifier: Some("RunMat:tf:InvalidSampleTime"),
    when: "Sample time is neither -1 nor a finite non-negative scalar, or an integer sample time is not exactly representable as double.",
    message: "tf: sample time must be -1 or a finite non-negative scalar",
};
pub const TF_ERROR_INVALID_VARIABLE: BuiltinErrorDescriptor = BuiltinErrorDescriptor {
    code: "RM.TF.INVALID_VARIABLE",
    identifier: Some("RunMat:tf:InvalidVariable"),
    when: "Variable option is not a supported control variable name.",
    message: "tf: invalid Variable option",
};
pub const TF_ERROR_INVALID_COEFFICIENTS: BuiltinErrorDescriptor = BuiltinErrorDescriptor {
    code: "RM.TF.INVALID_COEFFICIENTS",
    identifier: Some("RunMat:tf:InvalidCoefficients"),
    when: "Numerator/denominator coefficients are not valid finite vectors, or integer coefficients are not exactly representable as double.",
    message: "tf: invalid coefficients",
};
pub const TF_ERROR_DENOMINATOR_INVALID: BuiltinErrorDescriptor = BuiltinErrorDescriptor {
    code: "RM.TF.DENOMINATOR_INVALID",
    identifier: Some("RunMat:tf:DenominatorInvalid"),
    when: "Denominator coefficient vector is empty or all zeros.",
    message: "tf: invalid denominator coefficients",
};
pub const TF_ERROR_INTERNAL: BuiltinErrorDescriptor = BuiltinErrorDescriptor {
    code: "RM.TF.INTERNAL",
    identifier: Some("RunMat:tf:Internal"),
    when: "Internal tensor/object construction or a shared transfer-function numerical operation failed.",
    message: "tf: internal error",
};
pub const TF_ERROR_INVALID_MODEL: BuiltinErrorDescriptor = BuiltinErrorDescriptor {
    code: "RM.TF.INVALID_MODEL",
    identifier: Some("RunMat:tf:InvalidModel"),
    when: "A transfer-function object has missing or invalid model properties.",
    message: "tf: invalid model",
};
pub const TF_ERROR_UNSUPPORTED_MODEL: BuiltinErrorDescriptor = BuiltinErrorDescriptor {
    code: "RM.TF.UNSUPPORTED_MODEL",
    identifier: Some("RunMat:tf:UnsupportedModel"),
    when: "The shared transfer-function parser receives an unsupported model representation.",
    message: "tf: unsupported model",
};
const TF_ERRORS: [BuiltinErrorDescriptor; 9] = [
    TF_ERROR_INVALID_ARGUMENT,
    TF_ERROR_INVALID_OPTION,
    TF_ERROR_INVALID_SAMPLE_TIME,
    TF_ERROR_INVALID_VARIABLE,
    TF_ERROR_INVALID_COEFFICIENTS,
    TF_ERROR_DENOMINATOR_INVALID,
    TF_ERROR_INTERNAL,
    TF_ERROR_INVALID_MODEL,
    TF_ERROR_UNSUPPORTED_MODEL,
];
pub const TF_DESCRIPTOR: BuiltinDescriptor = BuiltinDescriptor {
    signatures: &TF_SIGNATURES,
    output_mode: BuiltinOutputMode::Fixed,
    completion_policy: BuiltinCompletionPolicy::Public,
    errors: &TF_ERRORS,
};
