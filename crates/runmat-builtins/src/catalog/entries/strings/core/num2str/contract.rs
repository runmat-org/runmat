use crate::*;

const OUTPUTS: [BuiltinParamDescriptor; 1] = [BuiltinParamDescriptor {
    name: "txt",
    ty: BuiltinParamType::Any,
    arity: BuiltinParamArity::Required,
    default: None,
    description: "Host character array of formatted numeric values.",
}];
const INPUT_VALUE: BuiltinParamDescriptor = BuiltinParamDescriptor {
    name: "A",
    ty: BuiltinParamType::Any,
    arity: BuiltinParamArity::Required,
    default: None,
    description: "Real or complex numeric or logical scalar, vector, or matrix.",
};
const INPUT_A: [BuiltinParamDescriptor; 1] = [INPUT_VALUE];
const INPUT_PRECISION: [BuiltinParamDescriptor; 2] = [
    INPUT_VALUE,
    BuiltinParamDescriptor {
        name: "p",
        ty: BuiltinParamType::IntegerScalar,
        arity: BuiltinParamArity::Required,
        default: None,
        description: "General-format precision from 0 through 52.",
    },
];
const INPUT_FORMAT: [BuiltinParamDescriptor; 2] = [
    INPUT_VALUE,
    BuiltinParamDescriptor {
        name: "formatSpec",
        ty: BuiltinParamType::StringScalar,
        arity: BuiltinParamArity::Required,
        default: None,
        description: "Single-value printf-style f, e, or g format.",
    },
];
const INPUT_LOCAL: [BuiltinParamDescriptor; 2] = [
    INPUT_VALUE,
    BuiltinParamDescriptor {
        name: "local",
        ty: BuiltinParamType::StringScalar,
        arity: BuiltinParamArity::Required,
        default: None,
        description: "The local decimal-separator option.",
    },
];
const INPUT_OPTION_LOCAL: [BuiltinParamDescriptor; 3] = [
    INPUT_VALUE,
    BuiltinParamDescriptor {
        name: "p_or_format",
        ty: BuiltinParamType::Any,
        arity: BuiltinParamArity::Required,
        default: None,
        description: "Precision or format string.",
    },
    BuiltinParamDescriptor {
        name: "local",
        ty: BuiltinParamType::StringScalar,
        arity: BuiltinParamArity::Required,
        default: None,
        description: "The local decimal-separator option.",
    },
];
const SIGNATURES: [BuiltinSignatureDescriptor; 5] = [
    BuiltinSignatureDescriptor {
        label: "txt = num2str(A)",
        inputs: &INPUT_A,
        outputs: &OUTPUTS,
    },
    BuiltinSignatureDescriptor {
        label: "txt = num2str(A, p)",
        inputs: &INPUT_PRECISION,
        outputs: &OUTPUTS,
    },
    BuiltinSignatureDescriptor {
        label: "txt = num2str(A, formatSpec)",
        inputs: &INPUT_FORMAT,
        outputs: &OUTPUTS,
    },
    BuiltinSignatureDescriptor {
        label: "txt = num2str(A, 'local')",
        inputs: &INPUT_LOCAL,
        outputs: &OUTPUTS,
    },
    BuiltinSignatureDescriptor {
        label: "txt = num2str(A, p_or_format, 'local')",
        inputs: &INPUT_OPTION_LOCAL,
        outputs: &OUTPUTS,
    },
];
pub const NUM2STR_ERROR_INVALID_INPUT: BuiltinErrorDescriptor = BuiltinErrorDescriptor {
    code: "RM.NUM2STR.INVALID_INPUT",
    identifier: Some("RunMat:num2str:InvalidInput"),
    when: "Input is not a supported numeric or logical scalar, vector, or matrix.",
    message: "num2str: unsupported input type",
};
pub const NUM2STR_ERROR_INVALID_OPTION: BuiltinErrorDescriptor = BuiltinErrorDescriptor {
    code: "RM.NUM2STR.INVALID_OPTION",
    identifier: Some("RunMat:num2str:InvalidOption"),
    when: "Options are malformed or too many arguments were supplied.",
    message: "num2str: invalid option arguments",
};
pub const NUM2STR_ERROR_INVALID_PRECISION: BuiltinErrorDescriptor = BuiltinErrorDescriptor {
    code: "RM.NUM2STR.INVALID_PRECISION",
    identifier: Some("RunMat:num2str:InvalidPrecision"),
    when: "Precision is non-finite, fractional, or outside 0 through 52.",
    message: "num2str: invalid precision",
};
pub const NUM2STR_ERROR_INVALID_FORMAT: BuiltinErrorDescriptor = BuiltinErrorDescriptor {
    code: "RM.NUM2STR.INVALID_FORMAT",
    identifier: Some("RunMat:num2str:InvalidFormat"),
    when: "Format text is unsupported or malformed.",
    message: "num2str: unsupported format string",
};
pub const NUM2STR_ERROR_INTERNAL: BuiltinErrorDescriptor = BuiltinErrorDescriptor {
    code: "RM.NUM2STR.INTERNAL",
    identifier: Some("RunMat:num2str:InternalError"),
    when: "Character-array assembly failed.",
    message: "num2str: internal error",
};
pub const NUM2STR_ERROR_TOO_MANY_OUTPUTS: BuiltinErrorDescriptor = BuiltinErrorDescriptor {
    code: "RM.NUM2STR.TOO_MANY_OUTPUTS",
    identifier: Some("RunMat:num2str:TooManyOutputs"),
    when: "More than one output is requested.",
    message: "num2str: too many output arguments",
};
pub const NUM2STR_DESCRIPTOR: BuiltinDescriptor = BuiltinDescriptor {
    signatures: &SIGNATURES,
    output_mode: BuiltinOutputMode::Fixed,
    completion_policy: BuiltinCompletionPolicy::Public,
    errors: &[
        NUM2STR_ERROR_INVALID_INPUT,
        NUM2STR_ERROR_INVALID_OPTION,
        NUM2STR_ERROR_INVALID_PRECISION,
        NUM2STR_ERROR_INVALID_FORMAT,
        NUM2STR_ERROR_INTERNAL,
        NUM2STR_ERROR_TOO_MANY_OUTPUTS,
    ],
};
pub const NUM2STR_EXPLICIT_GPU_EXTENSION: BuiltinExtensionDescriptor = BuiltinExtensionDescriptor {
    id: "num2str-explicit-gpu-input",
    mode: BuiltinExtensionMode::RunMatOnly,
    description: "num2str host formatting for explicit gpuArray input is a RunMat extension",
    error_identifier: Some("RunMat:compatibility:Num2strExplicitGpuInputExtension"),
};
pub const NUM2STR_EXTENSIONS: [BuiltinExtensionDescriptor; 1] = [NUM2STR_EXPLICIT_GPU_EXTENSION];
const INTEGER_VALUE_INPUTS: [BuiltinIntegerInputCapability; 1] = [BuiltinIntegerInputCapability {
    name: "A", classes: &ALL_INTEGER_CLASSES,
    availability: BuiltinIntegerInputAvailability::Documented,
    scalar_double: BuiltinIntegerScalarDoubleRule::NotApplicable,
    notes: "All integer classes are formatted from native storage, including exact wide and complex-integer components.",
}];
const INTEGER_PRECISION_INPUTS: [BuiltinIntegerInputCapability; 1] = [BuiltinIntegerInputCapability {
    name: "p", classes: &ALL_INTEGER_CLASSES,
    availability: BuiltinIntegerInputAvailability::Documented,
    scalar_double: BuiltinIntegerScalarDoubleRule::Allowed,
    notes: "The precision control is decoded and range checked exactly before conversion to a host usize.",
}];
pub const NUM2STR_INTEGER_CAPABILITIES: [BuiltinIntegerCapabilityDescriptor; 2] = [
    BuiltinIntegerCapabilityDescriptor {
        form: "txt = num2str(integer_A, ...)", inputs: &INTEGER_VALUE_INPUTS,
        computation_domain: BuiltinIntegerComputationDomain::ExactInteger,
        output_class: BuiltinIntegerOutputClassRule::FunctionSpecific,
        overflow: BuiltinIntegerOverflowRule::NotApplicable,
        backend: BuiltinIntegerBackendRule::GatherFallback,
        overload: BuiltinIntegerOverloadKind::Multiple,
        notes: "Default and custom formatting retain exact signed or unsigned values above flintmax. Automatic resident input gathers through its owner; explicit device input is separately gated.",
    },
    BuiltinIntegerCapabilityDescriptor {
        form: "txt = num2str(A, integer_p[, 'local'])", inputs: &INTEGER_PRECISION_INPUTS,
        computation_domain: BuiltinIntegerComputationDomain::Structural,
        output_class: BuiltinIntegerOutputClassRule::FunctionSpecific,
        overflow: BuiltinIntegerOverflowRule::Error,
        backend: BuiltinIntegerBackendRule::GatherFallback,
        overload: BuiltinIntegerOverloadKind::ScalarOnly,
        notes: "Precision is bounded to 0 through 52 and does not change numeric storage.",
    },
];
