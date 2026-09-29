use crate::*;

const SS_OUTPUT_SYS: [BuiltinParamDescriptor; 1] = [BuiltinParamDescriptor {
    name: "sys",
    ty: BuiltinParamType::Any,
    arity: BuiltinParamArity::Required,
    default: None,
    description: "State-space model object.",
}];
const SS_PARAM_A: BuiltinParamDescriptor = BuiltinParamDescriptor {
    name: "A",
    ty: BuiltinParamType::NumericArray,
    arity: BuiltinParamArity::Required,
    default: None,
    description: "State matrix with shape n-by-n.",
};
const SS_PARAM_B: BuiltinParamDescriptor = BuiltinParamDescriptor {
    name: "B",
    ty: BuiltinParamType::NumericArray,
    arity: BuiltinParamArity::Required,
    default: None,
    description: "Input matrix with shape n-by-nu.",
};
const SS_PARAM_C: BuiltinParamDescriptor = BuiltinParamDescriptor {
    name: "C",
    ty: BuiltinParamType::NumericArray,
    arity: BuiltinParamArity::Required,
    default: None,
    description: "Output matrix with shape ny-by-n.",
};
const SS_PARAM_D: BuiltinParamDescriptor = BuiltinParamDescriptor {
    name: "D",
    ty: BuiltinParamType::NumericArray,
    arity: BuiltinParamArity::Required,
    default: None,
    description: "Feedthrough matrix with shape ny-by-nu.",
};
const SS_INPUTS_ABCD: [BuiltinParamDescriptor; 4] =
    [SS_PARAM_A, SS_PARAM_B, SS_PARAM_C, SS_PARAM_D];
const SS_INPUTS_ABCD_TS: [BuiltinParamDescriptor; 5] = [
    SS_PARAM_A,
    SS_PARAM_B,
    SS_PARAM_C,
    SS_PARAM_D,
    BuiltinParamDescriptor {
        name: "Ts",
        ty: BuiltinParamType::NumericScalar,
        arity: BuiltinParamArity::Optional,
        default: Some("0.0"),
        description: "Sample time (0 for continuous-time model).",
    },
];
const SS_INPUTS_ABCD_NAMEVALUE: [BuiltinParamDescriptor; 6] = [
    SS_PARAM_A,
    SS_PARAM_B,
    SS_PARAM_C,
    SS_PARAM_D,
    BuiltinParamDescriptor {
        name: "name",
        ty: BuiltinParamType::StringScalar,
        arity: BuiltinParamArity::Variadic,
        default: None,
        description: "Option name ('Ts' or 'SampleTime').",
    },
    BuiltinParamDescriptor {
        name: "value",
        ty: BuiltinParamType::Any,
        arity: BuiltinParamArity::Variadic,
        default: None,
        description: "Option value.",
    },
];
const SS_SIGNATURES: [BuiltinSignatureDescriptor; 4] = [
    BuiltinSignatureDescriptor {
        label: "sys = ss(A, B, C, D)",
        inputs: &SS_INPUTS_ABCD,
        outputs: &SS_OUTPUT_SYS,
    },
    BuiltinSignatureDescriptor {
        label: "sys = ss(A, B, C, D, Ts)",
        inputs: &SS_INPUTS_ABCD_TS,
        outputs: &SS_OUTPUT_SYS,
    },
    BuiltinSignatureDescriptor {
        label: "sys = ss(A, B, C, D, \"Ts\", Ts)",
        inputs: &SS_INPUTS_ABCD_NAMEVALUE,
        outputs: &SS_OUTPUT_SYS,
    },
    BuiltinSignatureDescriptor {
        label: "sys = ss(A, B, C, D, name, value, ...)",
        inputs: &SS_INPUTS_ABCD_NAMEVALUE,
        outputs: &SS_OUTPUT_SYS,
    },
];
pub const SS_ERROR_INVALID_ARGUMENT: BuiltinErrorDescriptor = BuiltinErrorDescriptor {
    code: "RM.SS.INVALID_ARGUMENT",
    identifier: Some("RunMat:ss:InvalidArgument"),
    when: "Arguments do not match supported ss invocation forms.",
    message: "ss: invalid argument",
};
pub const SS_ERROR_INVALID_OPTION: BuiltinErrorDescriptor = BuiltinErrorDescriptor {
    code: "RM.SS.INVALID_OPTION",
    identifier: Some("RunMat:ss:InvalidOption"),
    when: "A name/value option token is unsupported or malformed.",
    message: "ss: invalid option",
};
pub const SS_ERROR_INVALID_SAMPLE_TIME: BuiltinErrorDescriptor = BuiltinErrorDescriptor {
    code: "RM.SS.INVALID_SAMPLE_TIME",
    identifier: Some("RunMat:ss:InvalidSampleTime"),
    when: "Sample time is not a finite non-negative scalar, or an integer sample time cannot be represented exactly as double.",
    message: "ss: sample time must be a finite non-negative scalar",
};
pub const SS_ERROR_INVALID_MATRIX: BuiltinErrorDescriptor = BuiltinErrorDescriptor {
    code: "RM.SS.INVALID_MATRIX",
    identifier: Some("RunMat:ss:InvalidMatrix"),
    when: "An integer matrix value cannot cross the double-valued state-space boundary exactly.",
    message: "ss: invalid state-space matrix value",
};
pub const SS_ERROR_INVALID_DIMENSIONS: BuiltinErrorDescriptor = BuiltinErrorDescriptor {
    code: "RM.SS.INVALID_DIMENSIONS",
    identifier: Some("RunMat:ss:InvalidDimensions"),
    when: "A, B, C, and D dimensions do not define a consistent state-space model.",
    message: "ss: invalid state-space matrix dimensions",
};
pub const SS_ERROR_UNSUPPORTED_INPUT: BuiltinErrorDescriptor = BuiltinErrorDescriptor {
    code: "RM.SS.UNSUPPORTED_INPUT",
    identifier: Some("RunMat:ss:UnsupportedInput"),
    when: "An input is complex, sparse, logical, or another unsupported model form.",
    message: "ss: unsupported input",
};
pub const SS_ERROR_INTERNAL: BuiltinErrorDescriptor = BuiltinErrorDescriptor {
    code: "RM.SS.INTERNAL",
    identifier: Some("RunMat:ss:Internal"),
    when: "Internal tensor/object construction failed.",
    message: "ss: internal error",
};
const SS_ERRORS: [BuiltinErrorDescriptor; 7] = [
    SS_ERROR_INVALID_ARGUMENT,
    SS_ERROR_INVALID_OPTION,
    SS_ERROR_INVALID_SAMPLE_TIME,
    SS_ERROR_INVALID_MATRIX,
    SS_ERROR_INVALID_DIMENSIONS,
    SS_ERROR_UNSUPPORTED_INPUT,
    SS_ERROR_INTERNAL,
];
pub const SS_DESCRIPTOR: BuiltinDescriptor = BuiltinDescriptor {
    signatures: &SS_SIGNATURES,
    output_mode: BuiltinOutputMode::Fixed,
    completion_policy: BuiltinCompletionPolicy::Public,
    errors: &SS_ERRORS,
};

pub const SS_INTEGER_MATRIX_EXTENSION: BuiltinExtensionDescriptor = BuiltinExtensionDescriptor {
    id: "ss-integer-matrix-input",
    mode: BuiltinExtensionMode::RunMatOnly,
    description: "ss with typed-integer state-space matrices is a RunMat extension",
    error_identifier: Some("RunMat:compatibility:SsIntegerMatrixInputExtension"),
};
pub const SS_INTEGER_SAMPLE_TIME_EXTENSION: BuiltinExtensionDescriptor =
    BuiltinExtensionDescriptor {
        id: "ss-integer-sample-time",
        mode: BuiltinExtensionMode::RunMatOnly,
        description: "ss with a typed-integer sample time is a RunMat extension",
        error_identifier: Some("RunMat:compatibility:SsIntegerSampleTimeExtension"),
    };
pub const SS_EXPLICIT_GPU_EXTENSION: BuiltinExtensionDescriptor = BuiltinExtensionDescriptor {
    id: "ss-explicit-gpu-input",
    mode: BuiltinExtensionMode::RunMatOnly,
    description: "ss with explicit GPU input data is a RunMat extension",
    error_identifier: Some("RunMat:compatibility:SsExplicitGpuInputExtension"),
};
pub const SS_EXTENSIONS: [BuiltinExtensionDescriptor; 3] = [
    SS_INTEGER_MATRIX_EXTENSION,
    SS_INTEGER_SAMPLE_TIME_EXTENSION,
    SS_EXPLICIT_GPU_EXTENSION,
];

const SS_INTEGER_MATRIX_INPUTS: [BuiltinIntegerInputCapability; 4] = [
    ss_integer_matrix_input("A"),
    ss_integer_matrix_input("B"),
    ss_integer_matrix_input("C"),
    ss_integer_matrix_input("D"),
];
const SS_INTEGER_SAMPLE_TIME_INPUT: [BuiltinIntegerInputCapability; 1] =
    [BuiltinIntegerInputCapability {
        name: "Ts",
        classes: &ALL_INTEGER_CLASSES,
        availability: BuiltinIntegerInputAvailability::RunMatOnly,
        scalar_double: BuiltinIntegerScalarDoubleRule::Allowed,
        notes: "RunMat admits a typed scalar only when its exact value can enter the binary64 model metadata.",
    }];
const fn ss_integer_matrix_input(name: &'static str) -> BuiltinIntegerInputCapability {
    BuiltinIntegerInputCapability {
        name,
        classes: &ALL_INTEGER_CLASSES,
        availability: BuiltinIntegerInputAvailability::RunMatOnly,
        scalar_double: BuiltinIntegerScalarDoubleRule::NotApplicable,
        notes: "The documented public surface does not establish typed-integer storage; RunMat requires every value to be exact at the binary64 model boundary.",
    }
}
pub const SS_INTEGER_CAPABILITIES: [BuiltinIntegerCapabilityDescriptor; 2] = [
    BuiltinIntegerCapabilityDescriptor {
        form: "sys = ss(integer_A, integer_B, integer_C, integer_D, ...)",
        inputs: &SS_INTEGER_MATRIX_INPUTS,
        computation_domain: BuiltinIntegerComputationDomain::FloatingPoint,
        output_class: BuiltinIntegerOutputClassRule::Double,
        overflow: BuiltinIntegerOverflowRule::Error,
        backend: BuiltinIntegerBackendRule::GatherFallback,
        overload: BuiltinIntegerOverloadKind::Multiple,
        notes: "Each typed matrix is an independently gated RunMat extension and is checked before conversion; the state-space object's numeric matrices are binary64.",
    },
    BuiltinIntegerCapabilityDescriptor {
        form: "sys = ss(A, B, C, D, integer_Ts)",
        inputs: &SS_INTEGER_SAMPLE_TIME_INPUT,
        computation_domain: BuiltinIntegerComputationDomain::FloatingPoint,
        output_class: BuiltinIntegerOutputClassRule::Double,
        overflow: BuiltinIntegerOverflowRule::Error,
        backend: BuiltinIntegerBackendRule::GatherFallback,
        overload: BuiltinIntegerOverloadKind::StructuralParameter,
        notes: "Typed sample time is a gated RunMat extension stored as binary64 model metadata after an exact representability check; the public contract requires a numeric scalar without enumerating typed storage classes.",
    },
];
