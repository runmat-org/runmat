use crate::*;

pub const INTEGER_NUM_OUTPUTS_EXTENSION: BuiltinExtensionDescriptor = BuiltinExtensionDescriptor {
    id: "timeit-typed-integer-num-outputs",
    mode: BuiltinExtensionMode::RunMatOnly,
    description: "timeit with a typed-integer numOutputs value is a RunMat extension",
    error_identifier: Some("RunMat:compatibility:TimeitIntegerNumOutputsExtension"),
};
pub const EXPLICIT_GPU_NUM_OUTPUTS_EXTENSION: BuiltinExtensionDescriptor =
    BuiltinExtensionDescriptor {
        id: "timeit-explicit-gpu-num-outputs",
        mode: BuiltinExtensionMode::RunMatOnly,
        description: "timeit with an explicit gpuArray numOutputs value is a RunMat extension",
        error_identifier: Some("RunMat:compatibility:TimeitExplicitGpuNumOutputsExtension"),
    };
pub const EXTENSIONS: [BuiltinExtensionDescriptor; 2] = [
    INTEGER_NUM_OUTPUTS_EXTENSION,
    EXPLICIT_GPU_NUM_OUTPUTS_EXTENSION,
];

const INTEGER_NUM_OUTPUTS_INPUT: [BuiltinIntegerInputCapability; 1] =
    [BuiltinIntegerInputCapability {
        name: "numOutputs",
        classes: &ALL_INTEGER_CLASSES,
        availability: BuiltinIntegerInputAvailability::RunMatOnly,
        scalar_double: BuiltinIntegerScalarDoubleRule::Allowed,
        notes: "The public contract specifies an integer-valued output count without publishing native integer classes; ordinary integer-valued double input remains the compatibility form.",
    }];
pub const INTEGER_CAPABILITIES: [BuiltinIntegerCapabilityDescriptor; 1] =
    [BuiltinIntegerCapabilityDescriptor {
        form: "t = timeit(f, typed_integer_numOutputs)",
        inputs: &INTEGER_NUM_OUTPUTS_INPUT,
        computation_domain: BuiltinIntegerComputationDomain::Structural,
        output_class: BuiltinIntegerOutputClassRule::Double,
        overflow: BuiltinIntegerOverflowRule::Error,
        backend: BuiltinIntegerBackendRule::GatherFallback,
        overload: BuiltinIntegerOverloadKind::StructuralParameter,
        notes: "Admitted native integers are decoded exactly as a nonnegative platform output count. Automatic residency gathers transparently; explicit gpuArray intent is independently gated.",
    }];

const OUTPUT: [BuiltinParamDescriptor; 1] = [BuiltinParamDescriptor {
    name: "t",
    ty: BuiltinParamType::NumericScalar,
    arity: BuiltinParamArity::Required,
    default: None,
    description: "Median execution time per invocation in seconds.",
}];
const INPUTS_ONE: [BuiltinParamDescriptor; 1] = [BuiltinParamDescriptor {
    name: "f",
    ty: BuiltinParamType::Any,
    arity: BuiltinParamArity::Required,
    default: None,
    description: "Zero-input function handle to benchmark.",
}];
const INPUTS_TWO: [BuiltinParamDescriptor; 2] = [
    BuiltinParamDescriptor {
        name: "f",
        ty: BuiltinParamType::Any,
        arity: BuiltinParamArity::Required,
        default: None,
        description: "Zero-input function handle to benchmark.",
    },
    BuiltinParamDescriptor {
        name: "numOutputs",
        ty: BuiltinParamType::IntegerScalar,
        arity: BuiltinParamArity::Optional,
        default: Some("1"),
        description: "Requested output count for invoking the benchmarked handle.",
    },
];
const SIGNATURES: [BuiltinSignatureDescriptor; 2] = [
    BuiltinSignatureDescriptor {
        label: "t = timeit(f)",
        inputs: &INPUTS_ONE,
        outputs: &OUTPUT,
    },
    BuiltinSignatureDescriptor {
        label: "t = timeit(f, numOutputs)",
        inputs: &INPUTS_TWO,
        outputs: &OUTPUT,
    },
];

pub const ERROR_CONTEXT_UNAVAILABLE: BuiltinErrorDescriptor = BuiltinErrorDescriptor {
    code: "RM.TIMEIT.CONTEXT_UNAVAILABLE",
    identifier: Some("RunMat:timeit:ContextUnavailable"),
    when: "No invocation timing context is installed.",
    message: "timeit: runtime context is unavailable",
};
pub const ERROR_TOO_MANY_INPUTS: BuiltinErrorDescriptor = BuiltinErrorDescriptor {
    code: "RM.TIMEIT.TOO_MANY_INPUTS",
    identifier: Some("RunMat:timeit:TooManyInputs"),
    when: "More than two input arguments are supplied.",
    message: "timeit: too many input arguments",
};
pub const ERROR_NUM_OUTPUTS_SCALAR: BuiltinErrorDescriptor = BuiltinErrorDescriptor {
    code: "RM.TIMEIT.NUM_OUTPUTS_SCALAR",
    identifier: Some("RunMat:timeit:NumOutputsScalar"),
    when: "numOutputs is not a scalar numeric/integer value.",
    message: "timeit: numOutputs must be a scalar numeric value",
};
pub const ERROR_NUM_OUTPUTS_FINITE: BuiltinErrorDescriptor = BuiltinErrorDescriptor {
    code: "RM.TIMEIT.NUM_OUTPUTS_FINITE",
    identifier: Some("RunMat:timeit:NumOutputsFinite"),
    when: "numOutputs is NaN or infinite.",
    message: "timeit: numOutputs must be finite",
};
pub const ERROR_NUM_OUTPUTS_NONNEG: BuiltinErrorDescriptor = BuiltinErrorDescriptor {
    code: "RM.TIMEIT.NUM_OUTPUTS_NONNEGATIVE",
    identifier: Some("RunMat:timeit:NumOutputsNonnegative"),
    when: "numOutputs is negative.",
    message: "timeit: numOutputs must be a nonnegative integer",
};
pub const ERROR_NUM_OUTPUTS_INTEGER: BuiltinErrorDescriptor = BuiltinErrorDescriptor {
    code: "RM.TIMEIT.NUM_OUTPUTS_INTEGER",
    identifier: Some("RunMat:timeit:NumOutputsInteger"),
    when: "numOutputs has a non-integer numeric value.",
    message: "timeit: numOutputs must be an integer value",
};
pub const ERROR_EMPTY_HANDLE: BuiltinErrorDescriptor = BuiltinErrorDescriptor {
    code: "RM.TIMEIT.EMPTY_HANDLE",
    identifier: Some("RunMat:timeit:EmptyFunctionHandle"),
    when: "A function-handle string or payload is empty after trimming.",
    message: "timeit: empty function handle string",
};
pub const ERROR_EXPECTS_AT_HANDLE_STRING: BuiltinErrorDescriptor = BuiltinErrorDescriptor {
    code: "RM.TIMEIT.EXPECTS_AT_HANDLE_STRING",
    identifier: Some("RunMat:timeit:ExpectedAtHandleString"),
    when: "A string/char function handle does not begin with '@'.",
    message: "timeit: expected a function handle string beginning with '@'",
};
pub const ERROR_HANDLE_KIND: BuiltinErrorDescriptor = BuiltinErrorDescriptor {
    code: "RM.TIMEIT.HANDLE_KIND",
    identifier: Some("RunMat:timeit:HandleKind"),
    when: "Function handle argument is not a scalar string/char or callable handle value.",
    message: "timeit: function handle must be a string scalar or function handle",
};
pub const ERROR_FIRST_ARG_KIND: BuiltinErrorDescriptor = BuiltinErrorDescriptor {
    code: "RM.TIMEIT.FIRST_ARG_KIND",
    identifier: Some("RunMat:timeit:FirstArgKind"),
    when: "First argument is not a function handle value.",
    message: "timeit: first argument must be a function handle",
};
const ERRORS: [BuiltinErrorDescriptor; 10] = [
    ERROR_CONTEXT_UNAVAILABLE,
    ERROR_TOO_MANY_INPUTS,
    ERROR_NUM_OUTPUTS_SCALAR,
    ERROR_NUM_OUTPUTS_FINITE,
    ERROR_NUM_OUTPUTS_NONNEG,
    ERROR_NUM_OUTPUTS_INTEGER,
    ERROR_EMPTY_HANDLE,
    ERROR_EXPECTS_AT_HANDLE_STRING,
    ERROR_HANDLE_KIND,
    ERROR_FIRST_ARG_KIND,
];
pub const DESCRIPTOR: BuiltinDescriptor = BuiltinDescriptor {
    signatures: &SIGNATURES,
    output_mode: BuiltinOutputMode::Fixed,
    completion_policy: BuiltinCompletionPolicy::Public,
    errors: &ERRORS,
};
