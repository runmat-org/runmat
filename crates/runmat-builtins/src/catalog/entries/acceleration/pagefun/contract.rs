use crate::*;
pub const PAGEFUN_HOST_INPUT_EXTENSION: BuiltinExtensionDescriptor = BuiltinExtensionDescriptor {
    id: "pagefun-host-inputs",
    mode: BuiltinExtensionMode::RunMatOnly,
    description: "pagefun with no distributed or gpuArray input is a RunMat extension",
    error_identifier: Some("RunMat:compatibility:PagefunHostInputsExtension"),
};

pub const PAGEFUN_TEXT_CALLABLE_EXTENSION: BuiltinExtensionDescriptor =
    BuiltinExtensionDescriptor {
        id: "pagefun-text-callable",
        mode: BuiltinExtensionMode::RunMatOnly,
        description: "pagefun text callables are a RunMat extension",
        error_identifier: Some("RunMat:compatibility:PagefunTextCallableExtension"),
    };

pub const PAGEFUN_EXTENSIONS: [BuiltinExtensionDescriptor; 2] = [
    PAGEFUN_HOST_INPUT_EXTENSION,
    PAGEFUN_TEXT_CALLABLE_EXTENSION,
];

const PAGEFUN_INTEGER_MTIMES_INPUT: [BuiltinIntegerInputCapability; 1] =
    [BuiltinIntegerInputCapability {
        name: "A or B page operand",
        classes: &crate::ALL_INTEGER_CLASSES,
        availability: BuiltinIntegerInputAvailability::Documented,
        scalar_double: BuiltinIntegerScalarDoubleRule::Allowed,
        notes: "pagefun delegates each page to the selected function, and the documented mtimes contract admits an integer operand only when the other operand is scalar.",
    }];

pub const PAGEFUN_INTEGER_CAPABILITIES: [BuiltinIntegerCapabilityDescriptor; 1] =
    [BuiltinIntegerCapabilityDescriptor {
        form: "Y = pagefun(@mtimes, integer_A, B)",
        inputs: &PAGEFUN_INTEGER_MTIMES_INPUT,
        computation_domain: BuiltinIntegerComputationDomain::FunctionSpecific,
        output_class: BuiltinIntegerOutputClassRule::FunctionSpecific,
        overflow: BuiltinIntegerOverflowRule::FunctionSpecific,
        backend: BuiltinIntegerBackendRule::GpuRestricted,
        overload: BuiltinIntegerOverloadKind::ScalarOnly,
        notes: "Each page is dispatched through mtimes. Integer pages retain authoritative same-class storage and per-operation saturation when the other page operand is scalar; resident integer inputs use exact owner-aware host fallback because the provider page kernel is floating-only.",
    }];

const PAGEFUN_OUTPUT: [BuiltinParamDescriptor; 1] = [BuiltinParamDescriptor {
    name: "Y",
    ty: BuiltinParamType::Any,
    arity: BuiltinParamArity::Required,
    default: None,
    description: "Result of applying the operator page-by-page.",
}];

const PAGEFUN_INPUTS_MT: [BuiltinParamDescriptor; 3] = [
    BuiltinParamDescriptor {
        name: "func",
        ty: BuiltinParamType::Any,
        arity: BuiltinParamArity::Required,
        default: None,
        description: "Operator/function handle (currently `@mtimes`).",
    },
    BuiltinParamDescriptor {
        name: "A",
        ty: BuiltinParamType::Any,
        arity: BuiltinParamArity::Required,
        default: None,
        description: "First paged input array.",
    },
    BuiltinParamDescriptor {
        name: "B",
        ty: BuiltinParamType::Any,
        arity: BuiltinParamArity::Required,
        default: None,
        description: "Second paged input array.",
    },
];

const PAGEFUN_INPUTS_VARIADIC: [BuiltinParamDescriptor; 3] = [
    BuiltinParamDescriptor {
        name: "func",
        ty: BuiltinParamType::Any,
        arity: BuiltinParamArity::Required,
        default: None,
        description: "Operator/function handle.",
    },
    BuiltinParamDescriptor {
        name: "A1",
        ty: BuiltinParamType::Any,
        arity: BuiltinParamArity::Required,
        default: None,
        description: "First paged input array.",
    },
    BuiltinParamDescriptor {
        name: "An",
        ty: BuiltinParamType::Any,
        arity: BuiltinParamArity::Variadic,
        default: None,
        description: "Additional paged input arrays.",
    },
];

const PAGEFUN_SIGNATURES: [BuiltinSignatureDescriptor; 2] = [
    BuiltinSignatureDescriptor {
        label: "Y = pagefun(func, A, B)",
        inputs: &PAGEFUN_INPUTS_MT,
        outputs: &PAGEFUN_OUTPUT,
    },
    BuiltinSignatureDescriptor {
        label: "Y = pagefun(func, A1, An...)",
        inputs: &PAGEFUN_INPUTS_VARIADIC,
        outputs: &PAGEFUN_OUTPUT,
    },
];

pub const PAGEFUN_ERROR_INVALID_CALLABLE: BuiltinErrorDescriptor = BuiltinErrorDescriptor {
    code: "RM.PAGEFUN.INVALID_CALLABLE",
    identifier: Some("RunMat:pagefun:InvalidCallable"),
    when: "Function selector is not a supported pagefun operator handle.",
    message: "pagefun: invalid function selector",
};

pub const PAGEFUN_ERROR_UNSUPPORTED_OPERATION: BuiltinErrorDescriptor = BuiltinErrorDescriptor {
    code: "RM.PAGEFUN.UNSUPPORTED_OPERATION",
    identifier: Some("RunMat:pagefun:UnsupportedOperation"),
    when: "Operator is not implemented by pagefun.",
    message: "pagefun: unsupported operation",
};

pub const PAGEFUN_ERROR_INVALID_ARITY: BuiltinErrorDescriptor = BuiltinErrorDescriptor {
    code: "RM.PAGEFUN.INVALID_ARITY",
    identifier: Some("RunMat:pagefun:InvalidArity"),
    when: "Provided operand count does not satisfy operator arity requirements.",
    message: "pagefun: invalid number of input arrays",
};

pub const PAGEFUN_ERROR_INVALID_INPUT: BuiltinErrorDescriptor = BuiltinErrorDescriptor {
    code: "RM.PAGEFUN.INVALID_INPUT",
    identifier: Some("RunMat:pagefun:InvalidInput"),
    when: "Input values are not supported numeric/pageable arrays for the operation.",
    message: "pagefun: invalid input array",
};

pub const PAGEFUN_ERROR_PAGE_DIM_MISMATCH: BuiltinErrorDescriptor = BuiltinErrorDescriptor {
    code: "RM.PAGEFUN.PAGE_DIM_MISMATCH",
    identifier: Some("RunMat:pagefun:PageDimensionMismatch"),
    when: "Trailing page dimensions cannot be broadcast across inputs.",
    message: "pagefun: page dimensions are not compatible",
};

pub const PAGEFUN_ERROR_MATRIX_DIM_MISMATCH: BuiltinErrorDescriptor = BuiltinErrorDescriptor {
    code: "RM.PAGEFUN.MATRIX_DIM_MISMATCH",
    identifier: Some("RunMat:pagefun:MatrixDimensionMismatch"),
    when: "Matrix inner dimensions do not align for the selected operation.",
    message: "pagefun: inner matrix dimensions must agree",
};

pub const PAGEFUN_ERROR_RESULT_SHAPE: BuiltinErrorDescriptor = BuiltinErrorDescriptor {
    code: "RM.PAGEFUN.RESULT_SHAPE",
    identifier: Some("RunMat:pagefun:ResultShapeMismatch"),
    when: "Per-page operator results produce inconsistent matrix shapes.",
    message: "pagefun: result matrix shape mismatch",
};

pub const PAGEFUN_ERROR_RESULT_TYPE: BuiltinErrorDescriptor = BuiltinErrorDescriptor {
    code: "RM.PAGEFUN.RESULT_TYPE",
    identifier: Some("RunMat:pagefun:ResultTypeMismatch"),
    when: "Per-page operator result is not a supported matrix/scalar type.",
    message: "pagefun: operator returned unsupported result type",
};

pub const PAGEFUN_ERROR_INTERNAL: BuiltinErrorDescriptor = BuiltinErrorDescriptor {
    code: "RM.PAGEFUN.INTERNAL",
    identifier: Some("RunMat:pagefun:InternalError"),
    when: "Internal pagefun execution/assembly failure occurred.",
    message: "pagefun: internal error",
};

pub const PAGEFUN_ERRORS: [BuiltinErrorDescriptor; 10] = [
    PAGEFUN_ERROR_OUTPUTS,
    PAGEFUN_ERROR_INVALID_CALLABLE,
    PAGEFUN_ERROR_UNSUPPORTED_OPERATION,
    PAGEFUN_ERROR_INVALID_ARITY,
    PAGEFUN_ERROR_INVALID_INPUT,
    PAGEFUN_ERROR_PAGE_DIM_MISMATCH,
    PAGEFUN_ERROR_MATRIX_DIM_MISMATCH,
    PAGEFUN_ERROR_RESULT_SHAPE,
    PAGEFUN_ERROR_RESULT_TYPE,
    PAGEFUN_ERROR_INTERNAL,
];

pub const PAGEFUN_DESCRIPTOR: BuiltinDescriptor = BuiltinDescriptor {
    signatures: &PAGEFUN_SIGNATURES,
    output_mode: BuiltinOutputMode::Fixed,
    completion_policy: BuiltinCompletionPolicy::Public,
    errors: &PAGEFUN_ERRORS,
};

pub const PAGEFUN_ERROR_OUTPUTS: BuiltinErrorDescriptor = BuiltinErrorDescriptor {
    code: "RM.PAGEFUN.TOO_MANY_OUTPUTS",
    identifier: Some("RunMat:pagefun:TooManyOutputs"),
    when: "More than one output is requested.",
    message: "pagefun: expected at most one output",
};
