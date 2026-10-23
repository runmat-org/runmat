use crate::*;

pub const DIFF_CHARACTER_INPUT_EXTENSION: BuiltinExtensionDescriptor = BuiltinExtensionDescriptor {
    id: "diff-character-input",
    mode: BuiltinExtensionMode::RunMatOnly,
    description: "diff with character input is a RunMat extension",
    error_identifier: Some("RunMat:compatibility:DiffCharacterInputExtension"),
};

pub const DIFF_ZERO_ORDER_EXTENSION: BuiltinExtensionDescriptor = BuiltinExtensionDescriptor {
    id: "diff-zero-order",
    mode: BuiltinExtensionMode::RunMatOnly,
    description: "diff with zero difference order is a RunMat extension",
    error_identifier: Some("RunMat:compatibility:DiffZeroOrderExtension"),
};

pub const DIFF_EMPTY_CONTROL_EXTENSION: BuiltinExtensionDescriptor = BuiltinExtensionDescriptor {
    id: "diff-empty-control",
    mode: BuiltinExtensionMode::RunMatOnly,
    description: "diff with an empty order or dimension placeholder is a RunMat extension",
    error_identifier: Some("RunMat:compatibility:DiffEmptyControlExtension"),
};

pub const DIFF_EXTENSIONS: [BuiltinExtensionDescriptor; 3] = [
    DIFF_CHARACTER_INPUT_EXTENSION,
    DIFF_ZERO_ORDER_EXTENSION,
    DIFF_EMPTY_CONTROL_EXTENSION,
];

const DIFF_INTEGER_DATA_INPUT: [BuiltinIntegerInputCapability; 1] =
    [BuiltinIntegerInputCapability {
        name: "X",
        classes: &crate::ALL_INTEGER_CLASSES,
        availability: BuiltinIntegerInputAvailability::Documented,
        scalar_double: BuiltinIntegerScalarDoubleRule::NotApplicable,
        notes: "The compatibility target documents every integer class for finite-difference data.",
    }];

const DIFF_INTEGER_ORDER_INPUT: [BuiltinIntegerInputCapability; 1] =
    [BuiltinIntegerInputCapability {
        name: "n",
        classes: &crate::ALL_INTEGER_CLASSES,
        availability: BuiltinIntegerInputAvailability::Documented,
        scalar_double: BuiltinIntegerScalarDoubleRule::NotApplicable,
        notes: "All eight integer classes are documented for the positive scalar difference order.",
    }];

const DIFF_INTEGER_DIM_INPUT: [BuiltinIntegerInputCapability; 1] =
    [BuiltinIntegerInputCapability {
        name: "dim",
        classes: &crate::ALL_INTEGER_CLASSES,
        availability: BuiltinIntegerInputAvailability::Documented,
        scalar_double: BuiltinIntegerScalarDoubleRule::NotApplicable,
        notes: "All eight integer classes are documented for the positive scalar dimension.",
    }];

pub const DIFF_INTEGER_CAPABILITIES: [BuiltinIntegerCapabilityDescriptor; 3] = [
    BuiltinIntegerCapabilityDescriptor {
        form: "Y = diff(integer_X, n?, dim?)",
        inputs: &DIFF_INTEGER_DATA_INPUT,
        computation_domain: BuiltinIntegerComputationDomain::ExactInteger,
        output_class: BuiltinIntegerOutputClassRule::PreserveInput,
        overflow: BuiltinIntegerOverflowRule::Saturate,
        backend: BuiltinIntegerBackendRule::GatherFallback,
        overload: BuiltinIntegerOverloadKind::FunctionSpecific,
        notes: "Each recursive adjacent subtraction preserves X's integer class and saturates at its bounds. Automatic resident input can use an admitted exact typed host fallback and owner-aware restoration. Explicit gpuArray integer input has no resident difference route and is rejected before gathering.",
    },
    BuiltinIntegerCapabilityDescriptor {
        form: "Y = diff(X, integer_n)",
        inputs: &DIFF_INTEGER_ORDER_INPUT,
        computation_domain: BuiltinIntegerComputationDomain::Structural,
        output_class: BuiltinIntegerOutputClassRule::FunctionSpecific,
        overflow: BuiltinIntegerOverflowRule::Error,
        backend: BuiltinIntegerBackendRule::HostAndGpu,
        overload: BuiltinIntegerOverloadKind::StructuralParameter,
        notes: "The positive scalar order is decoded exactly before host or provider execution.",
    },
    BuiltinIntegerCapabilityDescriptor {
        form: "Y = diff(X, n, integer_dim)",
        inputs: &DIFF_INTEGER_DIM_INPUT,
        computation_domain: BuiltinIntegerComputationDomain::Structural,
        output_class: BuiltinIntegerOutputClassRule::FunctionSpecific,
        overflow: BuiltinIntegerOverflowRule::Error,
        backend: BuiltinIntegerBackendRule::HostAndGpu,
        overload: BuiltinIntegerOverloadKind::StructuralParameter,
        notes: "The positive one-based dimension is decoded exactly before host or provider execution.",
    },
];

const DIFF_OUTPUT_B: [BuiltinParamDescriptor; 1] = [BuiltinParamDescriptor {
    name: "B",
    ty: BuiltinParamType::NumericArray,
    arity: BuiltinParamArity::Required,
    default: None,
    description: "Finite differences along the selected dimension.",
}];

const DIFF_INPUTS_X: [BuiltinParamDescriptor; 1] = [BuiltinParamDescriptor {
    name: "X",
    ty: BuiltinParamType::Any,
    arity: BuiltinParamArity::Required,
    default: None,
    description: "Input scalar or array.",
}];

const DIFF_INPUTS_X_N: [BuiltinParamDescriptor; 2] = [
    BuiltinParamDescriptor {
        name: "X",
        ty: BuiltinParamType::Any,
        arity: BuiltinParamArity::Required,
        default: None,
        description: "Input scalar or array.",
    },
    BuiltinParamDescriptor {
        name: "n",
        ty: BuiltinParamType::Any,
        arity: BuiltinParamArity::Optional,
        default: Some("1"),
        description: "Difference order (non-negative integer scalar or empty placeholder).",
    },
];

const DIFF_INPUTS_X_N_DIM: [BuiltinParamDescriptor; 3] = [
    BuiltinParamDescriptor {
        name: "X",
        ty: BuiltinParamType::Any,
        arity: BuiltinParamArity::Required,
        default: None,
        description: "Input scalar or array.",
    },
    BuiltinParamDescriptor {
        name: "n",
        ty: BuiltinParamType::Any,
        arity: BuiltinParamArity::Optional,
        default: Some("1"),
        description: "Difference order (non-negative integer scalar or empty placeholder).",
    },
    BuiltinParamDescriptor {
        name: "dim",
        ty: BuiltinParamType::Any,
        arity: BuiltinParamArity::Optional,
        default: Some("[]"),
        description: "Reduction dimension (positive integer scalar or empty placeholder).",
    },
];

const DIFF_SYMBOLIC_OUTPUT: [BuiltinParamDescriptor; 1] = [BuiltinParamDescriptor {
    name: "D",
    ty: BuiltinParamType::Any,
    arity: BuiltinParamArity::Required,
    default: None,
    description:
        "Scalar symbolic derivative expression; no numerical finite differences are performed.",
}];
const DIFF_SYMBOLIC_EXPRESSION: BuiltinParamDescriptor = BuiltinParamDescriptor {
    name: "expr",
    ty: BuiltinParamType::Any,
    arity: BuiltinParamArity::Required,
    default: None,
    description:
        "Scalar symbolic expression. An omitted variable is inferred only when unambiguous.",
};
const DIFF_SYMBOLIC_VARIABLE: BuiltinParamDescriptor = BuiltinParamDescriptor {
    name:"var",ty:BuiltinParamType::Any,arity:BuiltinParamArity::Required,default:None,
    description:"Symbolic variable identity or scalar text naming a variable; arbitrary numeric data is not a variable.",
};
const DIFF_SYMBOLIC_ORDER: BuiltinParamDescriptor = BuiltinParamDescriptor {
    name: "n",
    ty: BuiltinParamType::Any,
    arity: BuiltinParamArity::Required,
    default: None,
    description: "Constant nonnegative integer scalar order, no greater than uint32 maximum.",
};
const DIFF_SYMBOLIC_CORE: [BuiltinParamDescriptor; 1] = [DIFF_SYMBOLIC_EXPRESSION];
const DIFF_SYMBOLIC_VAR: [BuiltinParamDescriptor; 2] =
    [DIFF_SYMBOLIC_EXPRESSION, DIFF_SYMBOLIC_VARIABLE];
const DIFF_SYMBOLIC_N: [BuiltinParamDescriptor; 2] =
    [DIFF_SYMBOLIC_EXPRESSION, DIFF_SYMBOLIC_ORDER];
const DIFF_SYMBOLIC_VAR_N: [BuiltinParamDescriptor; 3] = [
    DIFF_SYMBOLIC_EXPRESSION,
    DIFF_SYMBOLIC_VARIABLE,
    DIFF_SYMBOLIC_ORDER,
];
const DIFF_SYMBOLIC_N_VAR: [BuiltinParamDescriptor; 3] = [
    DIFF_SYMBOLIC_EXPRESSION,
    DIFF_SYMBOLIC_ORDER,
    DIFF_SYMBOLIC_VARIABLE,
];
const DIFF_SIGNATURES: [BuiltinSignatureDescriptor; 8] = [
    BuiltinSignatureDescriptor {
        label: "B = diff(X)",
        inputs: &DIFF_INPUTS_X,
        outputs: &DIFF_OUTPUT_B,
    },
    BuiltinSignatureDescriptor {
        label: "B = diff(X, n)",
        inputs: &DIFF_INPUTS_X_N,
        outputs: &DIFF_OUTPUT_B,
    },
    BuiltinSignatureDescriptor {
        label: "B = diff(X, n, dim)",
        inputs: &DIFF_INPUTS_X_N_DIM,
        outputs: &DIFF_OUTPUT_B,
    },
    BuiltinSignatureDescriptor {
        label: "D = diff(expr)",
        inputs: &DIFF_SYMBOLIC_CORE,
        outputs: &DIFF_SYMBOLIC_OUTPUT,
    },
    BuiltinSignatureDescriptor {
        label: "D = diff(expr, var)",
        inputs: &DIFF_SYMBOLIC_VAR,
        outputs: &DIFF_SYMBOLIC_OUTPUT,
    },
    BuiltinSignatureDescriptor {
        label: "D = diff(expr, n)",
        inputs: &DIFF_SYMBOLIC_N,
        outputs: &DIFF_SYMBOLIC_OUTPUT,
    },
    BuiltinSignatureDescriptor {
        label: "D = diff(expr, var, n)",
        inputs: &DIFF_SYMBOLIC_VAR_N,
        outputs: &DIFF_SYMBOLIC_OUTPUT,
    },
    BuiltinSignatureDescriptor {
        label: "D = diff(expr, n, var)",
        inputs: &DIFF_SYMBOLIC_N_VAR,
        outputs: &DIFF_SYMBOLIC_OUTPUT,
    },
];

pub const DIFF_ERROR_INVALID_ARGUMENT: BuiltinErrorDescriptor = BuiltinErrorDescriptor {
    code: "RM.DIFF.INVALID_ARGUMENT",
    identifier: Some("RunMat:diff:InvalidArgument"),
    when: "Argument count/order/dimension/order grammar is invalid.",
    message: "diff: invalid argument",
};

pub const DIFF_ERROR_INVALID_INPUT: BuiltinErrorDescriptor = BuiltinErrorDescriptor {
    code: "RM.DIFF.INVALID_INPUT",
    identifier: Some("RunMat:diff:InvalidInput"),
    when: "Input value cannot be converted to a supported diff domain.",
    message: "diff: invalid input",
};

pub const DIFF_ERROR_INTERNAL: BuiltinErrorDescriptor = BuiltinErrorDescriptor {
    code: "RM.DIFF.INTERNAL",
    identifier: Some("RunMat:diff:Internal"),
    when: "Finite-difference execution fails due to conversion, gather, allocation, or reshape operations.",
    message: "diff: internal failure",
};

pub const DIFF_ERROR_TOO_MANY_OUTPUTS: BuiltinErrorDescriptor = BuiltinErrorDescriptor {
    code: "RM.DIFF.TOO_MANY_OUTPUTS",
    identifier: Some("RunMat:diff:TooManyOutputs"),
    when: "More than one output is requested.",
    message: "diff: too many output arguments",
};

pub const DIFF_ERRORS: [BuiltinErrorDescriptor; 4] = [
    DIFF_ERROR_INVALID_ARGUMENT,
    DIFF_ERROR_INVALID_INPUT,
    DIFF_ERROR_INTERNAL,
    DIFF_ERROR_TOO_MANY_OUTPUTS,
];

pub const DIFF_DESCRIPTOR: BuiltinDescriptor = BuiltinDescriptor {
    signatures: &DIFF_SIGNATURES,
    output_mode: BuiltinOutputMode::Fixed,
    completion_policy: BuiltinCompletionPolicy::Public,
    errors: &DIFF_ERRORS,
};
