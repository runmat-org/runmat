use crate::*;

const PROD_OUTPUT: [BuiltinParamDescriptor; 1] = [BuiltinParamDescriptor {
    name: "B",
    ty: BuiltinParamType::Any,
    arity: BuiltinParamArity::Required,
    default: None,
    description:
        "Product reduction in the selected numeric class, or logical for native logical input.",
}];

const PROD_INPUTS_CORE: [BuiltinParamDescriptor; 1] = [BuiltinParamDescriptor {
    name: "A",
    ty: BuiltinParamType::Any,
    arity: BuiltinParamArity::Required,
    default: None,
    description: "Input array.",
}];

const PROD_INPUTS_DIM: [BuiltinParamDescriptor; 2] = [
    BuiltinParamDescriptor {
        name: "A",
        ty: BuiltinParamType::Any,
        arity: BuiltinParamArity::Required,
        default: None,
        description: "Input array.",
    },
    BuiltinParamDescriptor {
        name: "dim",
        ty: BuiltinParamType::Any,
        arity: BuiltinParamArity::Optional,
        default: None,
        description: "Positive dimension selector or vector of dimensions; omission selects the first non-singleton axis. Only vector-shaped empty selectors (1-by-0 or 0-by-1) select this default; a 0-by-0 empty array is not a selector.",
    },
];

const PROD_INPUTS_ALL: [BuiltinParamDescriptor; 2] = [
    BuiltinParamDescriptor {
        name: "A",
        ty: BuiltinParamType::Any,
        arity: BuiltinParamArity::Required,
        default: None,
        description: "Input array.",
    },
    BuiltinParamDescriptor {
        name: "all",
        ty: BuiltinParamType::StringScalar,
        arity: BuiltinParamArity::Optional,
        default: Some("\"all\""),
        description: "Reduce across all dimensions.",
    },
];

const PROD_INPUTS_MISSINGFLAG: [BuiltinParamDescriptor; 2] = [
    BuiltinParamDescriptor {
        name: "A",
        ty: BuiltinParamType::Any,
        arity: BuiltinParamArity::Required,
        default: None,
        description: "Input array.",
    },
    BuiltinParamDescriptor {
        name: "missingflag",
        ty: BuiltinParamType::StringScalar,
        arity: BuiltinParamArity::Optional,
        default: Some("\"includemissing\""),
        description: "Missing-data handling mode: \"includemissing\" or \"omitmissing\".",
    },
];

const PROD_INPUTS_OUTTYPE: [BuiltinParamDescriptor; 2] = [
    BuiltinParamDescriptor {
        name: "A",
        ty: BuiltinParamType::Any,
        arity: BuiltinParamArity::Required,
        default: None,
        description: "Input array.",
    },
    BuiltinParamDescriptor {
        name: "outtype",
        ty: BuiltinParamType::StringScalar,
        arity: BuiltinParamArity::Optional,
        default: Some("\"default\""),
        description: "Output class specifier: \"double\", \"default\", or \"native\".",
    },
];

const PROD_INPUTS_DIM_MISSINGFLAG: [BuiltinParamDescriptor; 3] = [
    BuiltinParamDescriptor {
        name: "A",
        ty: BuiltinParamType::Any,
        arity: BuiltinParamArity::Required,
        default: None,
        description: "Input array.",
    },
    BuiltinParamDescriptor {
        name: "dim",
        ty: BuiltinParamType::Any,
        arity: BuiltinParamArity::Optional,
        default: None,
        description: "Positive dimension selector or vector of dimensions; omission selects the first non-singleton axis. Only vector-shaped empty selectors (1-by-0 or 0-by-1) select this default; a 0-by-0 empty array is not a selector.",
    },
    BuiltinParamDescriptor {
        name: "missingflag",
        ty: BuiltinParamType::StringScalar,
        arity: BuiltinParamArity::Optional,
        default: Some("\"includemissing\""),
        description: "Missing-data handling mode: \"includemissing\" or \"omitmissing\".",
    },
];

const PROD_INPUTS_MISSINGFLAG_DIM: [BuiltinParamDescriptor; 3] = [
    BuiltinParamDescriptor {
        name: "A",
        ty: BuiltinParamType::Any,
        arity: BuiltinParamArity::Required,
        default: None,
        description: "Input array.",
    },
    BuiltinParamDescriptor {
        name: "missingflag",
        ty: BuiltinParamType::StringScalar,
        arity: BuiltinParamArity::Optional,
        default: Some("\"includemissing\""),
        description: "Missing-data handling mode: \"includemissing\" or \"omitmissing\".",
    },
    BuiltinParamDescriptor {
        name: "dim",
        ty: BuiltinParamType::Any,
        arity: BuiltinParamArity::Optional,
        default: None,
        description: "Positive dimension selector or vector of dimensions; omission selects the first non-singleton axis. Only vector-shaped empty selectors (1-by-0 or 0-by-1) select this default; a 0-by-0 empty array is not a selector.",
    },
];

const PROD_INPUTS_OPTIONS: [BuiltinParamDescriptor; 2] = [
    BuiltinParamDescriptor {name:"A",ty:BuiltinParamType::Any,arity:BuiltinParamArity::Required,default:None,description:"Dense real numeric or logical input."},
    BuiltinParamDescriptor {
        name: "options", ty: BuiltinParamType::Any, arity: BuiltinParamArity::Variadic,
        default: None,
        description: "One dimension/vector/all selection, optional missing-data flags (last wins), and one default/double/native class control, in either supported order. A vector-shaped empty selection retains the default axis. No like form is accepted.",
    },
];
const PROD_SIGNATURES: [BuiltinSignatureDescriptor; 9] = [
    BuiltinSignatureDescriptor {
        label: "B = prod(A)",
        inputs: &PROD_INPUTS_CORE,
        outputs: &PROD_OUTPUT,
    },
    BuiltinSignatureDescriptor {
        label: "B = prod(A, dim)",
        inputs: &PROD_INPUTS_DIM,
        outputs: &PROD_OUTPUT,
    },
    BuiltinSignatureDescriptor {
        label: "B = prod(A, \"all\")",
        inputs: &PROD_INPUTS_ALL,
        outputs: &PROD_OUTPUT,
    },
    BuiltinSignatureDescriptor {
        label: "B = prod(A, missingflag)",
        inputs: &PROD_INPUTS_MISSINGFLAG,
        outputs: &PROD_OUTPUT,
    },
    BuiltinSignatureDescriptor {
        label: "B = prod(A, outtype)",
        inputs: &PROD_INPUTS_OUTTYPE,
        outputs: &PROD_OUTPUT,
    },
    BuiltinSignatureDescriptor {
        label: "B = prod(A, dim, missingflag)",
        inputs: &PROD_INPUTS_DIM_MISSINGFLAG,
        outputs: &PROD_OUTPUT,
    },
    BuiltinSignatureDescriptor {
        label: "B = prod(A, missingflag, dim)",
        inputs: &PROD_INPUTS_MISSINGFLAG_DIM,
        outputs: &PROD_OUTPUT,
    },
    BuiltinSignatureDescriptor {
        label: "B = prod(A, vecdim)",
        inputs: &PROD_INPUTS_DIM,
        outputs: &PROD_OUTPUT,
    },
    BuiltinSignatureDescriptor {
        label: "B = prod(A, options...)",
        inputs: &PROD_INPUTS_OPTIONS,
        outputs: &PROD_OUTPUT,
    },
];

pub const PROD_ERROR_INVALID_ARGUMENT: BuiltinErrorDescriptor = BuiltinErrorDescriptor {
    code: "RM.PROD.INVALID_ARGUMENT",
    identifier: Some("RunMat:prod:InvalidArgument"),
    when: "Dimension, missing-data flag, or output class argument grammar is invalid.",
    message: "prod: invalid argument",
};

pub const PROD_ERROR_INVALID_INPUT: BuiltinErrorDescriptor = BuiltinErrorDescriptor {
    code: "RM.PROD.INVALID_INPUT",
    identifier: Some("RunMat:prod:InvalidInput"),
    when: "Input value type is unsupported for prod reduction.",
    message: "prod: invalid input",
};

pub const PROD_ERROR_COMPLEX_UNSUPPORTED: BuiltinErrorDescriptor = BuiltinErrorDescriptor {
    code: "RM.PROD.COMPLEX_UNSUPPORTED",
    identifier: Some("RunMat:prod:ComplexUnsupported"),
    when: "Complex inputs are passed to prod where complex support is not implemented.",
    message: "prod: complex inputs are not yet supported",
};

pub const PROD_ERROR_INTERNAL: BuiltinErrorDescriptor = BuiltinErrorDescriptor {
    code: "RM.PROD.INTERNAL",
    identifier: Some("RunMat:prod:Internal"),
    when:
        "Reduction execution fails due to conversion, provider, allocation, or coercion operations.",
    message: "prod: internal reduction failure",
};

pub const PROD_ERRORS: [BuiltinErrorDescriptor; 5] = [
    PROD_ERROR_INVALID_ARGUMENT,
    PROD_ERROR_INVALID_INPUT,
    PROD_ERROR_COMPLEX_UNSUPPORTED,
    PROD_ERROR_INTERNAL,
    PROD_ERROR_TOO_MANY_OUTPUTS,
];

pub const PROD_DESCRIPTOR: BuiltinDescriptor = BuiltinDescriptor {
    signatures: &PROD_SIGNATURES,
    output_mode: BuiltinOutputMode::Fixed,
    completion_policy: BuiltinCompletionPolicy::Public,
    errors: &PROD_ERRORS,
};

const PROD_INTEGER_INPUTS: [BuiltinIntegerInputCapability; 2] = [
    BuiltinIntegerInputCapability {
        name: "A",
        classes: &crate::ALL_INTEGER_CLASSES,
        availability: BuiltinIntegerInputAvailability::Documented,
        scalar_double: BuiltinIntegerScalarDoubleRule::NotApplicable,
        notes: "Real arrays accept all eight integer storage classes. All complex data, including complex-integer storage, is currently unsupported by prod.",
    },
    BuiltinIntegerInputCapability {
        name: "dim_or_vecdim",
        classes: &crate::ALL_INTEGER_CLASSES,
        availability: BuiltinIntegerInputAvailability::Documented,
        scalar_double: BuiltinIntegerScalarDoubleRule::Allowed,
        notes: "Dimension scalars and vectors explicitly accept every integer class as well as integer-valued floating selectors.",
    },
];

pub const PROD_INTEGER_CAPABILITIES: [BuiltinIntegerCapabilityDescriptor; 2] = [
    BuiltinIntegerCapabilityDescriptor {
        form: "B = prod(A, dim_or_vecdim, nanflag, \"default\"|\"double\")",
        inputs: &PROD_INTEGER_INPUTS,
        computation_domain: BuiltinIntegerComputationDomain::FloatingPoint,
        output_class: BuiltinIntegerOutputClassRule::Double,
        overflow: BuiltinIntegerOverflowRule::NotApplicable,
        backend: BuiltinIntegerBackendRule::GatherFallback,
        overload: BuiltinIntegerOverloadKind::Multiple,
        notes: "Default and explicit-double products convert integer operands before floating multiplication; they never cast an already saturated native product. Current providers have no admitted resident integer-to-floating cast. Automatic placement can use admitted host fallback; explicit gpuArray input is rejected before gathering.",
    },
    BuiltinIntegerCapabilityDescriptor {
        form: "B = prod(A, dim_or_vecdim, nanflag, \"native\")",
        inputs: &PROD_INTEGER_INPUTS,
        computation_domain: BuiltinIntegerComputationDomain::ExactInteger,
        output_class: BuiltinIntegerOutputClassRule::PreserveInput,
        overflow: BuiltinIntegerOverflowRule::Saturate,
        backend: BuiltinIntegerBackendRule::HostAndGpu,
        overload: BuiltinIntegerOverloadKind::Multiple,
        notes: "Native integer products preserve the input class and saturate at its bounds; host and provider paths retain exact integer storage, although documented reduction order can affect signed saturated results.",
    },
];

pub const PROD_ERROR_TOO_MANY_OUTPUTS: BuiltinErrorDescriptor = BuiltinErrorDescriptor {
    code: "RM.PROD.TOO_MANY_OUTPUTS",
    identifier: Some("RunMat:prod:TooManyOutputs"),
    when: "More outputs are requested than this public form returns.",
    message: "prod: too many output arguments",
};
