use crate::*;
const SUM_OUTPUT: [BuiltinParamDescriptor; 1] = [BuiltinParamDescriptor {
    name: "S",
    ty: BuiltinParamType::NumericArray,
    arity: BuiltinParamArity::Required,
    default: None,
    description: "Sum reduction result.",
}];

const SUM_INPUTS_A: [BuiltinParamDescriptor; 1] = [BuiltinParamDescriptor {
    name: "A",
    ty: BuiltinParamType::Any,
    arity: BuiltinParamArity::Required,
    default: None,
    description: "Input array.",
}];

const SUM_INPUTS_A_DIM: [BuiltinParamDescriptor; 2] = [
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
        default: Some("[]"),
        description: "Dimension selector or vector of dimensions.",
    },
];

const SUM_INPUTS_A_ALL: [BuiltinParamDescriptor; 2] = [
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

const SUM_INPUTS_A_MISSINGFLAG: [BuiltinParamDescriptor; 2] = [
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

const SUM_INPUTS_A_OUTTYPE: [BuiltinParamDescriptor; 2] = [
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

const SUM_INPUTS_A_DIM_MISSINGFLAG: [BuiltinParamDescriptor; 3] = [
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
        default: Some("[]"),
        description: "Dimension selector or vector of dimensions.",
    },
    BuiltinParamDescriptor {
        name: "missingflag",
        ty: BuiltinParamType::StringScalar,
        arity: BuiltinParamArity::Optional,
        default: Some("\"includemissing\""),
        description: "Missing-data handling mode: \"includemissing\" or \"omitmissing\".",
    },
];

const SUM_INPUTS_A_MISSINGFLAG_DIM: [BuiltinParamDescriptor; 3] = [
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
        default: Some("[]"),
        description: "Dimension selector or vector of dimensions.",
    },
];

const SUM_INPUTS_VARIADIC: [BuiltinParamDescriptor; 2] = [
    BuiltinParamDescriptor {
        name: "A",
        ty: BuiltinParamType::Any,
        arity: BuiltinParamArity::Required,
        default: None,
        description: "Input array.",
    },
    BuiltinParamDescriptor {
        name: "options",
        ty: BuiltinParamType::Any,
        arity: BuiltinParamArity::Variadic,
        default: None,
        description: "Admitted dimension, missing-data, and output-class controls.",
    },
];

const SUM_SIGNATURES: [BuiltinSignatureDescriptor; 9] = [
    BuiltinSignatureDescriptor {
        label: "S = sum(A)",
        inputs: &SUM_INPUTS_A,
        outputs: &SUM_OUTPUT,
    },
    BuiltinSignatureDescriptor {
        label: "S = sum(A, dim)",
        inputs: &SUM_INPUTS_A_DIM,
        outputs: &SUM_OUTPUT,
    },
    BuiltinSignatureDescriptor {
        label: "S = sum(A, \"all\")",
        inputs: &SUM_INPUTS_A_ALL,
        outputs: &SUM_OUTPUT,
    },
    BuiltinSignatureDescriptor {
        label: "S = sum(A, missingflag)",
        inputs: &SUM_INPUTS_A_MISSINGFLAG,
        outputs: &SUM_OUTPUT,
    },
    BuiltinSignatureDescriptor {
        label: "S = sum(A, outtype)",
        inputs: &SUM_INPUTS_A_OUTTYPE,
        outputs: &SUM_OUTPUT,
    },
    BuiltinSignatureDescriptor {
        label: "S = sum(A, dim, missingflag)",
        inputs: &SUM_INPUTS_A_DIM_MISSINGFLAG,
        outputs: &SUM_OUTPUT,
    },
    BuiltinSignatureDescriptor {
        label: "S = sum(A, missingflag, dim)",
        inputs: &SUM_INPUTS_A_MISSINGFLAG_DIM,
        outputs: &SUM_OUTPUT,
    },
    BuiltinSignatureDescriptor {
        label: "S = sum(A, vecdim)",
        inputs: &SUM_INPUTS_A_DIM,
        outputs: &SUM_OUTPUT,
    },
    BuiltinSignatureDescriptor {
        label: "S = sum(A, options...)",
        inputs: &SUM_INPUTS_VARIADIC,
        outputs: &SUM_OUTPUT,
    },
];

pub const SUM_ERROR_INVALID_ARGUMENT: BuiltinErrorDescriptor = BuiltinErrorDescriptor {
    code: "RM.SUM.INVALID_ARGUMENT",
    identifier: Some("RunMat:sum:InvalidArgument"),
    when: "Dimension, missing-data flag, or output class argument grammar is invalid.",
    message: "sum: invalid argument",
};

pub const SUM_ERROR_INVALID_INPUT: BuiltinErrorDescriptor = BuiltinErrorDescriptor {
    code: "RM.SUM.INVALID_INPUT",
    identifier: Some("RunMat:sum:InvalidInput"),
    when: "Input values cannot be converted to supported sum reduction domains.",
    message: "sum: invalid input",
};

pub const SUM_ERROR_INTERNAL: BuiltinErrorDescriptor = BuiltinErrorDescriptor {
    code: "RM.SUM.INTERNAL",
    identifier: Some("RunMat:sum:Internal"),
    when:
        "Reduction execution fails due to conversion, provider, allocation, or coercion operations.",
    message: "sum: internal reduction failure",
};

const SUM_ERRORS: [BuiltinErrorDescriptor; 3] = [
    SUM_ERROR_INVALID_ARGUMENT,
    SUM_ERROR_INVALID_INPUT,
    SUM_ERROR_INTERNAL,
];

pub const SUM_DESCRIPTOR: BuiltinDescriptor = BuiltinDescriptor {
    signatures: &SUM_SIGNATURES,
    output_mode: BuiltinOutputMode::Fixed,
    completion_policy: BuiltinCompletionPolicy::Public,
    errors: &SUM_ERRORS,
};

const INTEGER_INPUTS: [BuiltinIntegerInputCapability; 2] = [
    BuiltinIntegerInputCapability {
        name: "A",
        classes: &crate::ALL_INTEGER_CLASSES,
        availability: BuiltinIntegerInputAvailability::Documented,
        scalar_double: BuiltinIntegerScalarDoubleRule::NotApplicable,
        notes: "Ordinary real arrays accept all eight integer storage classes; complex-integer reduction remains a separately tracked conformance question.",
    },
    BuiltinIntegerInputCapability {
        name: "dim_or_vecdim",
        classes: &crate::ALL_INTEGER_CLASSES,
        availability: BuiltinIntegerInputAvailability::Documented,
        scalar_double: BuiltinIntegerScalarDoubleRule::Allowed,
        notes: "Dimension scalars and vectors explicitly accept every integer class as well as integer-valued floating selectors.",
    },
];

pub const SUM_INTEGER_CAPABILITIES: [BuiltinIntegerCapabilityDescriptor; 2] = [
    BuiltinIntegerCapabilityDescriptor {
        form: "S = sum(A, dim_or_vecdim, nanflag, \"default\"|\"double\")",
        inputs: &INTEGER_INPUTS,
        computation_domain: BuiltinIntegerComputationDomain::FloatingPoint,
        output_class: BuiltinIntegerOutputClassRule::Double,
        overflow: BuiltinIntegerOverflowRule::NotApplicable,
        backend: BuiltinIntegerBackendRule::HostAndGpu,
        overload: BuiltinIntegerOverloadKind::Multiple,
        notes: "Default and explicit-double integer sums compute and return double; omitted dimensions and all share the same rule. Providers with a floating integer reduction keep the calculation resident; other providers use the exact-owner gather and class-preserving restoration policy.",
    },
    BuiltinIntegerCapabilityDescriptor {
        form: "S = sum(A, dim_or_vecdim, nanflag, \"native\")",
        inputs: &INTEGER_INPUTS,
        computation_domain: BuiltinIntegerComputationDomain::ExactInteger,
        output_class: BuiltinIntegerOutputClassRule::PreserveInput,
        overflow: BuiltinIntegerOverflowRule::Saturate,
        backend: BuiltinIntegerBackendRule::HostAndGpu,
        overload: BuiltinIntegerOverloadKind::Multiple,
        notes: "Native integer sums preserve the input class and saturate at its bounds; host and provider paths retain exact integer storage, although documented reduction order can affect signed saturated results.",
    },
];
