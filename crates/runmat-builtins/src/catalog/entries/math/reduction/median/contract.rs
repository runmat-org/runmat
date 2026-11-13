use crate::*;
const MEDIAN_OUTPUT: [BuiltinParamDescriptor; 1] = [BuiltinParamDescriptor {
    name: "M",
    ty: BuiltinParamType::NumericArray,
    arity: BuiltinParamArity::Required,
    default: None,
    description: "Median reduction result.",
}];

const MEDIAN_INPUTS_A: [BuiltinParamDescriptor; 1] = [BuiltinParamDescriptor {
    name: "A",
    ty: BuiltinParamType::Any,
    arity: BuiltinParamArity::Required,
    default: None,
    description: "Input array.",
}];

const MEDIAN_INPUTS_A_AXES: [BuiltinParamDescriptor; 2] = [
    BuiltinParamDescriptor {
        name: "A",
        ty: BuiltinParamType::Any,
        arity: BuiltinParamArity::Required,
        default: None,
        description: "Input array.",
    },
    BuiltinParamDescriptor {
        name: "axes",
        ty: BuiltinParamType::Any,
        arity: BuiltinParamArity::Optional,
        default: Some("[]"),
        description: "Dimension selector, vector of dimensions, or \"all\".",
    },
];

const MEDIAN_INPUTS_A_NANFLAG: [BuiltinParamDescriptor; 2] = [
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
        description: "Missing-value handling mode.",
    },
];

const MEDIAN_INPUTS_A_AXES_NANFLAG: [BuiltinParamDescriptor; 3] = [
    BuiltinParamDescriptor {
        name: "A",
        ty: BuiltinParamType::Any,
        arity: BuiltinParamArity::Required,
        default: None,
        description: "Input array.",
    },
    BuiltinParamDescriptor {
        name: "axes",
        ty: BuiltinParamType::Any,
        arity: BuiltinParamArity::Optional,
        default: Some("[]"),
        description: "Dimension selector, vector of dimensions, or \"all\".",
    },
    BuiltinParamDescriptor {
        name: "missingflag",
        ty: BuiltinParamType::StringScalar,
        arity: BuiltinParamArity::Optional,
        default: Some("\"includemissing\""),
        description: "Missing-value handling mode.",
    },
];

const MEDIAN_INPUTS_A_NANFLAG_AXES: [BuiltinParamDescriptor; 3] = [
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
        description: "Missing-value handling mode.",
    },
    BuiltinParamDescriptor {
        name: "axes",
        ty: BuiltinParamType::Any,
        arity: BuiltinParamArity::Optional,
        default: Some("[]"),
        description: "Dimension selector, vector of dimensions, or \"all\".",
    },
];

const MEDIAN_INPUTS_A_WEIGHTS: [BuiltinParamDescriptor; 3] = [
    BuiltinParamDescriptor {
        name: "A",
        ty: BuiltinParamType::Any,
        arity: BuiltinParamArity::Required,
        default: None,
        description: "Input array.",
    },
    BuiltinParamDescriptor {
        name: "Weights",
        ty: BuiltinParamType::StringScalar,
        arity: BuiltinParamArity::Required,
        default: Some("\"Weights\""),
        description: "Weighting-scheme name.",
    },
    BuiltinParamDescriptor {
        name: "W",
        ty: BuiltinParamType::NumericArray,
        arity: BuiltinParamArity::Required,
        default: None,
        description: "Nonnegative single- or double-precision weighting scheme.",
    },
];

const MEDIAN_SIGNATURES: [BuiltinSignatureDescriptor; 10] = [
    BuiltinSignatureDescriptor {
        label: "M = median(A)",
        inputs: &MEDIAN_INPUTS_A,
        outputs: &MEDIAN_OUTPUT,
    },
    BuiltinSignatureDescriptor {
        label: "M = median(A, dim)",
        inputs: &MEDIAN_INPUTS_A_AXES,
        outputs: &MEDIAN_OUTPUT,
    },
    BuiltinSignatureDescriptor {
        label: "M = median(A, vecdim)",
        inputs: &MEDIAN_INPUTS_A_AXES,
        outputs: &MEDIAN_OUTPUT,
    },
    BuiltinSignatureDescriptor {
        label: "M = median(A, \"all\")",
        inputs: &MEDIAN_INPUTS_A_AXES,
        outputs: &MEDIAN_OUTPUT,
    },
    BuiltinSignatureDescriptor {
        label: "M = median(A, [])",
        inputs: &MEDIAN_INPUTS_A_AXES,
        outputs: &MEDIAN_OUTPUT,
    },
    BuiltinSignatureDescriptor {
        label: "M = median(A, missingflag)",
        inputs: &MEDIAN_INPUTS_A_NANFLAG,
        outputs: &MEDIAN_OUTPUT,
    },
    BuiltinSignatureDescriptor {
        label: "M = median(A, axes, missingflag)",
        inputs: &MEDIAN_INPUTS_A_AXES_NANFLAG,
        outputs: &MEDIAN_OUTPUT,
    },
    BuiltinSignatureDescriptor {
        label: "M = median(A, missingflag, axes)",
        inputs: &MEDIAN_INPUTS_A_NANFLAG_AXES,
        outputs: &MEDIAN_OUTPUT,
    },
    BuiltinSignatureDescriptor {
        label: "M = median(A, missingflag, \"all\")",
        inputs: &MEDIAN_INPUTS_A_NANFLAG_AXES,
        outputs: &MEDIAN_OUTPUT,
    },
    BuiltinSignatureDescriptor {
        label: "M = median(___, Weights=W)",
        inputs: &MEDIAN_INPUTS_A_WEIGHTS,
        outputs: &MEDIAN_OUTPUT,
    },
];

pub const MEDIAN_ERROR_INVALID_ARGUMENT: BuiltinErrorDescriptor = BuiltinErrorDescriptor {
    code: "RM.MEDIAN.INVALID_ARGUMENT",
    identifier: Some("RunMat:median:InvalidArgument"),
    when: "Dimension selectors, missing flags, or argument ordering are invalid.",
    message: "median: invalid argument",
};

pub const MEDIAN_ERROR_INVALID_INPUT: BuiltinErrorDescriptor = BuiltinErrorDescriptor {
    code: "RM.MEDIAN.INVALID_INPUT",
    identifier: Some("RunMat:median:InvalidInput"),
    when: "Input values cannot be converted to supported median reduction domains.",
    message: "median: invalid input",
};

pub const MEDIAN_ERROR_INTERNAL: BuiltinErrorDescriptor = BuiltinErrorDescriptor {
    code: "RM.MEDIAN.INTERNAL",
    identifier: Some("RunMat:median:Internal"),
    when: "Median reduction fails due to tensor allocation or device fallback internals.",
    message: "median: internal reduction failure",
};

const MEDIAN_ERRORS: [BuiltinErrorDescriptor; 3] = [
    MEDIAN_ERROR_INVALID_ARGUMENT,
    MEDIAN_ERROR_INVALID_INPUT,
    MEDIAN_ERROR_INTERNAL,
];

pub const MEDIAN_DESCRIPTOR: BuiltinDescriptor = BuiltinDescriptor {
    signatures: &MEDIAN_SIGNATURES,
    output_mode: BuiltinOutputMode::Fixed,
    completion_policy: BuiltinCompletionPolicy::Public,
    errors: &MEDIAN_ERRORS,
};

const INTEGER_DATA_AND_DIM_INPUTS: [BuiltinIntegerInputCapability; 2] = [
    BuiltinIntegerInputCapability {
        name: "A",
        classes: &crate::ALL_INTEGER_CLASSES,
        availability: BuiltinIntegerInputAvailability::Documented,
        scalar_double: BuiltinIntegerScalarDoubleRule::NotApplicable,
        notes: "Numeric median data explicitly accepts all eight integer classes.",
    },
    BuiltinIntegerInputCapability {
        name: "dim_or_vecdim",
        classes: &crate::ALL_INTEGER_CLASSES,
        availability: BuiltinIntegerInputAvailability::Documented,
        scalar_double: BuiltinIntegerScalarDoubleRule::Allowed,
        notes: "Positive integer dimension selectors are decoded exactly from typed integer or integer-valued floating storage.",
    },
];

const WEIGHTED_INTEGER_INPUTS: [BuiltinIntegerInputCapability; 3] = [
    BuiltinIntegerInputCapability {
        name: "A",
        classes: &crate::ALL_INTEGER_CLASSES,
        availability: BuiltinIntegerInputAvailability::Documented,
        scalar_double: BuiltinIntegerScalarDoubleRule::NotApplicable,
        notes: "Weighted median retains the documented all-class integer data contract.",
    },
    BuiltinIntegerInputCapability {
        name: "dim",
        classes: &crate::ALL_INTEGER_CLASSES,
        availability: BuiltinIntegerInputAvailability::Documented,
        scalar_double: BuiltinIntegerScalarDoubleRule::Allowed,
        notes: "Weighted median permits one operating dimension but not vecdim or all.",
    },
    BuiltinIntegerInputCapability {
        name: "W",
        classes: &[],
        availability: BuiltinIntegerInputAvailability::Rejected,
        scalar_double: BuiltinIntegerScalarDoubleRule::NotApplicable,
        notes: "Weights are documented and enforced as nonnegative single or double arrays; integer and logical weights are rejected.",
    },
];

pub const MEDIAN_INTEGER_CAPABILITIES: [BuiltinIntegerCapabilityDescriptor; 2] = [
    BuiltinIntegerCapabilityDescriptor {
        form: "M = median(A, dim_or_vecdim, missingflag)",
        inputs: &INTEGER_DATA_AND_DIM_INPUTS,
        computation_domain: BuiltinIntegerComputationDomain::ExactInteger,
        output_class: BuiltinIntegerOutputClassRule::PreserveInput,
        overflow: BuiltinIntegerOverflowRule::NotApplicable,
        backend: BuiltinIntegerBackendRule::GatherFallback,
        overload: BuiltinIntegerOverloadKind::Multiple,
        notes: "Integer median preserves the input class; even-cardinality midpoint conversion rounds nearest with ties away from zero, and resident integer order statistics gather then re-upload exactly.",
    },
    BuiltinIntegerCapabilityDescriptor {
        form: "M = median(A, dim, missingflag, Weights=W)",
        inputs: &WEIGHTED_INTEGER_INPUTS,
        computation_domain: BuiltinIntegerComputationDomain::FunctionSpecific,
        output_class: BuiltinIntegerOutputClassRule::PreserveInput,
        overflow: BuiltinIntegerOverflowRule::NotApplicable,
        backend: BuiltinIntegerBackendRule::GatherFallback,
        overload: BuiltinIntegerOverloadKind::Multiple,
        notes: "Weighted integer median selects an input-class value at cumulative 50 percent using floating weights; resident input and weights may gather, and the result is re-uploaded with its exact integer class.",
    },
];
