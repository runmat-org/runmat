use crate::*;
const MEAN_OUTPUT: [BuiltinParamDescriptor; 1] = [BuiltinParamDescriptor {
    name: "M",
    ty: BuiltinParamType::NumericArray,
    arity: BuiltinParamArity::Required,
    default: None,
    description: "Mean reduction result.",
}];

const MEAN_INPUTS_A: [BuiltinParamDescriptor; 1] = [BuiltinParamDescriptor {
    name: "A",
    ty: BuiltinParamType::Any,
    arity: BuiltinParamArity::Required,
    default: None,
    description: "Input array.",
}];

const MEAN_INPUTS_A_DIM: [BuiltinParamDescriptor; 2] = [
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

const MEAN_INPUTS_A_ALL: [BuiltinParamDescriptor; 2] = [
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

const MEAN_INPUTS_A_MISSINGFLAG: [BuiltinParamDescriptor; 2] = [
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

const MEAN_INPUTS_A_OUTTYPE: [BuiltinParamDescriptor; 2] = [
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

const MEAN_INPUTS_A_DIM_MISSINGFLAG: [BuiltinParamDescriptor; 3] = [
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

const MEAN_INPUTS_A_MISSINGFLAG_DIM: [BuiltinParamDescriptor; 3] = [
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

const MEAN_INPUTS_A_WEIGHTS: [BuiltinParamDescriptor; 3] = [
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

const MEAN_SIGNATURES: [BuiltinSignatureDescriptor; 9] = [
    BuiltinSignatureDescriptor {
        label: "M = mean(A)",
        inputs: &MEAN_INPUTS_A,
        outputs: &MEAN_OUTPUT,
    },
    BuiltinSignatureDescriptor {
        label: "M = mean(A, dim)",
        inputs: &MEAN_INPUTS_A_DIM,
        outputs: &MEAN_OUTPUT,
    },
    BuiltinSignatureDescriptor {
        label: "M = mean(A, \"all\")",
        inputs: &MEAN_INPUTS_A_ALL,
        outputs: &MEAN_OUTPUT,
    },
    BuiltinSignatureDescriptor {
        label: "M = mean(A, missingflag)",
        inputs: &MEAN_INPUTS_A_MISSINGFLAG,
        outputs: &MEAN_OUTPUT,
    },
    BuiltinSignatureDescriptor {
        label: "M = mean(A, outtype)",
        inputs: &MEAN_INPUTS_A_OUTTYPE,
        outputs: &MEAN_OUTPUT,
    },
    BuiltinSignatureDescriptor {
        label: "M = mean(A, dim, missingflag)",
        inputs: &MEAN_INPUTS_A_DIM_MISSINGFLAG,
        outputs: &MEAN_OUTPUT,
    },
    BuiltinSignatureDescriptor {
        label: "M = mean(A, missingflag, dim)",
        inputs: &MEAN_INPUTS_A_MISSINGFLAG_DIM,
        outputs: &MEAN_OUTPUT,
    },
    BuiltinSignatureDescriptor {
        label: "M = mean(A, vecdim)",
        inputs: &MEAN_INPUTS_A_DIM,
        outputs: &MEAN_OUTPUT,
    },
    BuiltinSignatureDescriptor {
        label: "M = mean(___, Weights=W)",
        inputs: &MEAN_INPUTS_A_WEIGHTS,
        outputs: &MEAN_OUTPUT,
    },
];

pub const MEAN_ERROR_INVALID_ARGUMENT: BuiltinErrorDescriptor = BuiltinErrorDescriptor {
    code: "RM.MEAN.INVALID_ARGUMENT",
    identifier: Some("RunMat:mean:InvalidArgument"),
    when: "Dimension, missing-data, weights, or output class argument grammar is invalid.",
    message: "mean: invalid argument",
};

pub const MEAN_ERROR_INVALID_INPUT: BuiltinErrorDescriptor = BuiltinErrorDescriptor {
    code: "RM.MEAN.INVALID_INPUT",
    identifier: Some("RunMat:mean:InvalidInput"),
    when: "Input values cannot be converted to supported mean reduction domains.",
    message: "mean: invalid input",
};

pub const MEAN_ERROR_INTERNAL: BuiltinErrorDescriptor = BuiltinErrorDescriptor {
    code: "RM.MEAN.INTERNAL",
    identifier: Some("RunMat:mean:Internal"),
    when:
        "Reduction execution fails due to conversion, provider, allocation, or coercion operations.",
    message: "mean: internal reduction failure",
};

const MEAN_ERRORS: [BuiltinErrorDescriptor; 3] = [
    MEAN_ERROR_INVALID_ARGUMENT,
    MEAN_ERROR_INVALID_INPUT,
    MEAN_ERROR_INTERNAL,
];

pub const MEAN_DESCRIPTOR: BuiltinDescriptor = BuiltinDescriptor {
    signatures: &MEAN_SIGNATURES,
    output_mode: BuiltinOutputMode::Fixed,
    completion_policy: BuiltinCompletionPolicy::Public,
    errors: &MEAN_ERRORS,
};

const INTEGER_DATA_AND_DIM_INPUTS: [BuiltinIntegerInputCapability; 2] = [
    BuiltinIntegerInputCapability {
        name: "A",
        classes: &crate::ALL_INTEGER_CLASSES,
        availability: BuiltinIntegerInputAvailability::Documented,
        scalar_double: BuiltinIntegerScalarDoubleRule::NotApplicable,
        notes: "Ordinary real arrays accept all eight integer storage classes.",
    },
    BuiltinIntegerInputCapability {
        name: "dim_or_vecdim",
        classes: &crate::ALL_INTEGER_CLASSES,
        availability: BuiltinIntegerInputAvailability::Documented,
        scalar_double: BuiltinIntegerScalarDoubleRule::Allowed,
        notes: "Optional dimensions accept exact positive integer scalars or vectors in every integer class; integer-valued double selectors remain allowed.",
    },
];

const WEIGHTED_INTEGER_INPUTS: [BuiltinIntegerInputCapability; 3] = [
    BuiltinIntegerInputCapability {
        name: "A",
        classes: &crate::ALL_INTEGER_CLASSES,
        availability: BuiltinIntegerInputAvailability::Documented,
        scalar_double: BuiltinIntegerScalarDoubleRule::NotApplicable,
        notes: "Weighted reduction retains the ordinary all-class integer data contract.",
    },
    BuiltinIntegerInputCapability {
        name: "dim",
        classes: &crate::ALL_INTEGER_CLASSES,
        availability: BuiltinIntegerInputAvailability::Documented,
        scalar_double: BuiltinIntegerScalarDoubleRule::Allowed,
        notes: "The optional weighted-reduction dimension accepts every integer class and integer-valued double; weights cannot be combined with vecdim or all.",
    },
    BuiltinIntegerInputCapability {
        name: "W",
        classes: &[],
        availability: BuiltinIntegerInputAvailability::Rejected,
        scalar_double: BuiltinIntegerScalarDoubleRule::NotApplicable,
        notes: "Weights are restricted to nonnegative single or double arrays; integer and logical weights are rejected before reduction.",
    },
];

pub const MEAN_INTEGER_CAPABILITIES: [BuiltinIntegerCapabilityDescriptor; 4] = [
    BuiltinIntegerCapabilityDescriptor {
        form: "M = mean(A, dim_or_vecdim, missingflag, \"default\"|\"double\")",
        inputs: &INTEGER_DATA_AND_DIM_INPUTS,
        computation_domain: BuiltinIntegerComputationDomain::FloatingPoint,
        output_class: BuiltinIntegerOutputClassRule::Double,
        overflow: BuiltinIntegerOverflowRule::NotApplicable,
        backend: BuiltinIntegerBackendRule::HostAndGpu,
        overload: BuiltinIntegerOverloadKind::Multiple,
        notes: "Default and explicit-double integer means enter the floating reduction domain and return double; all and omitted dimensions share this rule, and supported resident inputs remain resident.",
    },
    BuiltinIntegerCapabilityDescriptor {
        form: "M = mean(A, dim_or_vecdim, missingflag, \"native\")",
        inputs: &INTEGER_DATA_AND_DIM_INPUTS,
        computation_domain: BuiltinIntegerComputationDomain::ExactInteger,
        output_class: BuiltinIntegerOutputClassRule::PreserveInput,
        overflow: BuiltinIntegerOverflowRule::NotApplicable,
        backend: BuiltinIntegerBackendRule::HostAndGpu,
        overload: BuiltinIntegerOverloadKind::Multiple,
        notes: "Native integer mean widens the full accumulation and performs one nearest/ties-away division; the mean remains within the input class range without intermediate saturation, and CPU/provider paths preserve all eight classes.",
    },
    BuiltinIntegerCapabilityDescriptor {
        form: "M = mean(A, dim, missingflag, Weights=W, \"default\"|\"double\")",
        inputs: &WEIGHTED_INTEGER_INPUTS,
        computation_domain: BuiltinIntegerComputationDomain::FloatingPoint,
        output_class: BuiltinIntegerOutputClassRule::Double,
        overflow: BuiltinIntegerOverflowRule::NotApplicable,
        backend: BuiltinIntegerBackendRule::HostAndGpu,
        overload: BuiltinIntegerOverloadKind::Multiple,
        notes: "Integer data with floating weights returns double by default; integer weights are rejected, and resident fallback output is re-uploaded.",
    },
    BuiltinIntegerCapabilityDescriptor {
        form: "M = mean(A, dim, missingflag, Weights=W, \"native\")",
        inputs: &WEIGHTED_INTEGER_INPUTS,
        computation_domain: BuiltinIntegerComputationDomain::FunctionSpecific,
        output_class: BuiltinIntegerOutputClassRule::PreserveInput,
        overflow: BuiltinIntegerOverflowRule::NotApplicable,
        backend: BuiltinIntegerBackendRule::GatherFallback,
        overload: BuiltinIntegerOverloadKind::Multiple,
        notes: "Native weighted integer mean anchors exact wide integer values, applies nonnegative floating weights, rounds once with ties away from zero, remains within the input class range, and re-uploads gathered resident results.",
    },
];
