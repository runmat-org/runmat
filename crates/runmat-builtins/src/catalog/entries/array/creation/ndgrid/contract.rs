//! Canonical ndgrid signatures, errors, extensions, and integer forms.
use crate::*;

const NDGRID_OUTPUTS: [BuiltinParamDescriptor; 3] = [
    BuiltinParamDescriptor {
        name: "X1",
        ty: BuiltinParamType::NumericArray,
        arity: BuiltinParamArity::Required,
        default: None,
        description: "Full grid coordinates for the first dimension.",
    },
    BuiltinParamDescriptor {
        name: "X2",
        ty: BuiltinParamType::NumericArray,
        arity: BuiltinParamArity::Optional,
        default: None,
        description: "Full grid coordinates for the second dimension.",
    },
    BuiltinParamDescriptor {
        name: "Xn",
        ty: BuiltinParamType::NumericArray,
        arity: BuiltinParamArity::Optional,
        default: None,
        description: "Full grid coordinates for the nth dimension.",
    },
];

const NDGRID_SINGLE_INPUT: [BuiltinParamDescriptor; 1] = [BuiltinParamDescriptor {
    name: "xg",
    ty: BuiltinParamType::NumericArray,
    arity: BuiltinParamArity::Required,
    default: None,
    description: "Grid vector reused for every requested output dimension.",
}];

const NDGRID_MULTI_INPUT: [BuiltinParamDescriptor; 3] = [
    BuiltinParamDescriptor {
        name: "x1",
        ty: BuiltinParamType::NumericArray,
        arity: BuiltinParamArity::Required,
        default: None,
        description: "Grid vector for the first dimension.",
    },
    BuiltinParamDescriptor {
        name: "x2",
        ty: BuiltinParamType::NumericArray,
        arity: BuiltinParamArity::Optional,
        default: None,
        description: "Grid vector for the second dimension.",
    },
    BuiltinParamDescriptor {
        name: "xn",
        ty: BuiltinParamType::NumericArray,
        arity: BuiltinParamArity::Optional,
        default: None,
        description: "Grid vector for the nth dimension.",
    },
];

const NDGRID_SIGNATURES: [BuiltinSignatureDescriptor; 2] = [
    BuiltinSignatureDescriptor {
        label: "[X1, X2, ..., Xn] = ndgrid(xg)",
        inputs: &NDGRID_SINGLE_INPUT,
        outputs: &NDGRID_OUTPUTS,
    },
    BuiltinSignatureDescriptor {
        label: "[X1, X2, ..., Xn] = ndgrid(x1, x2, ..., xn)",
        inputs: &NDGRID_MULTI_INPUT,
        outputs: &NDGRID_OUTPUTS,
    },
];

const NDGRID_INTEGER_INPUTS: [BuiltinIntegerInputCapability; 1] =
    [BuiltinIntegerInputCapability {
        name: "x1, x2, ..., xn",
        classes: &ALL_INTEGER_CLASSES,
        availability: BuiltinIntegerInputAvailability::Documented,
        scalar_double: BuiltinIntegerScalarDoubleRule::NotApplicable,
        notes: "Every documented integer grid-vector class is replicated exactly and each corresponding output preserves that input class.",
    }];

pub const NDGRID_INTEGER_CAPABILITIES: [BuiltinIntegerCapabilityDescriptor; 1] =
    [BuiltinIntegerCapabilityDescriptor {
        form: "[X1,...,Xn] = ndgrid(integer_x1, ..., integer_xn)",
        inputs: &NDGRID_INTEGER_INPUTS,
        computation_domain: BuiltinIntegerComputationDomain::ExactInteger,
        output_class: BuiltinIntegerOutputClassRule::PreserveInput,
        overflow: BuiltinIntegerOverflowRule::NotApplicable,
        backend: BuiltinIntegerBackendRule::GpuRestricted,
        overload: BuiltinIntegerOverloadKind::Multiple,
        notes: "Host axes preserve authoritative native storage independently. MATLAB documents gpuArray axes only for single and double; explicit integer gpuArray axes reject, while automatic integer residency gathers transparently to exact host grids.",
    }];

pub const NDGRID_ERROR_MISSING_AXIS: BuiltinErrorDescriptor = BuiltinErrorDescriptor {
    code: "RM.NDGRID.MISSING_AXIS",
    identifier: Some("RunMat:ndgrid:MissingAxis"),
    when: "No grid vector inputs are provided.",
    message: "ndgrid: at least one input vector is required",
};

pub const NDGRID_ERROR_INVALID_AXIS: BuiltinErrorDescriptor = BuiltinErrorDescriptor {
    code: "RM.NDGRID.INVALID_AXIS",
    identifier: Some("RunMat:ndgrid:InvalidAxis"),
    when: "An input is not numeric vector data.",
    message: "ndgrid: inputs must be numeric vectors",
};

pub const NDGRID_ERROR_TOO_MANY_OUTPUTS: BuiltinErrorDescriptor = BuiltinErrorDescriptor {
    code: "RM.NDGRID.TOO_MANY_OUTPUTS",
    identifier: Some("RunMat:ndgrid:TooManyOutputs"),
    when: "More outputs are requested than available input grid vectors.",
    message: "ndgrid: too many output arguments for the supplied grid vectors",
};

pub const NDGRID_ERROR_INTERNAL: BuiltinErrorDescriptor = BuiltinErrorDescriptor {
    code: "RM.NDGRID.INTERNAL",
    identifier: Some("RunMat:ndgrid:Internal"),
    when: "Output allocation, GPU gather, or GPU upload fails.",
    message: "ndgrid: internal error",
};

const NDGRID_ERRORS: [BuiltinErrorDescriptor; 4] = [
    NDGRID_ERROR_MISSING_AXIS,
    NDGRID_ERROR_INVALID_AXIS,
    NDGRID_ERROR_TOO_MANY_OUTPUTS,
    NDGRID_ERROR_INTERNAL,
];

pub const NDGRID_DESCRIPTOR: BuiltinDescriptor = BuiltinDescriptor {
    signatures: &NDGRID_SIGNATURES,
    output_mode: BuiltinOutputMode::ByRequestedOutputCount,
    completion_policy: BuiltinCompletionPolicy::Public,
    errors: &NDGRID_ERRORS,
};
