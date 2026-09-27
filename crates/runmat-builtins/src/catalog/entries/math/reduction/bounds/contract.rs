use crate::*;
const OUTPUT_MIN: BuiltinParamDescriptor = BuiltinParamDescriptor {
    name: "S",
    ty: BuiltinParamType::NumericArray,
    arity: BuiltinParamArity::Required,
    default: None,
    description: "Smallest values.",
};

const OUTPUT_MAX: BuiltinParamDescriptor = BuiltinParamDescriptor {
    name: "L",
    ty: BuiltinParamType::NumericArray,
    arity: BuiltinParamArity::Required,
    default: None,
    description: "Largest values.",
};

const OUTPUTS: [BuiltinParamDescriptor; 2] = [OUTPUT_MIN, OUTPUT_MAX];

const INPUT_A: BuiltinParamDescriptor = BuiltinParamDescriptor {
    name: "A",
    ty: BuiltinParamType::Any,
    arity: BuiltinParamArity::Required,
    default: None,
    description: "Input array.",
};

const INPUT_DIM: BuiltinParamDescriptor = BuiltinParamDescriptor {
    name: "dim",
    ty: BuiltinParamType::Any,
    arity: BuiltinParamArity::Optional,
    default: None,
    description: "Dimension selector, vector of dimensions, or \"all\".",
};

pub const BOUNDS_INPUT_NANFLAG: BuiltinParamDescriptor = BuiltinParamDescriptor {
    name: "missingflag",
    ty: BuiltinParamType::StringScalar,
    arity: BuiltinParamArity::Optional,
    default: Some("\"omitmissing\""),
    description: "Missing-value handling mode.",
};

const INPUTS_A: [BuiltinParamDescriptor; 1] = [INPUT_A];
const INPUTS_A_DIM: [BuiltinParamDescriptor; 2] = [INPUT_A, INPUT_DIM];
const INPUTS_A_NANFLAG: [BuiltinParamDescriptor; 2] = [INPUT_A, BOUNDS_INPUT_NANFLAG];
const INPUTS_A_DIM_NANFLAG: [BuiltinParamDescriptor; 3] =
    [INPUT_A, INPUT_DIM, BOUNDS_INPUT_NANFLAG];

const BOUNDS_SIGNATURES: [BuiltinSignatureDescriptor; 5] = [
    BuiltinSignatureDescriptor {
        label: "[S, L] = bounds(A)",
        inputs: &INPUTS_A,
        outputs: &OUTPUTS,
    },
    BuiltinSignatureDescriptor {
        label: "[S, L] = bounds(A, dim)",
        inputs: &INPUTS_A_DIM,
        outputs: &OUTPUTS,
    },
    BuiltinSignatureDescriptor {
        label: "[S, L] = bounds(A, vecdim)",
        inputs: &INPUTS_A_DIM,
        outputs: &OUTPUTS,
    },
    BuiltinSignatureDescriptor {
        label: "[S, L] = bounds(A, missingflag)",
        inputs: &INPUTS_A_NANFLAG,
        outputs: &OUTPUTS,
    },
    BuiltinSignatureDescriptor {
        label: "[S, L] = bounds(A, dim, missingflag)",
        inputs: &INPUTS_A_DIM_NANFLAG,
        outputs: &OUTPUTS,
    },
];

pub const BOUNDS_ERROR_INVALID_ARGUMENT: BuiltinErrorDescriptor = BuiltinErrorDescriptor {
    code: "RM.BOUNDS.INVALID_ARGUMENT",
    identifier: Some("RunMat:bounds:InvalidArgument"),
    when: "Dimension selectors, missing flags, or argument ordering are invalid.",
    message: "bounds: invalid argument",
};

pub const BOUNDS_ERROR_INTERNAL: BuiltinErrorDescriptor = BuiltinErrorDescriptor {
    code: "RM.BOUNDS.INTERNAL",
    identifier: Some("RunMat:bounds:Internal"),
    when: "Min/max evaluation fails internally.",
    message: "bounds: internal failure",
};

pub const BOUNDS_ERRORS: [BuiltinErrorDescriptor; 2] =
    [BOUNDS_ERROR_INVALID_ARGUMENT, BOUNDS_ERROR_INTERNAL];

pub const BOUNDS_DESCRIPTOR: BuiltinDescriptor = BuiltinDescriptor {
    signatures: &BOUNDS_SIGNATURES,
    output_mode: BuiltinOutputMode::ByRequestedOutputCount,
    completion_policy: BuiltinCompletionPolicy::Public,
    errors: &BOUNDS_ERRORS,
};

const INTEGER_INPUTS: [BuiltinIntegerInputCapability; 2] = [
    BuiltinIntegerInputCapability {
        name: "A",
        classes: &ALL_INTEGER_CLASSES,
        availability: BuiltinIntegerInputAvailability::Documented,
        scalar_double: BuiltinIntegerScalarDoubleRule::NotApplicable,
        notes: "Ordinary real arrays accept all eight integer storage classes; paired complex-integer storage is rejected before provider access, including empty input.",
    },
    BuiltinIntegerInputCapability {
        name: "dim_or_vecdim",
        classes: &ALL_INTEGER_CLASSES,
        availability: BuiltinIntegerInputAvailability::Documented,
        scalar_double: BuiltinIntegerScalarDoubleRule::Allowed,
        notes: "Positive integer dimension selectors are decoded exactly from typed integer or integer-valued floating storage.",
    },
];

pub const BOUNDS_INTEGER_CAPABILITIES: [BuiltinIntegerCapabilityDescriptor; 1] =
    [BuiltinIntegerCapabilityDescriptor {
        form: "[S, L] = bounds(A, dim_or_vecdim, missingflag)",
        inputs: &INTEGER_INPUTS,
        computation_domain: BuiltinIntegerComputationDomain::ExactInteger,
        output_class: BuiltinIntegerOutputClassRule::PreserveInput,
        overflow: BuiltinIntegerOverflowRule::NotApplicable,
        backend: BuiltinIntegerBackendRule::GatherFallback,
        overload: BuiltinIntegerOverloadKind::Multiple,
        notes: "Both extrema preserve the integer input class; resident inputs currently gather through the shared min/max implementation and both exact results are re-uploaded.",
    }];

pub const BOUNDS_EXTRA_OUTPUTS_EXTENSION: BuiltinExtensionDescriptor = BuiltinExtensionDescriptor {
    id: "bounds-extra-outputs",
    mode: BuiltinExtensionMode::RunMatOnly,
    description: "bounds pads outputs beyond its two documented results with scalar double zero",
    error_identifier: Some("RunMat:compatibility:BoundsExtraOutputsExtension"),
};
pub const BOUNDS_EXTENSIONS: [BuiltinExtensionDescriptor; 1] = [BOUNDS_EXTRA_OUTPUTS_EXTENSION];
