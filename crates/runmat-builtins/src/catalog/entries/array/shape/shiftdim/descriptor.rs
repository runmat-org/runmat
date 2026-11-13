use crate::*;

const INPUT_A: BuiltinParamDescriptor = BuiltinParamDescriptor {
    name: "A",
    ty: BuiltinParamType::Any,
    arity: BuiltinParamArity::Required,
    default: None,
    description: "Input array or scalar.",
};
const INPUT_N: BuiltinParamDescriptor = BuiltinParamDescriptor {
    name: "n",
    ty: BuiltinParamType::NumericScalar,
    arity: BuiltinParamArity::Required,
    default: None,
    description: "Finite real integer dimension shift.",
};
const OUTPUT_B: BuiltinParamDescriptor = BuiltinParamDescriptor {
    name: "B",
    ty: BuiltinParamType::Any,
    arity: BuiltinParamArity::Required,
    default: None,
    description: "Array with dimensions shifted and native class preserved.",
};
const OUTPUT_M: BuiltinParamDescriptor = BuiltinParamDescriptor {
    name: "m",
    ty: BuiltinParamType::NumericScalar,
    arity: BuiltinParamArity::Required,
    default: None,
    description: "Number of leading singleton dimensions removed.",
};
const INPUTS_A: [BuiltinParamDescriptor; 1] = [INPUT_A];
const INPUTS_A_N: [BuiltinParamDescriptor; 2] = [INPUT_A, INPUT_N];
const OUTPUTS_B: [BuiltinParamDescriptor; 1] = [OUTPUT_B];
const OUTPUTS_B_M: [BuiltinParamDescriptor; 2] = [OUTPUT_B, OUTPUT_M];

pub const SHIFTDIM_SIGNATURES: [BuiltinSignatureDescriptor; 3] = [
    BuiltinSignatureDescriptor {
        label: "B = shiftdim(A)",
        inputs: &INPUTS_A,
        outputs: &OUTPUTS_B,
    },
    BuiltinSignatureDescriptor {
        label: "[B, m] = shiftdim(A)",
        inputs: &INPUTS_A,
        outputs: &OUTPUTS_B_M,
    },
    BuiltinSignatureDescriptor {
        label: "B = shiftdim(A, n)",
        inputs: &INPUTS_A_N,
        outputs: &OUTPUTS_B,
    },
];

pub const SHIFTDIM_ERROR_INVALID_SHIFT: BuiltinErrorDescriptor = BuiltinErrorDescriptor {
    code: "RM.SHIFTDIM.INVALID_SHIFT",
    identifier: Some("RunMat:shiftdim:InvalidShift"),
    when: "n is not a finite real integer scalar representable as a platform shift.",
    message: "shiftdim: n must be a finite real integer scalar",
};
pub const SHIFTDIM_ERROR_TOO_MANY_INPUTS: BuiltinErrorDescriptor = BuiltinErrorDescriptor {
    code: "RM.SHIFTDIM.TOO_MANY_INPUTS",
    identifier: Some("RunMat:shiftdim:TooManyInputs"),
    when: "More than two input arguments are supplied.",
    message: "shiftdim: too many input arguments",
};
pub const SHIFTDIM_ERROR_TOO_MANY_OUTPUTS: BuiltinErrorDescriptor = BuiltinErrorDescriptor {
    code: "RM.SHIFTDIM.TOO_MANY_OUTPUTS",
    identifier: Some("RunMat:shiftdim:TooManyOutputs"),
    when: "More than two outputs are requested, or m is requested when n is supplied.",
    message: "shiftdim: invalid number of output arguments",
};
pub const SHIFTDIM_ERROR_UNSUPPORTED_INPUT: BuiltinErrorDescriptor = BuiltinErrorDescriptor {
    code: "RM.SHIFTDIM.UNSUPPORTED_INPUT",
    identifier: Some("RunMat:shiftdim:UnsupportedInput"),
    when: "A has a type or representation that cannot express the requested shifted shape.",
    message: "shiftdim: unsupported input type or shape",
};
pub const SHIFTDIM_ERRORS: [BuiltinErrorDescriptor; 4] = [
    SHIFTDIM_ERROR_INVALID_SHIFT,
    SHIFTDIM_ERROR_TOO_MANY_INPUTS,
    SHIFTDIM_ERROR_TOO_MANY_OUTPUTS,
    SHIFTDIM_ERROR_UNSUPPORTED_INPUT,
];
pub const SHIFTDIM_DESCRIPTOR: BuiltinDescriptor = BuiltinDescriptor {
    signatures: &SHIFTDIM_SIGNATURES,
    output_mode: BuiltinOutputMode::ByRequestedOutputCount,
    completion_policy: BuiltinCompletionPolicy::Public,
    errors: &SHIFTDIM_ERRORS,
};

const INTEGER_INPUTS: [BuiltinIntegerInputCapability; 1] = [BuiltinIntegerInputCapability {
    name: "A", classes: &ALL_INTEGER_CLASSES,
    availability: BuiltinIntegerInputAvailability::Documented,
    scalar_double: BuiltinIntegerScalarDoubleRule::NotApplicable,
    notes: "Dimension shifting preserves exact native integer elements, including paired complex integer storage.",
}];
const INTEGER_SHIFT_INPUT: [BuiltinIntegerInputCapability; 1] = [BuiltinIntegerInputCapability {
    name: "n",
    classes: &ALL_INTEGER_CLASSES,
    availability: BuiltinIntegerInputAvailability::Documented,
    scalar_double: BuiltinIntegerScalarDoubleRule::Allowed,
    notes: "Host integer controls are decoded exactly and rejected outside the platform signed shift range before provider work.",
}];
pub const SHIFTDIM_INTEGER_CAPABILITIES: [BuiltinIntegerCapabilityDescriptor; 2] =
    [BuiltinIntegerCapabilityDescriptor {
        form: "B = shiftdim(integer_A [, n])",
        inputs: &INTEGER_INPUTS,
        computation_domain: BuiltinIntegerComputationDomain::Structural,
        output_class: BuiltinIntegerOutputClassRule::PreserveInput,
        overflow: BuiltinIntegerOverflowRule::NotApplicable,
        backend: BuiltinIntegerBackendRule::HostAndGpu,
        overload: BuiltinIntegerOverloadKind::Multiple,
        notes: "Structural shift controls are decoded exactly from native integer values; shape metadata or an exact-class permutation changes while payload bits are preserved.",
    }, BuiltinIntegerCapabilityDescriptor {
        form: "B = shiftdim(A, integer_n)",
        inputs: &INTEGER_SHIFT_INPUT,
        computation_domain: BuiltinIntegerComputationDomain::Structural,
        output_class: BuiltinIntegerOutputClassRule::PreserveInput,
        overflow: BuiltinIntegerOverflowRule::Error,
        backend: BuiltinIntegerBackendRule::HostAndGpu,
        overload: BuiltinIntegerOverloadKind::StructuralParameter,
        notes: "The control is host-only; an in-range native integer shift can still operate on resident A without converting the control through binary64.",
    }];
