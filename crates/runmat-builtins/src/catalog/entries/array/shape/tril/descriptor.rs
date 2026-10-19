use crate::*;

const OUTPUTS: [BuiltinParamDescriptor; 1] = [BuiltinParamDescriptor {
    name: "B",
    ty: BuiltinParamType::Any,
    arity: BuiltinParamArity::Required,
    default: None,
    description: "Lower-triangular output with the input class and shape.",
}];

const INPUT_A_PARAM: BuiltinParamDescriptor = BuiltinParamDescriptor {
    name: "A",
    ty: BuiltinParamType::Any,
    arity: BuiltinParamArity::Required,
    default: None,
    description: "Dense numeric, logical, character, complex, or resident array.",
};
const INPUT_A: [BuiltinParamDescriptor; 1] = [INPUT_A_PARAM];

const INPUT_A_K: [BuiltinParamDescriptor; 2] = [
    INPUT_A_PARAM,
    BuiltinParamDescriptor {
        name: "k",
        ty: BuiltinParamType::NumericScalar,
        arity: BuiltinParamArity::Required,
        default: None,
        description: "Finite integer-valued diagonal offset.",
    },
];

const SIGNATURES: [BuiltinSignatureDescriptor; 2] = [
    BuiltinSignatureDescriptor {
        label: "B = tril(A)",
        inputs: &INPUT_A,
        outputs: &OUTPUTS,
    },
    BuiltinSignatureDescriptor {
        label: "B = tril(A, k)",
        inputs: &INPUT_A_K,
        outputs: &OUTPUTS,
    },
];

pub const TRIL_ERROR_TOO_MANY_INPUTS: BuiltinErrorDescriptor = BuiltinErrorDescriptor {
    code: "RM.TRIL.TOO_MANY_INPUTS",
    identifier: Some("RunMat:tril:TooManyInputs"),
    when: "More than one optional argument is supplied after A.",
    message: "tril: too many input arguments",
};
pub const TRIL_ERROR_TOO_MANY_OUTPUTS: BuiltinErrorDescriptor = BuiltinErrorDescriptor {
    code: "RM.TRIL.TOO_MANY_OUTPUTS",
    identifier: Some("RunMat:tril:TooManyOutputs"),
    when: "More than one output is requested.",
    message: "tril: too many output arguments",
};
pub const TRIL_ERROR_INVALID_OFFSET: BuiltinErrorDescriptor = BuiltinErrorDescriptor {
    code: "RM.TRIL.INVALID_OFFSET",
    identifier: Some("RunMat:tril:InvalidOffset"),
    when: "The diagonal offset is not a finite integer-valued real scalar in the platform index range.",
    message: "tril: invalid diagonal offset",
};
pub const TRIL_ERROR_UNSUPPORTED_INPUT: BuiltinErrorDescriptor = BuiltinErrorDescriptor {
    code: "RM.TRIL.UNSUPPORTED_INPUT",
    identifier: Some("RunMat:tril:UnsupportedInput"),
    when: "A is sparse, textual string data, a cell, object, structure, symbolic value, callable, or another unsupported carrier.",
    message: "tril: unsupported input type",
};
pub const TRIL_ERROR_INTERNAL: BuiltinErrorDescriptor = BuiltinErrorDescriptor {
    code: "RM.TRIL.INTERNAL",
    identifier: Some("RunMat:tril:Internal"),
    when: "Mask allocation, provider execution, gathering, or owner-provider restoration fails.",
    message: "tril: internal operation failed",
};

const ERRORS: [BuiltinErrorDescriptor; 5] = [
    TRIL_ERROR_TOO_MANY_INPUTS,
    TRIL_ERROR_TOO_MANY_OUTPUTS,
    TRIL_ERROR_INVALID_OFFSET,
    TRIL_ERROR_UNSUPPORTED_INPUT,
    TRIL_ERROR_INTERNAL,
];

pub const TRIL_DESCRIPTOR: BuiltinDescriptor = BuiltinDescriptor {
    signatures: &SIGNATURES,
    output_mode: BuiltinOutputMode::Fixed,
    completion_policy: BuiltinCompletionPolicy::Public,
    errors: &ERRORS,
};

const INTEGER_DATA: [BuiltinIntegerInputCapability; 1] = [BuiltinIntegerInputCapability {
    name: "A",
    classes: &ALL_INTEGER_CLASSES,
    availability: BuiltinIntegerInputAvailability::Documented,
    scalar_double: BuiltinIntegerScalarDoubleRule::NotApplicable,
    notes: "Masking writes a same-class zero and preserves every native integer value that remains selected.",
}];
const INTEGER_OFFSET: [BuiltinIntegerInputCapability; 1] = [BuiltinIntegerInputCapability {
    name: "k",
    classes: &ALL_INTEGER_CLASSES,
    availability: BuiltinIntegerInputAvailability::Documented,
    scalar_double: BuiltinIntegerScalarDoubleRule::Allowed,
    notes: "Native integer controls are decoded exactly and must fit isize; floating controls must be finite integer-valued scalars.",
}];
pub const TRIL_INTEGER_CAPABILITIES: [BuiltinIntegerCapabilityDescriptor; 2] = [
    BuiltinIntegerCapabilityDescriptor {
        form: "B = tril(integer_A [, k])",
        inputs: &INTEGER_DATA,
        computation_domain: BuiltinIntegerComputationDomain::Structural,
        output_class: BuiltinIntegerOutputClassRule::PreserveInput,
        overflow: BuiltinIntegerOverflowRule::NotApplicable,
        backend: BuiltinIntegerBackendRule::HostAndGpu,
        overload: BuiltinIntegerOverloadKind::Multiple,
        notes: "Host masking preserves native storage. Resident integer values use exact gather/mask/owner-provider restoration because provider floating kernels are inapplicable.",
    },
    BuiltinIntegerCapabilityDescriptor {
        form: "B = tril(A, integer_k)",
        inputs: &INTEGER_OFFSET,
        computation_domain: BuiltinIntegerComputationDomain::Structural,
        output_class: BuiltinIntegerOutputClassRule::FunctionSpecific,
        overflow: BuiltinIntegerOverflowRule::Error,
        backend: BuiltinIntegerBackendRule::GatherFallback,
        overload: BuiltinIntegerOverloadKind::StructuralParameter,
        notes: "The structural control is gathered when resident, decoded without binary64 conversion, and validated before masking A.",
    },
];
