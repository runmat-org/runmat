use crate::*;

pub const TRIU_PAGED_INPUT_EXTENSION: BuiltinExtensionDescriptor = BuiltinExtensionDescriptor {
    id: "triu-paged-input",
    mode: BuiltinExtensionMode::RunMatOnly,
    description: "triu on paged arrays is a RunMat extension",
    error_identifier: Some("RunMat:compatibility:TriuPagedInputExtension"),
};
pub const TRIU_EXTENSIONS: [BuiltinExtensionDescriptor; 1] = [TRIU_PAGED_INPUT_EXTENSION];

const OUTPUTS: [BuiltinParamDescriptor; 1] = [BuiltinParamDescriptor {
    name: "B",
    ty: BuiltinParamType::Any,
    arity: BuiltinParamArity::Required,
    default: None,
    description: "Upper-triangular output with the input class and shape.",
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
        label: "B = triu(A)",
        inputs: &INPUT_A,
        outputs: &OUTPUTS,
    },
    BuiltinSignatureDescriptor {
        label: "B = triu(A, k)",
        inputs: &INPUT_A_K,
        outputs: &OUTPUTS,
    },
];

pub const TRIU_ERROR_TOO_MANY_INPUTS: BuiltinErrorDescriptor = BuiltinErrorDescriptor {
    code: "RM.TRIU.TOO_MANY_INPUTS",
    identifier: Some("RunMat:triu:TooManyInputs"),
    when: "More than one optional argument is supplied after A.",
    message: "triu: too many input arguments",
};
pub const TRIU_ERROR_TOO_MANY_OUTPUTS: BuiltinErrorDescriptor = BuiltinErrorDescriptor {
    code: "RM.TRIU.TOO_MANY_OUTPUTS",
    identifier: Some("RunMat:triu:TooManyOutputs"),
    when: "More than one output is requested.",
    message: "triu: too many output arguments",
};
pub const TRIU_ERROR_INVALID_OFFSET: BuiltinErrorDescriptor = BuiltinErrorDescriptor {
    code: "RM.TRIU.INVALID_OFFSET", identifier: Some("RunMat:triu:InvalidOffset"),
    when: "The diagonal offset is not a finite integer-valued real scalar in the platform index range.",
    message: "triu: invalid diagonal offset",
};
pub const TRIU_ERROR_UNSUPPORTED_INPUT: BuiltinErrorDescriptor = BuiltinErrorDescriptor {
    code: "RM.TRIU.UNSUPPORTED_INPUT", identifier: Some("RunMat:triu:UnsupportedInput"),
    when: "A is sparse, textual string data, a cell, object, structure, symbolic value, callable, or another unsupported carrier.",
    message: "triu: unsupported input type",
};
pub const TRIU_ERROR_INTERNAL: BuiltinErrorDescriptor = BuiltinErrorDescriptor {
    code: "RM.TRIU.INTERNAL",
    identifier: Some("RunMat:triu:Internal"),
    when: "Mask allocation, provider execution, gathering, or owner-provider restoration fails.",
    message: "triu: internal operation failed",
};
const ERRORS: [BuiltinErrorDescriptor; 5] = [
    TRIU_ERROR_TOO_MANY_INPUTS,
    TRIU_ERROR_TOO_MANY_OUTPUTS,
    TRIU_ERROR_INVALID_OFFSET,
    TRIU_ERROR_UNSUPPORTED_INPUT,
    TRIU_ERROR_INTERNAL,
];
pub const TRIU_DESCRIPTOR: BuiltinDescriptor = BuiltinDescriptor {
    signatures: &SIGNATURES,
    output_mode: BuiltinOutputMode::Fixed,
    completion_policy: BuiltinCompletionPolicy::Public,
    errors: &ERRORS,
};

const INTEGER_DATA: [BuiltinIntegerInputCapability; 1] = [BuiltinIntegerInputCapability {
    name: "A", classes: &ALL_INTEGER_CLASSES, availability: BuiltinIntegerInputAvailability::Documented,
    scalar_double: BuiltinIntegerScalarDoubleRule::NotApplicable,
    notes: "Masking writes a same-class zero and preserves every native integer value that remains selected.",
}];
const INTEGER_OFFSET: [BuiltinIntegerInputCapability; 1] = [BuiltinIntegerInputCapability {
    name: "k", classes: &ALL_INTEGER_CLASSES, availability: BuiltinIntegerInputAvailability::Documented,
    scalar_double: BuiltinIntegerScalarDoubleRule::Allowed,
    notes: "Native integer controls are decoded exactly and must fit isize; floating controls must be finite integer-valued scalars.",
}];
pub const TRIU_INTEGER_CAPABILITIES: [BuiltinIntegerCapabilityDescriptor; 2] = [
    BuiltinIntegerCapabilityDescriptor {
        form: "B = triu(integer_A [, k])", inputs: &INTEGER_DATA,
        computation_domain: BuiltinIntegerComputationDomain::Structural,
        output_class: BuiltinIntegerOutputClassRule::PreserveInput,
        overflow: BuiltinIntegerOverflowRule::NotApplicable,
        backend: BuiltinIntegerBackendRule::HostAndGpu,
        overload: BuiltinIntegerOverloadKind::Multiple,
        notes: "Host masking preserves native storage. Resident integer values use exact gather/mask/owner-provider restoration because provider floating kernels are inapplicable.",
    },
    BuiltinIntegerCapabilityDescriptor {
        form: "B = triu(A, integer_k)", inputs: &INTEGER_OFFSET,
        computation_domain: BuiltinIntegerComputationDomain::Structural,
        output_class: BuiltinIntegerOutputClassRule::FunctionSpecific,
        overflow: BuiltinIntegerOverflowRule::Error,
        backend: BuiltinIntegerBackendRule::GatherFallback,
        overload: BuiltinIntegerOverloadKind::StructuralParameter,
        notes: "The structural control is gathered when resident, decoded without binary64 conversion, and validated before masking A.",
    },
];
