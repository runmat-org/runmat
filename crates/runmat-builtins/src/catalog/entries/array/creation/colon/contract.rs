//! Canonical colon signatures, errors, extensions, and integer forms.
use crate::*;

pub const COLON_LOGICAL_INPUT_EXTENSION: BuiltinExtensionDescriptor = BuiltinExtensionDescriptor {
    id: "colon-logical-input",
    mode: BuiltinExtensionMode::RunMatOnly,
    description: "colon with logical operands is a RunMat extension",
    error_identifier: Some("RunMat:compatibility:ColonLogicalInputExtension"),
};

pub const COLON_ZERO_IMAGINARY_COMPLEX_EXTENSION: BuiltinExtensionDescriptor =
    BuiltinExtensionDescriptor {
        id: "colon-zero-imaginary-complex",
        mode: BuiltinExtensionMode::RunMatOnly,
        description: "colon with zero-imaginary complex operands is a RunMat extension",
        error_identifier: Some("RunMat:compatibility:ColonZeroImaginaryComplexExtension"),
    };

pub const COLON_GPU_64_BIT_INTEGER_EXTENSION: BuiltinExtensionDescriptor =
    BuiltinExtensionDescriptor {
        id: "colon-gpu-64-bit-integer",
        mode: BuiltinExtensionMode::RunMatOnly,
        description: "colon with resident int64 or uint64 operands is a RunMat extension",
        error_identifier: Some("RunMat:compatibility:ColonGpu64BitIntegerExtension"),
    };

pub const COLON_EXTENSIONS: [BuiltinExtensionDescriptor; 3] = [
    COLON_LOGICAL_INPUT_EXTENSION,
    COLON_ZERO_IMAGINARY_COMPLEX_EXTENSION,
    COLON_GPU_64_BIT_INTEGER_EXTENSION,
];

const COLON_INTEGER_INPUTS_TWO: [BuiltinIntegerInputCapability; 2] = [
    BuiltinIntegerInputCapability { name: "start", classes: &ALL_INTEGER_CLASSES, availability: BuiltinIntegerInputAvailability::Documented, scalar_double: BuiltinIntegerScalarDoubleRule::Allowed, notes: "A typed-integer operand selects the output integer class; all typed-integer operands must use that class." },
    BuiltinIntegerInputCapability { name: "stop", classes: &ALL_INTEGER_CLASSES, availability: BuiltinIntegerInputAvailability::Documented, scalar_double: BuiltinIntegerScalarDoubleRule::Allowed, notes: "The bound must be representable by the selected output class." },
];

const COLON_INTEGER_INPUTS_THREE: [BuiltinIntegerInputCapability; 3] = [
    BuiltinIntegerInputCapability { name: "start", classes: &ALL_INTEGER_CLASSES, availability: BuiltinIntegerInputAvailability::Documented, scalar_double: BuiltinIntegerScalarDoubleRule::Allowed, notes: "A typed-integer operand selects the output integer class; all typed-integer operands must use that class." },
    BuiltinIntegerInputCapability { name: "step", classes: &ALL_INTEGER_CLASSES, availability: BuiltinIntegerInputAvailability::Documented, scalar_double: BuiltinIntegerScalarDoubleRule::Allowed, notes: "The explicit step is decoded exactly and may be negative even for an unsigned output class." },
    BuiltinIntegerInputCapability { name: "stop", classes: &ALL_INTEGER_CLASSES, availability: BuiltinIntegerInputAvailability::Documented, scalar_double: BuiltinIntegerScalarDoubleRule::Allowed, notes: "The bound must be representable by the selected output class." },
];

pub const COLON_INTEGER_CAPABILITIES: [BuiltinIntegerCapabilityDescriptor; 2] = [
    BuiltinIntegerCapabilityDescriptor { form: "x = colon(integer_start, integer_stop)", inputs: &COLON_INTEGER_INPUTS_TWO, computation_domain: BuiltinIntegerComputationDomain::ExactInteger, output_class: BuiltinIntegerOutputClassRule::PreserveInput, overflow: BuiltinIntegerOverflowRule::Error, backend: BuiltinIntegerBackendRule::GpuRestricted, overload: BuiltinIntegerOverloadKind::ScalarOnly, notes: "All eight integer classes use exact native storage with an implicit +1 step." },
    BuiltinIntegerCapabilityDescriptor { form: "x = colon(integer_start, integer_step, integer_stop)", inputs: &COLON_INTEGER_INPUTS_THREE, computation_domain: BuiltinIntegerComputationDomain::ExactInteger, output_class: BuiltinIntegerOutputClassRule::PreserveInput, overflow: BuiltinIntegerOverflowRule::Error, backend: BuiltinIntegerBackendRule::GpuRestricted, overload: BuiltinIntegerOverloadKind::ScalarOnly, notes: "All eight integer classes use exact native storage. MATLAB-compatible GPU execution excludes int64 and uint64; RunMat mode accepts them through an exact owning-provider gather/upload fallback." },
];

const COLON_OUTPUT: [BuiltinParamDescriptor; 1] = [BuiltinParamDescriptor {
    name: "x",
    ty: BuiltinParamType::Any,
    arity: BuiltinParamArity::Required,
    default: None,
    description: "Arithmetic progression row vector (numeric or character).",
}];

const COLON_SIG_TWO_INPUTS: [BuiltinParamDescriptor; 2] = [
    BuiltinParamDescriptor {
        name: "start",
        ty: BuiltinParamType::Any,
        arity: BuiltinParamArity::Required,
        default: None,
        description: "Start scalar value.",
    },
    BuiltinParamDescriptor {
        name: "stop",
        ty: BuiltinParamType::Any,
        arity: BuiltinParamArity::Required,
        default: None,
        description: "Stop scalar value (implicit step = 1).",
    },
];

const COLON_SIG_THREE_INPUTS: [BuiltinParamDescriptor; 3] = [
    BuiltinParamDescriptor {
        name: "start",
        ty: BuiltinParamType::Any,
        arity: BuiltinParamArity::Required,
        default: None,
        description: "Start scalar value.",
    },
    BuiltinParamDescriptor {
        name: "step",
        ty: BuiltinParamType::Any,
        arity: BuiltinParamArity::Required,
        default: None,
        description: "Increment; zero returns an empty row vector.",
    },
    BuiltinParamDescriptor {
        name: "stop",
        ty: BuiltinParamType::Any,
        arity: BuiltinParamArity::Required,
        default: None,
        description: "Stop scalar value.",
    },
];

const COLON_SIGNATURES: [BuiltinSignatureDescriptor; 2] = [
    BuiltinSignatureDescriptor {
        label: "x = colon(start, stop)",
        inputs: &COLON_SIG_TWO_INPUTS,
        outputs: &COLON_OUTPUT,
    },
    BuiltinSignatureDescriptor {
        label: "x = colon(start, step, stop)",
        inputs: &COLON_SIG_THREE_INPUTS,
        outputs: &COLON_OUTPUT,
    },
];

pub const COLON_ERROR_ARG_COUNT: BuiltinErrorDescriptor = BuiltinErrorDescriptor {
    code: "RM.COLON.ARG_COUNT",
    identifier: None,
    when: "More than three input arguments are provided.",
    message: "colon: expected two or three input arguments",
};

pub const COLON_ERROR_NON_SCALAR_INPUT: BuiltinErrorDescriptor = BuiltinErrorDescriptor {
    code: "RM.COLON.NON_SCALAR_INPUT",
    identifier: None,
    when: "At least one input is not scalar.",
    message: "colon: expected scalar input",
};

pub const COLON_ERROR_NON_FINITE_INPUT: BuiltinErrorDescriptor = BuiltinErrorDescriptor {
    code: "RM.COLON.NON_FINITE_INPUT",
    identifier: None,
    when: "At least one input scalar is non-finite.",
    message: "colon: inputs must be finite numeric scalars",
};

pub const COLON_ERROR_COMPLEX_IMAGINARY_NONZERO: BuiltinErrorDescriptor = BuiltinErrorDescriptor {
    code: "RM.COLON.COMPLEX_IMAGINARY_NONZERO",
    identifier: None,
    when: "Complex inputs have non-zero imaginary parts.",
    message: "colon: complex inputs must have zero imaginary part",
};

pub const COLON_ERROR_UNSUPPORTED_STRING_INPUT: BuiltinErrorDescriptor = BuiltinErrorDescriptor {
    code: "RM.COLON.UNSUPPORTED_STRING_INPUT",
    identifier: None,
    when: "String-like values are used as scalar bounds/step.",
    message: "colon: inputs must be real scalar values; received a string-like argument",
};

pub const COLON_ERROR_CHAR_NON_INTEGER_CODEPOINT: BuiltinErrorDescriptor = BuiltinErrorDescriptor {
    code: "RM.COLON.CHAR_NON_INTEGER_CODEPOINT",
    identifier: None,
    when: "Character sequence values are non-integer.",
    message: "colon: character sequence requires integer code points",
};

pub const COLON_ERROR_CHAR_CODEPOINT_RANGE: BuiltinErrorDescriptor = BuiltinErrorDescriptor {
    code: "RM.COLON.CHAR_CODEPOINT_RANGE",
    identifier: None,
    when: "Character sequence values are outside valid Unicode range.",
    message: "colon: character code point out of range",
};

pub const COLON_ERROR_SEQUENCE_RANGE: BuiltinErrorDescriptor = BuiltinErrorDescriptor {
    code: "RM.COLON.SEQUENCE_RANGE",
    identifier: None,
    when: "Computed progression span/ratio is non-finite.",
    message: "colon: sequence length exceeds representable range",
};

pub const COLON_ERROR_SEQUENCE_LIMIT: BuiltinErrorDescriptor = BuiltinErrorDescriptor {
    code: "RM.COLON.SEQUENCE_LIMIT",
    identifier: None,
    when: "Computed progression length exceeds platform limits.",
    message: "colon: sequence length exceeds platform limits",
};

pub const COLON_ERROR_INTERNAL: BuiltinErrorDescriptor = BuiltinErrorDescriptor {
    code: "RM.COLON.INTERNAL",
    identifier: None,
    when: "Internal tensor/character output construction failed.",
    message: "colon: internal error",
};

const COLON_ERRORS: [BuiltinErrorDescriptor; 10] = [
    COLON_ERROR_ARG_COUNT,
    COLON_ERROR_NON_SCALAR_INPUT,
    COLON_ERROR_NON_FINITE_INPUT,
    COLON_ERROR_COMPLEX_IMAGINARY_NONZERO,
    COLON_ERROR_UNSUPPORTED_STRING_INPUT,
    COLON_ERROR_CHAR_NON_INTEGER_CODEPOINT,
    COLON_ERROR_CHAR_CODEPOINT_RANGE,
    COLON_ERROR_SEQUENCE_RANGE,
    COLON_ERROR_SEQUENCE_LIMIT,
    COLON_ERROR_INTERNAL,
];

pub const COLON_DESCRIPTOR: BuiltinDescriptor = BuiltinDescriptor {
    signatures: &COLON_SIGNATURES,
    output_mode: BuiltinOutputMode::Fixed,
    completion_policy: BuiltinCompletionPolicy::Public,
    errors: &COLON_ERRORS,
};
