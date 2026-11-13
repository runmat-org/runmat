use crate::*;

pub const CHAR_RESIDENT_NUMERIC_EXTENSION: BuiltinExtensionDescriptor =
    BuiltinExtensionDescriptor {
        id: "char-resident-numeric-input",
        mode: BuiltinExtensionMode::RunMatOnly,
        description: "char with interactive resident numeric input is a RunMat extension",
        error_identifier: Some("RunMat:compatibility:CharResidentNumericInputExtension"),
    };
pub const CHAR_LOGICAL_INPUT_EXTENSION: BuiltinExtensionDescriptor = BuiltinExtensionDescriptor {
    id: "char-logical-input",
    mode: BuiltinExtensionMode::RunMatOnly,
    description: "char with logical input is a RunMat extension",
    error_identifier: Some("RunMat:compatibility:CharLogicalInputExtension"),
};

pub const CHAR_EXTENSIONS: [BuiltinExtensionDescriptor; 2] = [
    CHAR_LOGICAL_INPUT_EXTENSION,
    CHAR_RESIDENT_NUMERIC_EXTENSION,
];

const CHAR_OUTPUT: [BuiltinParamDescriptor; 1] = [BuiltinParamDescriptor {
    name: "C",
    ty: BuiltinParamType::Any,
    arity: BuiltinParamArity::Required,
    default: None,
    description: "Character array result.",
}];

const CHAR_INPUT_SINGLE: [BuiltinParamDescriptor; 1] = [BuiltinParamDescriptor {
    name: "X",
    ty: BuiltinParamType::Any,
    arity: BuiltinParamArity::Required,
    default: None,
    description: "Input value to convert into character data.",
}];

const CHAR_INPUT_VARIADIC: [BuiltinParamDescriptor; 1] = [BuiltinParamDescriptor {
    name: "X...",
    ty: BuiltinParamType::Any,
    arity: BuiltinParamArity::Variadic,
    default: None,
    description: "Multiple inputs converted row-wise and padded.",
}];

const CHAR_SIGNATURES: [BuiltinSignatureDescriptor; 3] = [
    BuiltinSignatureDescriptor {
        label: "C = char()",
        inputs: &[],
        outputs: &CHAR_OUTPUT,
    },
    BuiltinSignatureDescriptor {
        label: "C = char(X)",
        inputs: &CHAR_INPUT_SINGLE,
        outputs: &CHAR_OUTPUT,
    },
    BuiltinSignatureDescriptor {
        label: "C = char(X...)",
        inputs: &CHAR_INPUT_VARIADIC,
        outputs: &CHAR_OUTPUT,
    },
];

pub const CHAR_ERROR_INVALID_INPUT: BuiltinErrorDescriptor = BuiltinErrorDescriptor {
    code: "RM.CHAR.INVALID_INPUT",
    identifier: Some("RunMat:char:InvalidInput"),
    when: "Input type cannot be converted to character data.",
    message: "char: invalid input",
};

pub const CHAR_ERROR_INVALID_CODEPOINT: BuiltinErrorDescriptor = BuiltinErrorDescriptor {
    code: "RM.CHAR.INVALID_CODEPOINT",
    identifier: Some("RunMat:char:InvalidCodePoint"),
    when: "Numeric input cannot be represented by RunMat's scalar-value character storage.",
    message: "char: numeric input cannot be represented as a RunMat character",
};

pub const CHAR_ERROR_DIMENSION: BuiltinErrorDescriptor = BuiltinErrorDescriptor {
    code: "RM.CHAR.INVALID_DIMENSION",
    identifier: Some("RunMat:char:InvalidDimension"),
    when: "Array inputs are not 2-D (or trailing singleton dimensions).",
    message: "char: inputs must be 2-D",
};

pub const CHAR_ERROR_INTERNAL: BuiltinErrorDescriptor = BuiltinErrorDescriptor {
    code: "RM.CHAR.INTERNAL",
    identifier: Some("RunMat:char:InternalError"),
    when: "Internal character array construction failed.",
    message: "char: internal error",
};

pub const CHAR_ERRORS: [BuiltinErrorDescriptor; 4] = [
    CHAR_ERROR_INVALID_INPUT,
    CHAR_ERROR_INVALID_CODEPOINT,
    CHAR_ERROR_DIMENSION,
    CHAR_ERROR_INTERNAL,
];

pub const CHAR_DESCRIPTOR: BuiltinDescriptor = BuiltinDescriptor {
    signatures: &CHAR_SIGNATURES,
    output_mode: BuiltinOutputMode::Fixed,
    completion_policy: BuiltinCompletionPolicy::Public,
    errors: &CHAR_ERRORS,
};

const CHAR_INTEGER_INPUTS: [BuiltinIntegerInputCapability; 1] = [BuiltinIntegerInputCapability {
    name: "X",
    classes: &ALL_INTEGER_CLASSES,
    availability: BuiltinIntegerInputAvailability::Documented,
    scalar_double: BuiltinIntegerScalarDoubleRule::Allowed,
    notes: "All eight integer classes are documented numeric-code inputs. Floating values truncate toward zero and all numeric codes clamp to the UTF-16 code-unit interval 0..65535.",
}];

pub const CHAR_INTEGER_CAPABILITIES: [BuiltinIntegerCapabilityDescriptor; 1] =
    [BuiltinIntegerCapabilityDescriptor {
        form: "C = char(integer_X) or char(X1, ..., XN)",
        inputs: &CHAR_INTEGER_INPUTS,
        computation_domain: BuiltinIntegerComputationDomain::ExactInteger,
        output_class: BuiltinIntegerOutputClassRule::NotApplicable,
        overflow: BuiltinIntegerOverflowRule::Saturate,
        backend: BuiltinIntegerBackendRule::GatherFallback,
        overload: BuiltinIntegerOverloadKind::Multiple,
        notes: "Host integer values are decoded exactly, then clamped to 0..65535. RunMat CharArray stores Rust Unicode scalar values, so isolated UTF-16 surrogate code units U+D800..U+DFFF cannot yet be represented and produce an explicit error. Interactive resident numeric input is a mode-gated RunMat extension before gather.",
    }];
