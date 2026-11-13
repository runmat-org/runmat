use crate::*;

const STARTSWITH_OUTPUT: [BuiltinParamDescriptor; 1] = [BuiltinParamDescriptor {
    name: "tf",
    ty: BuiltinParamType::LogicalArray,
    arity: BuiltinParamArity::Required,
    default: None,
    description: "Logical result indicating whether each text element starts with the pattern.",
}];

const STARTSWITH_INPUTS_BASE: [BuiltinParamDescriptor; 2] = [
    BuiltinParamDescriptor {
        name: "str",
        ty: BuiltinParamType::Any,
        arity: BuiltinParamArity::Required,
        default: None,
        description: "Text input (string/char/cell/string array).",
    },
    BuiltinParamDescriptor {
        name: "pat",
        ty: BuiltinParamType::Any,
        arity: BuiltinParamArity::Required,
        default: None,
        description: "Pattern text (string/char/cell/string array).",
    },
];

const STARTSWITH_INPUTS_IGNORE_CASE_POSITIONAL: [BuiltinParamDescriptor; 3] = [
    BuiltinParamDescriptor {
        name: "str",
        ty: BuiltinParamType::Any,
        arity: BuiltinParamArity::Required,
        default: None,
        description: "Text input (string/char/cell/string array).",
    },
    BuiltinParamDescriptor {
        name: "pat",
        ty: BuiltinParamType::Any,
        arity: BuiltinParamArity::Required,
        default: None,
        description: "Pattern text (string/char/cell/string array).",
    },
    BuiltinParamDescriptor {
        name: "ignoreCase",
        ty: BuiltinParamType::Any,
        arity: BuiltinParamArity::Required,
        default: Some("false"),
        description: "Logical flag controlling case-sensitive matching.",
    },
];

const STARTSWITH_INPUTS_IGNORE_CASE_PAIR: [BuiltinParamDescriptor; 4] = [
    BuiltinParamDescriptor {
        name: "str",
        ty: BuiltinParamType::Any,
        arity: BuiltinParamArity::Required,
        default: None,
        description: "Text input (string/char/cell/string array).",
    },
    BuiltinParamDescriptor {
        name: "pat",
        ty: BuiltinParamType::Any,
        arity: BuiltinParamArity::Required,
        default: None,
        description: "Pattern text (string/char/cell/string array).",
    },
    BuiltinParamDescriptor {
        name: "name",
        ty: BuiltinParamType::StringScalar,
        arity: BuiltinParamArity::Required,
        default: Some("\"IgnoreCase\""),
        description: "Option name (`\"IgnoreCase\"`).",
    },
    BuiltinParamDescriptor {
        name: "value",
        ty: BuiltinParamType::Any,
        arity: BuiltinParamArity::Required,
        default: None,
        description: "Option value for `\"IgnoreCase\"`.",
    },
];

const STARTSWITH_SIGNATURES: [BuiltinSignatureDescriptor; 3] = [
    BuiltinSignatureDescriptor {
        label: "tf = startsWith(str, pat)",
        inputs: &STARTSWITH_INPUTS_BASE,
        outputs: &STARTSWITH_OUTPUT,
    },
    BuiltinSignatureDescriptor {
        label: "tf = startsWith(str, pat, ignoreCase)",
        inputs: &STARTSWITH_INPUTS_IGNORE_CASE_POSITIONAL,
        outputs: &STARTSWITH_OUTPUT,
    },
    BuiltinSignatureDescriptor {
        label: "tf = startsWith(str, pat, \"IgnoreCase\", value)",
        inputs: &STARTSWITH_INPUTS_IGNORE_CASE_PAIR,
        outputs: &STARTSWITH_OUTPUT,
    },
];

pub const STARTSWITH_ERROR_INVALID_INPUT: BuiltinErrorDescriptor = BuiltinErrorDescriptor {
    code: "RM.STARTSWITH.INVALID_INPUT",
    identifier: Some("RunMat:startsWith:InvalidInput"),
    when: "Text or pattern input is not a supported text container.",
    message: "startsWith: text and pattern inputs must be text values",
};

pub const STARTSWITH_ERROR_INVALID_OPTION: BuiltinErrorDescriptor = BuiltinErrorDescriptor {
    code: "RM.STARTSWITH.INVALID_OPTION",
    identifier: Some("RunMat:startsWith:InvalidOption"),
    when: "IgnoreCase option arguments are invalid or malformed.",
    message: "startsWith: invalid option arguments",
};

pub const STARTSWITH_ERROR_SHAPE_MISMATCH: BuiltinErrorDescriptor = BuiltinErrorDescriptor {
    code: "RM.STARTSWITH.SHAPE_MISMATCH",
    identifier: Some("RunMat:startsWith:ShapeMismatch"),
    when: "Text and pattern inputs are not broadcast-compatible.",
    message: "startsWith: input sizes are not broadcast-compatible",
};

pub const STARTSWITH_ERROR_INTERNAL: BuiltinErrorDescriptor = BuiltinErrorDescriptor {
    code: "RM.STARTSWITH.INTERNAL",
    identifier: Some("RunMat:startsWith:InternalError"),
    when: "Internal logical result assembly failed.",
    message: "startsWith: internal error",
};

const STARTSWITH_ERRORS: [BuiltinErrorDescriptor; 4] = [
    STARTSWITH_ERROR_INVALID_INPUT,
    STARTSWITH_ERROR_INVALID_OPTION,
    STARTSWITH_ERROR_SHAPE_MISMATCH,
    STARTSWITH_ERROR_INTERNAL,
];

pub const STARTSWITH_DESCRIPTOR: BuiltinDescriptor = BuiltinDescriptor {
    signatures: &STARTSWITH_SIGNATURES,
    output_mode: BuiltinOutputMode::Fixed,
    completion_policy: BuiltinCompletionPolicy::Public,
    errors: &STARTSWITH_ERRORS,
};

pub const STARTSWITH_POSITIONAL_IGNORE_CASE_EXTENSION: BuiltinExtensionDescriptor =
    BuiltinExtensionDescriptor {
        id: "startswith-positional-ignore-case",
        mode: BuiltinExtensionMode::RunMatOnly,
        description: "startsWith with a positional IgnoreCase flag is a RunMat extension",
        error_identifier: Some("RunMat:compatibility:StartsWithPositionalIgnoreCaseExtension"),
    };

pub const STARTSWITH_NUMERIC_IGNORE_CASE_EXTENSION: BuiltinExtensionDescriptor =
    BuiltinExtensionDescriptor {
        id: "startswith-numeric-ignore-case",
        mode: BuiltinExtensionMode::RunMatOnly,
        description: "startsWith with a numeric IgnoreCase value is a RunMat extension",
        error_identifier: Some("RunMat:compatibility:StartsWithNumericIgnoreCaseExtension"),
    };

pub const STARTSWITH_TEXT_IGNORE_CASE_EXTENSION: BuiltinExtensionDescriptor =
    BuiltinExtensionDescriptor {
        id: "startswith-text-ignore-case",
        mode: BuiltinExtensionMode::RunMatOnly,
        description: "startsWith with a textual IgnoreCase value is a RunMat extension",
        error_identifier: Some("RunMat:compatibility:StartsWithTextIgnoreCaseExtension"),
    };

pub const STARTSWITH_RESIDENT_IGNORE_CASE_EXTENSION: BuiltinExtensionDescriptor =
    BuiltinExtensionDescriptor {
        id: "startswith-resident-ignore-case",
        mode: BuiltinExtensionMode::RunMatOnly,
        description: "startsWith with a resident IgnoreCase value is a RunMat extension",
        error_identifier: Some("RunMat:compatibility:StartsWithResidentIgnoreCaseExtension"),
    };

pub const STARTSWITH_EXTENSIONS: [BuiltinExtensionDescriptor; 4] = [
    STARTSWITH_POSITIONAL_IGNORE_CASE_EXTENSION,
    STARTSWITH_NUMERIC_IGNORE_CASE_EXTENSION,
    STARTSWITH_TEXT_IGNORE_CASE_EXTENSION,
    STARTSWITH_RESIDENT_IGNORE_CASE_EXTENSION,
];

const STARTSWITH_INTEGER_IGNORE_CASE_INPUTS: [BuiltinIntegerInputCapability; 1] =
    [BuiltinIntegerInputCapability {
        name: "IgnoreCase",
        classes: &ALL_INTEGER_CLASSES,
        availability: BuiltinIntegerInputAvailability::RunMatOnly,
        scalar_double: BuiltinIntegerScalarDoubleRule::Allowed,
        notes: "RunMat mode accepts exact scalar integer zero as false and every nonzero integer as true; MATLAB-compatible mode requires the documented logical name-value form.",
    }];

pub const STARTSWITH_INTEGER_CAPABILITIES: [BuiltinIntegerCapabilityDescriptor; 2] = [
    BuiltinIntegerCapabilityDescriptor {
        form: "tf = startsWith(str, pat, 'IgnoreCase', integer_value)",
        inputs: &STARTSWITH_INTEGER_IGNORE_CASE_INPUTS,
        computation_domain: BuiltinIntegerComputationDomain::Predicate,
        output_class: BuiltinIntegerOutputClassRule::Logical,
        overflow: BuiltinIntegerOverflowRule::NotApplicable,
        backend: BuiltinIntegerBackendRule::GatherFallback,
        overload: BuiltinIntegerOverloadKind::ScalarOnly,
        notes: "Integer flags are compatibility-gated controls. Subject and pattern remain host text, multiple patterns are alternatives, and output has the subject shape.",
    },
    BuiltinIntegerCapabilityDescriptor {
        form: "tf = startsWith(str, pat, integer_ignore_case)",
        inputs: &STARTSWITH_INTEGER_IGNORE_CASE_INPUTS,
        computation_domain: BuiltinIntegerComputationDomain::Predicate,
        output_class: BuiltinIntegerOutputClassRule::Logical,
        overflow: BuiltinIntegerOverflowRule::NotApplicable,
        backend: BuiltinIntegerBackendRule::GatherFallback,
        overload: BuiltinIntegerOverloadKind::ScalarOnly,
        notes: "The positional integer flag requires both positional and numeric RunMat extension gates; a resident flag additionally requires the resident-control gate before gather.",
    },
];
