use crate::*;

const ENDSWITH_OUTPUT: [BuiltinParamDescriptor; 1] = [BuiltinParamDescriptor {
    name: "tf",
    ty: BuiltinParamType::LogicalArray,
    arity: BuiltinParamArity::Required,
    default: None,
    description: "Logical result indicating whether each text element ends with the pattern.",
}];

const ENDSWITH_INPUTS_BASE: [BuiltinParamDescriptor; 2] = [
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

const ENDSWITH_INPUTS_IGNORE_CASE_POSITIONAL: [BuiltinParamDescriptor; 3] = [
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

const ENDSWITH_INPUTS_IGNORE_CASE_PAIR: [BuiltinParamDescriptor; 4] = [
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

const ENDSWITH_SIGNATURES: [BuiltinSignatureDescriptor; 3] = [
    BuiltinSignatureDescriptor {
        label: "tf = endsWith(str, pat)",
        inputs: &ENDSWITH_INPUTS_BASE,
        outputs: &ENDSWITH_OUTPUT,
    },
    BuiltinSignatureDescriptor {
        label: "tf = endsWith(str, pat, ignoreCase)",
        inputs: &ENDSWITH_INPUTS_IGNORE_CASE_POSITIONAL,
        outputs: &ENDSWITH_OUTPUT,
    },
    BuiltinSignatureDescriptor {
        label: "tf = endsWith(str, pat, \"IgnoreCase\", value)",
        inputs: &ENDSWITH_INPUTS_IGNORE_CASE_PAIR,
        outputs: &ENDSWITH_OUTPUT,
    },
];

pub const ENDSWITH_ERROR_INVALID_INPUT: BuiltinErrorDescriptor = BuiltinErrorDescriptor {
    code: "RM.ENDSWITH.INVALID_INPUT",
    identifier: Some("RunMat:endsWith:InvalidInput"),
    when: "Text or pattern input is not a supported text container.",
    message: "endsWith: text and pattern inputs must be text values",
};

pub const ENDSWITH_ERROR_INVALID_OPTION: BuiltinErrorDescriptor = BuiltinErrorDescriptor {
    code: "RM.ENDSWITH.INVALID_OPTION",
    identifier: Some("RunMat:endsWith:InvalidOption"),
    when: "IgnoreCase option arguments are invalid or malformed.",
    message: "endsWith: invalid option arguments",
};

pub const ENDSWITH_ERROR_INTERNAL: BuiltinErrorDescriptor = BuiltinErrorDescriptor {
    code: "RM.ENDSWITH.INTERNAL",
    identifier: Some("RunMat:endsWith:InternalError"),
    when: "Internal logical result assembly failed.",
    message: "endsWith: internal error",
};

const ENDSWITH_ERRORS: [BuiltinErrorDescriptor; 3] = [
    ENDSWITH_ERROR_INVALID_INPUT,
    ENDSWITH_ERROR_INVALID_OPTION,
    ENDSWITH_ERROR_INTERNAL,
];

pub const ENDSWITH_DESCRIPTOR: BuiltinDescriptor = BuiltinDescriptor {
    signatures: &ENDSWITH_SIGNATURES,
    output_mode: BuiltinOutputMode::Fixed,
    completion_policy: BuiltinCompletionPolicy::Public,
    errors: &ENDSWITH_ERRORS,
};

pub const ENDSWITH_POSITIONAL_IGNORE_CASE_EXTENSION: BuiltinExtensionDescriptor =
    BuiltinExtensionDescriptor {
        id: "endswith-positional-ignore-case",
        mode: BuiltinExtensionMode::RunMatOnly,
        description: "endsWith with a positional IgnoreCase flag is a RunMat extension",
        error_identifier: Some("RunMat:compatibility:EndsWithPositionalIgnoreCaseExtension"),
    };

pub const ENDSWITH_NUMERIC_IGNORE_CASE_EXTENSION: BuiltinExtensionDescriptor =
    BuiltinExtensionDescriptor {
        id: "endswith-numeric-ignore-case",
        mode: BuiltinExtensionMode::RunMatOnly,
        description: "endsWith with a numeric IgnoreCase value is a RunMat extension",
        error_identifier: Some("RunMat:compatibility:EndsWithNumericIgnoreCaseExtension"),
    };

pub const ENDSWITH_TEXT_IGNORE_CASE_EXTENSION: BuiltinExtensionDescriptor =
    BuiltinExtensionDescriptor {
        id: "endswith-text-ignore-case",
        mode: BuiltinExtensionMode::RunMatOnly,
        description: "endsWith with a textual IgnoreCase value is a RunMat extension",
        error_identifier: Some("RunMat:compatibility:EndsWithTextIgnoreCaseExtension"),
    };

pub const ENDSWITH_RESIDENT_IGNORE_CASE_EXTENSION: BuiltinExtensionDescriptor =
    BuiltinExtensionDescriptor {
        id: "endswith-resident-ignore-case",
        mode: BuiltinExtensionMode::RunMatOnly,
        description: "endsWith with a resident IgnoreCase value is a RunMat extension",
        error_identifier: Some("RunMat:compatibility:EndsWithResidentIgnoreCaseExtension"),
    };

pub const ENDSWITH_EXTENSIONS: [BuiltinExtensionDescriptor; 4] = [
    ENDSWITH_POSITIONAL_IGNORE_CASE_EXTENSION,
    ENDSWITH_NUMERIC_IGNORE_CASE_EXTENSION,
    ENDSWITH_TEXT_IGNORE_CASE_EXTENSION,
    ENDSWITH_RESIDENT_IGNORE_CASE_EXTENSION,
];

const ENDSWITH_INTEGER_IGNORE_CASE_INPUTS: [BuiltinIntegerInputCapability; 1] =
    [BuiltinIntegerInputCapability {
        name: "IgnoreCase",
        classes: &ALL_INTEGER_CLASSES,
        availability: BuiltinIntegerInputAvailability::RunMatOnly,
        scalar_double: BuiltinIntegerScalarDoubleRule::Allowed,
        notes: "RunMat mode accepts exact scalar integer zero as false and every nonzero integer as true; MATLAB-compatible mode requires the documented logical name-value form.",
    }];

pub const ENDSWITH_INTEGER_CAPABILITIES: [BuiltinIntegerCapabilityDescriptor; 2] = [
    BuiltinIntegerCapabilityDescriptor {
        form: "tf = endsWith(str, pat, 'IgnoreCase', integer_value)",
        inputs: &ENDSWITH_INTEGER_IGNORE_CASE_INPUTS,
        computation_domain: BuiltinIntegerComputationDomain::Predicate,
        output_class: BuiltinIntegerOutputClassRule::Logical,
        overflow: BuiltinIntegerOverflowRule::NotApplicable,
        backend: BuiltinIntegerBackendRule::GatherFallback,
        overload: BuiltinIntegerOverloadKind::ScalarOnly,
        notes: "Integer flags are compatibility-gated controls. Subject and pattern remain host text, multiple patterns are alternatives, and output has the subject shape.",
    },
    BuiltinIntegerCapabilityDescriptor {
        form: "tf = endsWith(str, pat, integer_ignore_case)",
        inputs: &ENDSWITH_INTEGER_IGNORE_CASE_INPUTS,
        computation_domain: BuiltinIntegerComputationDomain::Predicate,
        output_class: BuiltinIntegerOutputClassRule::Logical,
        overflow: BuiltinIntegerOverflowRule::NotApplicable,
        backend: BuiltinIntegerBackendRule::GatherFallback,
        overload: BuiltinIntegerOverloadKind::ScalarOnly,
        notes: "The positional integer flag requires both positional and numeric RunMat extension gates; a resident flag additionally requires the resident-control gate before gather.",
    },
];
