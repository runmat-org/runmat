use crate::*;

const CONTAINS_OUTPUT: [BuiltinParamDescriptor; 1] = [BuiltinParamDescriptor {
    name: "tf",
    ty: BuiltinParamType::LogicalArray,
    arity: BuiltinParamArity::Required,
    default: None,
    description: "Logical result indicating whether each text element contains the pattern.",
}];

const CONTAINS_INPUTS_BASE: [BuiltinParamDescriptor; 2] = [
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

const CONTAINS_INPUTS_IGNORE_CASE_POSITIONAL: [BuiltinParamDescriptor; 3] = [
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

const CONTAINS_INPUTS_IGNORE_CASE_PAIR: [BuiltinParamDescriptor; 4] = [
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

const CONTAINS_SIGNATURES: [BuiltinSignatureDescriptor; 3] = [
    BuiltinSignatureDescriptor {
        label: "tf = contains(str, pat)",
        inputs: &CONTAINS_INPUTS_BASE,
        outputs: &CONTAINS_OUTPUT,
    },
    BuiltinSignatureDescriptor {
        label: "tf = contains(str, pat, ignoreCase)",
        inputs: &CONTAINS_INPUTS_IGNORE_CASE_POSITIONAL,
        outputs: &CONTAINS_OUTPUT,
    },
    BuiltinSignatureDescriptor {
        label: "tf = contains(str, pat, \"IgnoreCase\", value)",
        inputs: &CONTAINS_INPUTS_IGNORE_CASE_PAIR,
        outputs: &CONTAINS_OUTPUT,
    },
];

pub const CONTAINS_ERROR_INVALID_INPUT: BuiltinErrorDescriptor = BuiltinErrorDescriptor {
    code: "RM.CONTAINS.INVALID_INPUT",
    identifier: Some("RunMat:contains:InvalidInput"),
    when: "Text or pattern input is not a supported text container.",
    message: "contains: text and pattern inputs must be text values",
};

pub const CONTAINS_ERROR_INVALID_OPTION: BuiltinErrorDescriptor = BuiltinErrorDescriptor {
    code: "RM.CONTAINS.INVALID_OPTION",
    identifier: Some("RunMat:contains:InvalidOption"),
    when: "IgnoreCase option arguments are invalid or malformed.",
    message: "contains: invalid option arguments",
};

pub const CONTAINS_ERROR_SHAPE_MISMATCH: BuiltinErrorDescriptor = BuiltinErrorDescriptor {
    code: "RM.CONTAINS.SHAPE_MISMATCH",
    identifier: Some("RunMat:contains:ShapeMismatch"),
    when: "Text and pattern inputs are not broadcast-compatible.",
    message: "contains: input sizes are not broadcast-compatible",
};

pub const CONTAINS_ERROR_INTERNAL: BuiltinErrorDescriptor = BuiltinErrorDescriptor {
    code: "RM.CONTAINS.INTERNAL",
    identifier: Some("RunMat:contains:InternalError"),
    when: "Internal logical result assembly failed.",
    message: "contains: internal error",
};

const CONTAINS_ERRORS: [BuiltinErrorDescriptor; 4] = [
    CONTAINS_ERROR_INVALID_INPUT,
    CONTAINS_ERROR_INVALID_OPTION,
    CONTAINS_ERROR_SHAPE_MISMATCH,
    CONTAINS_ERROR_INTERNAL,
];

pub const CONTAINS_DESCRIPTOR: BuiltinDescriptor = BuiltinDescriptor {
    signatures: &CONTAINS_SIGNATURES,
    output_mode: BuiltinOutputMode::Fixed,
    completion_policy: BuiltinCompletionPolicy::Public,
    errors: &CONTAINS_ERRORS,
};

pub const CONTAINS_POSITIONAL_IGNORE_CASE_EXTENSION: BuiltinExtensionDescriptor =
    BuiltinExtensionDescriptor {
        id: "contains-positional-ignore-case",
        mode: BuiltinExtensionMode::RunMatOnly,
        description: "contains with a positional IgnoreCase flag is a RunMat extension",
        error_identifier: Some("RunMat:compatibility:ContainsPositionalIgnoreCaseExtension"),
    };

pub const CONTAINS_NUMERIC_IGNORE_CASE_EXTENSION: BuiltinExtensionDescriptor =
    BuiltinExtensionDescriptor {
        id: "contains-numeric-ignore-case",
        mode: BuiltinExtensionMode::RunMatOnly,
        description: "contains with a numeric IgnoreCase value is a RunMat extension",
        error_identifier: Some("RunMat:compatibility:ContainsNumericIgnoreCaseExtension"),
    };

pub const CONTAINS_TEXT_IGNORE_CASE_EXTENSION: BuiltinExtensionDescriptor =
    BuiltinExtensionDescriptor {
        id: "contains-text-ignore-case",
        mode: BuiltinExtensionMode::RunMatOnly,
        description: "contains with a textual IgnoreCase value is a RunMat extension",
        error_identifier: Some("RunMat:compatibility:ContainsTextIgnoreCaseExtension"),
    };

pub const CONTAINS_EXTENSIONS: [BuiltinExtensionDescriptor; 3] = [
    CONTAINS_POSITIONAL_IGNORE_CASE_EXTENSION,
    CONTAINS_NUMERIC_IGNORE_CASE_EXTENSION,
    CONTAINS_TEXT_IGNORE_CASE_EXTENSION,
];

const CONTAINS_INTEGER_IGNORE_CASE_INPUTS: [BuiltinIntegerInputCapability; 1] =
    [BuiltinIntegerInputCapability {
        name: "IgnoreCase",
        classes: &ALL_INTEGER_CLASSES,
        availability: BuiltinIntegerInputAvailability::RunMatOnly,
        scalar_double: BuiltinIntegerScalarDoubleRule::Allowed,
        notes: "RunMat mode accepts a scalar integer zero as false and any nonzero integer as true; MATLAB-compatible mode requires the documented logical name-value form.",
    }];

pub const CONTAINS_INTEGER_CAPABILITIES: [BuiltinIntegerCapabilityDescriptor; 2] = [
    BuiltinIntegerCapabilityDescriptor {
        form: "tf = contains(str, pat, 'IgnoreCase', integer_value)",
        inputs: &CONTAINS_INTEGER_IGNORE_CASE_INPUTS,
        computation_domain: BuiltinIntegerComputationDomain::Predicate,
        output_class: BuiltinIntegerOutputClassRule::Logical,
        overflow: BuiltinIntegerOverflowRule::NotApplicable,
        backend: BuiltinIntegerBackendRule::HostOnly,
        overload: BuiltinIntegerOverloadKind::ScalarOnly,
        notes: "Integer flags are a compatibility-gated RunMat convenience. Text matching remains host-only and returns logical output; numeric subject or pattern data is never treated as text.",
    },
    BuiltinIntegerCapabilityDescriptor {
        form: "tf = contains(str, pat, integer_ignore_case)",
        inputs: &CONTAINS_INTEGER_IGNORE_CASE_INPUTS,
        computation_domain: BuiltinIntegerComputationDomain::Predicate,
        output_class: BuiltinIntegerOutputClassRule::Logical,
        overflow: BuiltinIntegerOverflowRule::NotApplicable,
        backend: BuiltinIntegerBackendRule::HostOnly,
        overload: BuiltinIntegerOverloadKind::ScalarOnly,
        notes: "The positional integer flag requires both the positional and numeric RunMat extension gates. Zero is false and any nonzero exact integer is true; subject and pattern inputs remain text-only.",
    },
];
