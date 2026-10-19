use crate::*;

const ERROR_INPUTS_MESSAGE: [BuiltinParamDescriptor; 1] = [BuiltinParamDescriptor {
    name: "message",
    ty: BuiltinParamType::StringScalar,
    arity: BuiltinParamArity::Required,
    default: None,
    description: "Error message text.",
}];

const ERROR_INPUTS_MESSAGE_VARIADIC: [BuiltinParamDescriptor; 2] = [
    BuiltinParamDescriptor {
        name: "message",
        ty: BuiltinParamType::StringScalar,
        arity: BuiltinParamArity::Required,
        default: None,
        description: "Error message template text.",
    },
    BuiltinParamDescriptor {
        name: "A",
        ty: BuiltinParamType::Any,
        arity: BuiltinParamArity::Variadic,
        default: None,
        description: "Formatting values for the message template.",
    },
];

const ERROR_INPUTS_IDENTIFIER_MESSAGE: [BuiltinParamDescriptor; 3] = [
    BuiltinParamDescriptor {
        name: "message_id",
        ty: BuiltinParamType::StringScalar,
        arity: BuiltinParamArity::Required,
        default: None,
        description: "Message identifier.",
    },
    BuiltinParamDescriptor {
        name: "message",
        ty: BuiltinParamType::StringScalar,
        arity: BuiltinParamArity::Required,
        default: None,
        description: "Error message text.",
    },
    BuiltinParamDescriptor {
        name: "A",
        ty: BuiltinParamType::Any,
        arity: BuiltinParamArity::Variadic,
        default: None,
        description: "Formatting values for the message template.",
    },
];

const ERROR_INPUTS_STRUCT: [BuiltinParamDescriptor; 1] = [BuiltinParamDescriptor {
    name: "msg_struct",
    ty: BuiltinParamType::Any,
    arity: BuiltinParamArity::Required,
    default: None,
    description: "Struct containing identifier/message fields.",
}];

const ERROR_INPUTS_CORRECTION: [BuiltinParamDescriptor; 2] = [
    BuiltinParamDescriptor {
        name: "correction",
        ty: BuiltinParamType::Any,
        arity: BuiltinParamArity::Required,
        default: None,
        description: "MATLAB correction object.",
    },
    BuiltinParamDescriptor {
        name: "messageArguments",
        ty: BuiltinParamType::Any,
        arity: BuiltinParamArity::Variadic,
        default: None,
        description: "Identifier, message, and optional formatting values.",
    },
];

const ERROR_SIGNATURES: [BuiltinSignatureDescriptor; 5] = [
    BuiltinSignatureDescriptor {
        label: "error(msg)",
        inputs: &ERROR_INPUTS_MESSAGE,
        outputs: &[],
    },
    BuiltinSignatureDescriptor {
        label: "error(msg, A)",
        inputs: &ERROR_INPUTS_MESSAGE_VARIADIC,
        outputs: &[],
    },
    BuiltinSignatureDescriptor {
        label: "error(errID, ___)",
        inputs: &ERROR_INPUTS_IDENTIFIER_MESSAGE,
        outputs: &[],
    },
    BuiltinSignatureDescriptor {
        label: "error(errorStruct)",
        inputs: &ERROR_INPUTS_STRUCT,
        outputs: &[],
    },
    BuiltinSignatureDescriptor {
        label: "error(correction, ___)",
        inputs: &ERROR_INPUTS_CORRECTION,
        outputs: &[],
    },
];

pub const ERROR_ERROR_MISSING_MESSAGE: BuiltinErrorDescriptor = BuiltinErrorDescriptor {
    code: "RM.ERROR.MISSING_MESSAGE",
    identifier: Some("RunMat:error"),
    when: "No arguments are supplied.",
    message: "error: missing message argument",
};

pub const ERROR_ERROR_EXTRA_ARGS_MEXCEPTION: BuiltinErrorDescriptor = BuiltinErrorDescriptor {
    code: "RM.ERROR.MEXCEPTION_EXTRA_ARGS",
    identifier: Some("RunMat:error"),
    when: "Additional arguments are supplied after an MException input.",
    message: "error: additional arguments are not allowed when passing an MException",
};

pub const ERROR_ERROR_EXTRA_ARGS_STRUCT: BuiltinErrorDescriptor = BuiltinErrorDescriptor {
    code: "RM.ERROR.STRUCT_EXTRA_ARGS",
    identifier: Some("RunMat:error"),
    when: "Additional arguments are supplied after a message-struct input.",
    message: "error: additional arguments are not allowed when passing a message struct",
};

pub const ERROR_ERROR_STRUCT_NO_FIELDS: BuiltinErrorDescriptor = BuiltinErrorDescriptor {
    code: "RM.ERROR.STRUCT_NO_FIELDS",
    identifier: Some("RunMat:error"),
    when: "Message struct contains none of message, identifier, or stack.",
    message: "error: message struct must contain 'message', 'identifier', or 'stack'",
};

pub const ERROR_ERROR_STRUCT_STACK_UNSUPPORTED: BuiltinErrorDescriptor = BuiltinErrorDescriptor {
    code: "RM.ERROR.STRUCT_STACK_UNSUPPORTED",
    identifier: Some("RunMat:error"),
    when: "Message struct requests an explicit MATLAB stack that cannot be represented yet.",
    message: "error: explicit errorStruct stack is not supported yet",
};

pub const ERROR_ERROR_INVALID_INPUT: BuiltinErrorDescriptor = BuiltinErrorDescriptor {
    code: "RM.ERROR.INVALID_INPUT",
    identifier: Some("RunMat:error"),
    when: "Identifier/message inputs or format arguments are not string-compatible.",
    message: "error: invalid input argument",
};

pub const ERROR_ERROR_INVALID_IDENTIFIER: BuiltinErrorDescriptor = BuiltinErrorDescriptor {
    code: "RM.ERROR.INVALID_IDENTIFIER",
    identifier: Some("RunMat:error"),
    when: "Identifier fields do not follow documented colon-separated identifier grammar.",
    message: "error: invalid error identifier",
};

pub const ERROR_ERROR_CORRECTION_UNSUPPORTED: BuiltinErrorDescriptor = BuiltinErrorDescriptor {
    code: "RM.ERROR.CORRECTION_UNSUPPORTED",
    identifier: Some("RunMat:error"),
    when: "A documented matlab.lang.correction object is supplied before correction objects are representable.",
    message: "error: matlab.lang.correction objects are not supported yet",
};

const ERROR_ERRORS: [BuiltinErrorDescriptor; 8] = [
    ERROR_ERROR_MISSING_MESSAGE,
    ERROR_ERROR_EXTRA_ARGS_MEXCEPTION,
    ERROR_ERROR_EXTRA_ARGS_STRUCT,
    ERROR_ERROR_STRUCT_NO_FIELDS,
    ERROR_ERROR_STRUCT_STACK_UNSUPPORTED,
    ERROR_ERROR_INVALID_INPUT,
    ERROR_ERROR_INVALID_IDENTIFIER,
    ERROR_ERROR_CORRECTION_UNSUPPORTED,
];

pub const ERROR_DESCRIPTOR: BuiltinDescriptor = BuiltinDescriptor {
    signatures: &ERROR_SIGNATURES,
    output_mode: BuiltinOutputMode::Fixed,
    completion_policy: BuiltinCompletionPolicy::Public,
    errors: &ERROR_ERRORS,
};

pub const ERROR_MEXCEPTION_EXTENSION: BuiltinExtensionDescriptor = BuiltinExtensionDescriptor {
    id: "error-mexception-input",
    mode: BuiltinExtensionMode::RunMatOnly,
    description: "error(MException) is a RunMat extension; MATLAB uses throw or rethrow",
    error_identifier: Some("RunMat:compatibility:ErrorMExceptionExtension"),
};
pub const ERROR_UNQUALIFIED_IDENTIFIER_EXTENSION: BuiltinExtensionDescriptor =
    BuiltinExtensionDescriptor {
        id: "error-unqualified-identifier",
        mode: BuiltinExtensionMode::RunMatOnly,
        description:
            "Treating an unqualified leading token as an error identifier is a RunMat extension",
        error_identifier: Some("RunMat:compatibility:ErrorUnqualifiedIdentifierExtension"),
    };
pub const ERROR_STRUCT_ALIAS_EXTENSION: BuiltinExtensionDescriptor = BuiltinExtensionDescriptor {
    id: "error-struct-field-aliases",
    mode: BuiltinExtensionMode::RunMatOnly,
    description: "messageid and msg error-structure aliases are a RunMat extension",
    error_identifier: Some("RunMat:compatibility:ErrorStructAliasExtension"),
};
pub const ERROR_EXTENSIONS: [BuiltinExtensionDescriptor; 3] = [
    ERROR_MEXCEPTION_EXTENSION,
    ERROR_UNQUALIFIED_IDENTIFIER_EXTENSION,
    ERROR_STRUCT_ALIAS_EXTENSION,
];

const ERROR_INTEGER_FORMAT_INPUT: [BuiltinIntegerInputCapability; 1] =
    [BuiltinIntegerInputCapability {
        name: "A",
        classes: &ALL_INTEGER_CLASSES,
        availability: BuiltinIntegerInputAvailability::Documented,
        scalar_double: BuiltinIntegerScalarDoubleRule::NotApplicable,
        notes: "All eight typed-integer classes are documented numeric formatting values and remain exact through integer conversion specifiers.",
    }];
const ERROR_REJECTED_INTEGER_TEXT_INPUTS: [BuiltinIntegerInputCapability; 2] = [
    BuiltinIntegerInputCapability {
        name: "msg",
        classes: &[],
        availability: BuiltinIntegerInputAvailability::Rejected,
        scalar_double: BuiltinIntegerScalarDoubleRule::NotApplicable,
        notes: "The message role is a text scalar and never converts integer data to text implicitly.",
    },
    BuiltinIntegerInputCapability {
        name: "errID",
        classes: &[],
        availability: BuiltinIntegerInputAvailability::Rejected,
        scalar_double: BuiltinIntegerScalarDoubleRule::NotApplicable,
        notes: "The identifier role is a text scalar and rejects integer values before provider access.",
    },
];
pub const ERROR_INTEGER_CAPABILITIES: [BuiltinIntegerCapabilityDescriptor; 2] = [
    BuiltinIntegerCapabilityDescriptor {
        form: "error(msg, integer_A) or error(errID, msg, integer_A)",
        inputs: &ERROR_INTEGER_FORMAT_INPUT,
        computation_domain: BuiltinIntegerComputationDomain::FunctionSpecific,
        output_class: BuiltinIntegerOutputClassRule::NotApplicable,
        overflow: BuiltinIntegerOverflowRule::NotApplicable,
        backend: BuiltinIntegerBackendRule::GatherFallback,
        overload: BuiltinIntegerOverloadKind::Multiple,
        notes: "Host formatting is exact for integer conversions; documented GPU-array arguments gather only after host message and identifier validation.",
    },
    BuiltinIntegerCapabilityDescriptor {
        form: "error(integer_msg, ...) or error(integer_errID, ...)",
        inputs: &ERROR_REJECTED_INTEGER_TEXT_INPUTS,
        computation_domain: BuiltinIntegerComputationDomain::FunctionSpecific,
        output_class: BuiltinIntegerOutputClassRule::NotApplicable,
        overflow: BuiltinIntegerOverflowRule::NotApplicable,
        backend: BuiltinIntegerBackendRule::HostOnly,
        overload: BuiltinIntegerOverloadKind::FunctionSpecific,
        notes: "Integer values are valid only in formatting-value roles, never as message or identifier text.",
    },
];
const RETHROW_OUTPUT: [BuiltinParamDescriptor; 0] = [];

const RETHROW_INPUTS: [BuiltinParamDescriptor; 1] = [BuiltinParamDescriptor {
    name: "err",
    ty: BuiltinParamType::Any,
    arity: BuiltinParamArity::Required,
    default: None,
    description: "Error value to rethrow.",
}];

const RETHROW_SIGNATURES: [BuiltinSignatureDescriptor; 2] = [
    BuiltinSignatureDescriptor {
        label: "rethrow()",
        inputs: &[],
        outputs: &RETHROW_OUTPUT,
    },
    BuiltinSignatureDescriptor {
        label: "rethrow(err)",
        inputs: &RETHROW_INPUTS,
        outputs: &RETHROW_OUTPUT,
    },
];

pub const RETHROW_ERROR_NO_ACTIVE_CATCH: BuiltinErrorDescriptor = BuiltinErrorDescriptor {
    code: "RM.RETHROW.NO_ACTIVE_CATCH",
    identifier: Some("RunMat:rethrow:NoActiveCatch"),
    when: "The RunMat shorthand is invoked outside a lexical catch block.",
    message: "rethrow(): no active lexical catch exception",
};
pub const RETHROW_ERROR_INVALID_ARITY: BuiltinErrorDescriptor = BuiltinErrorDescriptor {
    code: "RM.RETHROW.INVALID_ARITY",
    identifier: Some("RunMat:rethrow:InvalidArity"),
    when: "More than one argument is supplied.",
    message: "rethrow accepts zero or one argument",
};
const RETHROW_ERRORS: [BuiltinErrorDescriptor; 2] =
    [RETHROW_ERROR_NO_ACTIVE_CATCH, RETHROW_ERROR_INVALID_ARITY];

pub const RETHROW_DESCRIPTOR: BuiltinDescriptor = BuiltinDescriptor {
    signatures: &RETHROW_SIGNATURES,
    output_mode: BuiltinOutputMode::Fixed,
    completion_policy: BuiltinCompletionPolicy::Public,
    errors: &RETHROW_ERRORS,
};

pub const RETHROW_ACTIVE_CATCH_EXTENSION: BuiltinExtensionDescriptor = BuiltinExtensionDescriptor {
    id: "rethrow-active-catch-shorthand",
    mode: BuiltinExtensionMode::RunMatOnly,
    description: "rethrow() without an explicit MException is a RunMat catch-block extension",
    error_identifier: Some("RunMat:compatibility:RethrowActiveCatchExtension"),
};
pub const RETHROW_TEXT_EXTENSION: BuiltinExtensionDescriptor = BuiltinExtensionDescriptor {
    id: "rethrow-text-input",
    mode: BuiltinExtensionMode::RunMatOnly,
    description: "rethrow(text) is a RunMat extension; MATLAB requires an MException",
    error_identifier: Some("RunMat:compatibility:RethrowTextExtension"),
};
pub const RETHROW_EXTENSIONS: [BuiltinExtensionDescriptor; 2] =
    [RETHROW_ACTIVE_CATCH_EXTENSION, RETHROW_TEXT_EXTENSION];

pub const RETHROW_INTEGER_AUDIT: BuiltinIntegerAuditDescriptor = BuiltinIntegerAuditDescriptor {
    kind: BuiltinIntegerAuditKind::NotApplicable,
    canonical_builtin: None,
    notes: "rethrow consumes a caught exception object and has no numeric data, control, class-preserving output, or provider role. Numeric, logical, symbolic, and resident values fail directly without gather or provider access.",
};
