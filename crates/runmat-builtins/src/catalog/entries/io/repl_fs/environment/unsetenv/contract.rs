use crate::*;

const NAME: BuiltinParamDescriptor = BuiltinParamDescriptor {
    name: "name",
    ty: BuiltinParamType::Any,
    arity: BuiltinParamArity::Required,
    default: None,
    description: "Character row, string scalar or array, or cell array of character rows.",
};
const STATUS: BuiltinParamDescriptor = BuiltinParamDescriptor {
    name: "status",
    ty: BuiltinParamType::NumericScalar,
    arity: BuiltinParamArity::Required,
    default: None,
    description: "RunMat extension: zero after successful removal and one for an invalid name.",
};
const SIGNATURES: &[BuiltinSignatureDescriptor] = &[
    BuiltinSignatureDescriptor {
        label: "unsetenv(name)",
        inputs: &[NAME],
        outputs: &[],
    },
    BuiltinSignatureDescriptor {
        label: "status = unsetenv(name)",
        inputs: &[NAME],
        outputs: &[STATUS],
    },
];
pub const UNSETENV_ERROR_ARITY: BuiltinErrorDescriptor = BuiltinErrorDescriptor {
    code: "RM.UNSETENV.ARITY",
    identifier: Some("RunMat:unsetenv:Arity"),
    when: "The call does not have exactly one input.",
    message: "unsetenv: expected exactly one input",
};
pub const UNSETENV_ERROR_NAME: BuiltinErrorDescriptor = BuiltinErrorDescriptor {
    code: "RM.UNSETENV.INVALID_NAME",
    identifier: Some("RunMat:unsetenv:InvalidName"),
    when: "An environment name is invalid or has an unsupported class or shape.",
    message:
        "unsetenv: names must be character vectors, string values, or cells of character vectors",
};
pub const UNSETENV_ERROR_TOO_MANY_OUTPUTS: BuiltinErrorDescriptor = BuiltinErrorDescriptor {
    code: "RM.UNSETENV.TOO_MANY_OUTPUTS",
    identifier: Some("RunMat:unsetenv:TooManyOutputs"),
    when: "More than one output is requested.",
    message: "unsetenv: too many output arguments",
};
pub const UNSETENV_DESCRIPTOR: BuiltinDescriptor = BuiltinDescriptor {
    signatures: SIGNATURES,
    output_mode: BuiltinOutputMode::ByRequestedOutputCount,
    completion_policy: BuiltinCompletionPolicy::Public,
    errors: &[
        UNSETENV_ERROR_ARITY,
        UNSETENV_ERROR_NAME,
        UNSETENV_ERROR_TOO_MANY_OUTPUTS,
    ],
};
pub const UNSETENV_STATUS_OUTPUT_EXTENSION: BuiltinExtensionDescriptor =
    BuiltinExtensionDescriptor {
        id: "unsetenv-status-output",
        mode: BuiltinExtensionMode::RunMatOnly,
        description: "unsetenv status output is a RunMat extension",
        error_identifier: Some("RunMat:compatibility:UnsetenvStatusOutputExtension"),
    };
pub(super) const EXTENSIONS: &[BuiltinExtensionDescriptor] = &[UNSETENV_STATUS_OUTPUT_EXTENSION];
pub const UNSETENV_INTEGER_AUDIT: BuiltinIntegerAuditDescriptor = BuiltinIntegerAuditDescriptor { kind: BuiltinIntegerAuditKind::NotApplicable, canonical_builtin: None, notes: "Environment names are text. Integer and provider-resident numeric values reject before provider or environment access." };
