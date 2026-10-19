use crate::*;

const OUTPUT: [BuiltinParamDescriptor; 1] = [BuiltinParamDescriptor {
    name: "line",
    ty: BuiltinParamType::Any,
    arity: BuiltinParamArity::Required,
    default: None,
    description: "String scalar without its terminator, or a 0-by-0 double on timeout.",
}];
const INPUTS: [BuiltinParamDescriptor; 1] = [BuiltinParamDescriptor {
    name: "client",
    ty: BuiltinParamType::Any,
    arity: BuiltinParamArity::Required,
    default: None,
    description: "Live tcpclient value.",
}];
const SIGNATURES: [BuiltinSignatureDescriptor; 1] = [BuiltinSignatureDescriptor {
    label: "line = readline(client)",
    inputs: &INPUTS,
    outputs: &OUTPUT,
}];

pub const READLINE_ERROR_INVALID_CLIENT: BuiltinErrorDescriptor = BuiltinErrorDescriptor {
    code: "RM.READLINE.INVALID_CLIENT",
    identifier: Some("RunMat:readline:InvalidTcpClient"),
    when: "Client handle is missing, malformed, invalid, or stale.",
    message: "readline: invalid tcpclient handle",
};
pub const READLINE_ERROR_NOT_CONNECTED: BuiltinErrorDescriptor = BuiltinErrorDescriptor {
    code: "RM.READLINE.NOT_CONNECTED",
    identifier: Some("RunMat:readline:NotConnected"),
    when: "Client has no active socket connection.",
    message: "readline: tcpclient is disconnected",
};
pub const READLINE_ERROR_INVALID_ARGUMENTS: BuiltinErrorDescriptor = BuiltinErrorDescriptor {
    code: "RM.READLINE.INVALID_ARGUMENTS",
    identifier: Some("RunMat:readline:InvalidArguments"),
    when: "Unexpected additional positional arguments are provided.",
    message: "readline: invalid argument list",
};
pub const READLINE_ERROR_INTERNAL: BuiltinErrorDescriptor = BuiltinErrorDescriptor {
    code: "RM.READLINE.INTERNAL",
    identifier: Some("RunMat:readline:InternalError"),
    when: "Internal socket or control-flow conversion fails.",
    message: "readline: internal socket error",
};
const ERRORS: [BuiltinErrorDescriptor; 6] = [
    super::super::NET_ERROR_UNSUPPORTED_TARGET,
    super::super::capability::NET_ERROR_CANCELLED,
    READLINE_ERROR_INVALID_CLIENT,
    READLINE_ERROR_NOT_CONNECTED,
    READLINE_ERROR_INVALID_ARGUMENTS,
    READLINE_ERROR_INTERNAL,
];
pub const READLINE_DESCRIPTOR: BuiltinDescriptor = BuiltinDescriptor {
    signatures: &SIGNATURES,
    output_mode: BuiltinOutputMode::Fixed,
    completion_policy: BuiltinCompletionPolicy::Public,
    errors: &ERRORS,
};
pub const READLINE_INTEGER_AUDIT: BuiltinIntegerAuditDescriptor = BuiltinIntegerAuditDescriptor {
    kind: BuiltinIntegerAuditKind::NotApplicable,
    canonical_builtin: None,
    notes: "The client input is a socket value; integer values are invalid clients, while the only numeric result is the empty double timeout sentinel.",
};
