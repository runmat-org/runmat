use crate::*;
const ACCEPT_OUTPUT_CLIENT: [BuiltinParamDescriptor; 1] = [BuiltinParamDescriptor {
    name: "client",
    ty: BuiltinParamType::Any,
    arity: BuiltinParamArity::Required,
    default: None,
    description: "tcpclient object with sealed authority for the accepted connection.",
}];
const ACCEPT_INPUTS_SERVER: [BuiltinParamDescriptor; 1] = [BuiltinParamDescriptor {
    name: "server",
    ty: BuiltinParamType::Any,
    arity: BuiltinParamArity::Required,
    default: None,
    description: "tcpserver object returned by tcpserver.",
}];
const ACCEPT_INPUTS_SERVER_TIMEOUT: [BuiltinParamDescriptor; 3] = [
    BuiltinParamDescriptor {
        name: "server",
        ty: BuiltinParamType::Any,
        arity: BuiltinParamArity::Required,
        default: None,
        description: "tcpserver object returned by tcpserver.",
    },
    BuiltinParamDescriptor {
        name: "name",
        ty: BuiltinParamType::StringScalar,
        arity: BuiltinParamArity::Required,
        default: Some("\"Timeout\""),
        description: "Name token. Supported value: \"Timeout\".",
    },
    BuiltinParamDescriptor {
        name: "timeout",
        ty: BuiltinParamType::NumericScalar,
        arity: BuiltinParamArity::Required,
        default: None,
        description: "Per-call accept timeout in seconds (non-negative, supports Inf).",
    },
];
const ACCEPT_INPUTS_SERVER_NAME_VALUE: [BuiltinParamDescriptor; 2] = [
    BuiltinParamDescriptor {
        name: "server",
        ty: BuiltinParamType::Any,
        arity: BuiltinParamArity::Required,
        default: None,
        description: "tcpserver object returned by tcpserver.",
    },
    BuiltinParamDescriptor {
        name: "name_value_pairs",
        ty: BuiltinParamType::Any,
        arity: BuiltinParamArity::Variadic,
        default: None,
        description: "Name/Value option pairs (currently Timeout only).",
    },
];
const ACCEPT_SIGNATURES: [BuiltinSignatureDescriptor; 3] = [
    BuiltinSignatureDescriptor {
        label: "client = accept(server)",
        inputs: &ACCEPT_INPUTS_SERVER,
        outputs: &ACCEPT_OUTPUT_CLIENT,
    },
    BuiltinSignatureDescriptor {
        label: "client = accept(server, \"Timeout\", timeout)",
        inputs: &ACCEPT_INPUTS_SERVER_TIMEOUT,
        outputs: &ACCEPT_OUTPUT_CLIENT,
    },
    BuiltinSignatureDescriptor {
        label: "client = accept(server, Name, Value, ...)",
        inputs: &ACCEPT_INPUTS_SERVER_NAME_VALUE,
        outputs: &ACCEPT_OUTPUT_CLIENT,
    },
];

pub const ACCEPT_ERROR_INVALID_SERVER: BuiltinErrorDescriptor = BuiltinErrorDescriptor {
    code: "RM.ACCEPT.INVALID_SERVER",
    identifier: Some("RunMat:accept:InvalidTcpServer"),
    when: "Server argument is missing/invalid or references a stale tcpserver handle.",
    message: "accept: invalid tcpserver handle",
};
pub const ACCEPT_ERROR_TIMEOUT: BuiltinErrorDescriptor = BuiltinErrorDescriptor {
    code: "RM.ACCEPT.TIMEOUT",
    identifier: Some("RunMat:accept:Timeout"),
    when: "No incoming connection arrives before timeout elapses.",
    message: "accept: timed out waiting for a client connection",
};
pub const ACCEPT_ERROR_INVALID_NAME_VALUE: BuiltinErrorDescriptor = BuiltinErrorDescriptor {
    code: "RM.ACCEPT.INVALID_NAME_VALUE",
    identifier: Some("RunMat:accept:InvalidNameValue"),
    when: "Name/Value options are malformed, unsupported, or have invalid values.",
    message: "accept: invalid name-value arguments",
};
pub const ACCEPT_ERROR_INTERNAL: BuiltinErrorDescriptor = BuiltinErrorDescriptor {
    code: "RM.ACCEPT.INTERNAL",
    identifier: Some("RunMat:accept:InternalError"),
    when: "Internal lock/stream setup operation fails.",
    message: "accept: internal error",
};
pub const ACCEPT_ERROR_ACCEPT_FAILED: BuiltinErrorDescriptor = BuiltinErrorDescriptor {
    code: "RM.ACCEPT.ACCEPT_FAILED",
    identifier: Some("RunMat:accept:AcceptFailed"),
    when: "Underlying socket accept call fails for reasons other than timeout.",
    message: "accept: failed to accept client",
};
const ACCEPT_ERRORS: [BuiltinErrorDescriptor; 7] = [
    super::super::NET_ERROR_UNSUPPORTED_TARGET,
    super::super::capability::NET_ERROR_CANCELLED,
    ACCEPT_ERROR_INVALID_SERVER,
    ACCEPT_ERROR_TIMEOUT,
    ACCEPT_ERROR_INVALID_NAME_VALUE,
    ACCEPT_ERROR_INTERNAL,
    ACCEPT_ERROR_ACCEPT_FAILED,
];
pub const ACCEPT_DESCRIPTOR: BuiltinDescriptor = BuiltinDescriptor {
    signatures: &ACCEPT_SIGNATURES,
    output_mode: BuiltinOutputMode::Fixed,
    completion_policy: BuiltinCompletionPolicy::Public,
    errors: &ACCEPT_ERRORS,
};

const ACCEPT_TIMEOUT_INTEGER_INPUTS: [BuiltinIntegerInputCapability; 1] =
    [BuiltinIntegerInputCapability {
        name: "Timeout",
        classes: &crate::ALL_INTEGER_CLASSES,
        availability: BuiltinIntegerInputAvailability::Documented,
        scalar_double: BuiltinIntegerScalarDoubleRule::Allowed,
        notes: "RunMat's safe additive accept API documents a nonnegative scalar timeout in every integer class, or a floating scalar including positive Inf.",
    }];
pub const ACCEPT_INTEGER_CAPABILITIES: [BuiltinIntegerCapabilityDescriptor; 1] =
    [BuiltinIntegerCapabilityDescriptor {
        form: "client = accept(server, \"Timeout\", integer_timeout)",
        inputs: &ACCEPT_TIMEOUT_INTEGER_INPUTS,
        computation_domain: BuiltinIntegerComputationDomain::FloatingPoint,
        output_class: BuiltinIntegerOutputClassRule::NotApplicable,
        overflow: BuiltinIntegerOverflowRule::Error,
        backend: BuiltinIntegerBackendRule::GatherFallback,
        overload: BuiltinIntegerOverloadKind::ScalarOnly,
        notes: "The timeout crosses one explicit binary64-seconds boundary and must fit std::time::Duration; networking and its returned client handle remain host-only, while resident scalar controls gather before validation.",
    }];
