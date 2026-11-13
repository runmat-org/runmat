use crate::*;

pub const READ_OUTPUT_DATA: [BuiltinParamDescriptor; 1] = [BuiltinParamDescriptor {
    name: "data",
    ty: BuiltinParamType::Any,
    arity: BuiltinParamArity::Required,
    default: None,
    description: "Available bytes or requested typed payload from the TCP client.",
}];
pub const READ_INPUTS_CLIENT: [BuiltinParamDescriptor; 1] = [BuiltinParamDescriptor {
    name: "client",
    ty: BuiltinParamType::Any,
    arity: BuiltinParamArity::Required,
    default: None,
    description: "tcpclient object with sealed session-owned socket authority.",
}];
pub const READ_INPUTS_CLIENT_COUNT: [BuiltinParamDescriptor; 2] = [
    BuiltinParamDescriptor {
        name: "client",
        ty: BuiltinParamType::Any,
        arity: BuiltinParamArity::Required,
        default: None,
        description: "tcpclient object with sealed session-owned socket authority.",
    },
    BuiltinParamDescriptor {
        name: "count",
        ty: BuiltinParamType::NumericScalar,
        arity: BuiltinParamArity::Required,
        default: None,
        description: "Requested element count.",
    },
];
pub const READ_INPUTS_CLIENT_COUNT_DATATYPE: [BuiltinParamDescriptor; 3] = [
    BuiltinParamDescriptor {
        name: "client",
        ty: BuiltinParamType::Any,
        arity: BuiltinParamArity::Required,
        default: None,
        description: "tcpclient object with sealed session-owned socket authority.",
    },
    BuiltinParamDescriptor {
        name: "count",
        ty: BuiltinParamType::NumericScalar,
        arity: BuiltinParamArity::Required,
        default: None,
        description: "Requested element count.",
    },
    BuiltinParamDescriptor {
        name: "datatype",
        ty: BuiltinParamType::StringScalar,
        arity: BuiltinParamArity::Optional,
        default: Some("\"uint8\""),
        description: "Data type label (for example \"uint8\", \"double\", \"char\", \"string\").",
    },
];
pub const READ_SIGNATURES: [BuiltinSignatureDescriptor; 3] = [
    BuiltinSignatureDescriptor {
        label: "data = read(client)",
        inputs: &READ_INPUTS_CLIENT,
        outputs: &READ_OUTPUT_DATA,
    },
    BuiltinSignatureDescriptor {
        label: "data = read(client, count)",
        inputs: &READ_INPUTS_CLIENT_COUNT,
        outputs: &READ_OUTPUT_DATA,
    },
    BuiltinSignatureDescriptor {
        label: "data = read(client, count, datatype)",
        inputs: &READ_INPUTS_CLIENT_COUNT_DATATYPE,
        outputs: &READ_OUTPUT_DATA,
    },
];

pub const READ_ERROR_INVALID_CLIENT: BuiltinErrorDescriptor = BuiltinErrorDescriptor {
    code: "RM.READ.INVALID_CLIENT",
    identifier: Some("RunMat:read:InvalidTcpClient"),
    when: "Client handle is missing, malformed, invalid, or disconnected.",
    message: "read: invalid tcpclient handle",
};
pub const READ_ERROR_INVALID_INPUT: BuiltinErrorDescriptor = BuiltinErrorDescriptor {
    code: "RM.READ.INVALID_INPUT",
    identifier: Some("RunMat:read:InvalidInput"),
    when: "Argument list shape is unsupported for read.",
    message: "read: invalid argument list",
};
pub const READ_ERROR_NOT_CONNECTED: BuiltinErrorDescriptor = BuiltinErrorDescriptor {
    code: "RM.READ.NOT_CONNECTED",
    identifier: Some("RunMat:read:NotConnected"),
    when: "Client has no active socket connection.",
    message: "read: tcpclient is disconnected",
};
pub const READ_ERROR_TIMEOUT: BuiltinErrorDescriptor = BuiltinErrorDescriptor {
    code: "RM.READ.TIMEOUT",
    identifier: Some("RunMat:read:Timeout"),
    when: "Socket read exceeds configured timeout.",
    message: "read: timed out waiting for data",
};
pub const READ_ERROR_CONNECTION_CLOSED: BuiltinErrorDescriptor = BuiltinErrorDescriptor {
    code: "RM.READ.CONNECTION_CLOSED",
    identifier: Some("RunMat:read:ConnectionClosed"),
    when: "Peer closes socket before requested payload is fully available.",
    message: "read: connection closed before the requested data was received",
};
pub const READ_ERROR_INVALID_COUNT: BuiltinErrorDescriptor = BuiltinErrorDescriptor {
    code: "RM.READ.INVALID_COUNT",
    identifier: Some("RunMat:read:InvalidCount"),
    when: "Requested count is non-scalar, negative, non-finite, or out of range.",
    message: "read: invalid count argument",
};
pub const READ_ERROR_INVALID_DATATYPE: BuiltinErrorDescriptor = BuiltinErrorDescriptor {
    code: "RM.READ.INVALID_DATATYPE",
    identifier: Some("RunMat:read:InvalidDataType"),
    when: "Datatype argument is not a supported scalar text label.",
    message: "read: invalid datatype argument",
};
pub const READ_ERROR_INTERNAL: BuiltinErrorDescriptor = BuiltinErrorDescriptor {
    code: "RM.READ.INTERNAL",
    identifier: Some("RunMat:read:InternalError"),
    when: "Internal socket/control-flow conversion fails.",
    message: "read: internal socket error",
};
pub const READ_ERRORS: [BuiltinErrorDescriptor; 10] = [
    super::super::NET_ERROR_UNSUPPORTED_TARGET,
    super::super::capability::NET_ERROR_CANCELLED,
    READ_ERROR_INVALID_CLIENT,
    READ_ERROR_INVALID_INPUT,
    READ_ERROR_NOT_CONNECTED,
    READ_ERROR_TIMEOUT,
    READ_ERROR_CONNECTION_CLOSED,
    READ_ERROR_INVALID_COUNT,
    READ_ERROR_INVALID_DATATYPE,
    READ_ERROR_INTERNAL,
];
pub const READ_DESCRIPTOR: BuiltinDescriptor = BuiltinDescriptor {
    signatures: &READ_SIGNATURES,
    output_mode: BuiltinOutputMode::Fixed,
    completion_policy: BuiltinCompletionPolicy::Public,
    errors: &READ_ERRORS,
};

pub const READ_INTEGER_COUNT_INPUT: [BuiltinIntegerInputCapability; 1] =
    [BuiltinIntegerInputCapability {
        name: "count",
        classes: &crate::ALL_INTEGER_CLASSES,
        availability: BuiltinIntegerInputAvailability::Documented,
        scalar_double: BuiltinIntegerScalarDoubleRule::Allowed,
        notes: "All eight integer classes are documented count inputs. RunMat reads the authoritative scalar exactly, requires a positive value representable as usize, and never routes the count through binary64.",
    }];
pub const READ_NO_INTEGER_INPUTS: [BuiltinIntegerInputCapability; 0] = [];
pub const READ_INTEGER_CAPABILITIES: [BuiltinIntegerCapabilityDescriptor; 3] = [
    BuiltinIntegerCapabilityDescriptor {
        form: "data = read(client)",
        inputs: &READ_NO_INTEGER_INPUTS,
        computation_domain: BuiltinIntegerComputationDomain::ExactInteger,
        output_class: BuiltinIntegerOutputClassRule::FunctionSpecific,
        overflow: BuiltinIntegerOverflowRule::NotApplicable,
        backend: BuiltinIntegerBackendRule::HostOnly,
        overload: BuiltinIntegerOverloadKind::FunctionSpecific,
        notes: "The current public worked example, default-datatype table, and transfer note converge on an exact uint8 result when datatype is omitted, despite one contradictory prose sentence on the same page.",
    },
    BuiltinIntegerCapabilityDescriptor {
        form: "data = read(client, integer_count)",
        inputs: &READ_INTEGER_COUNT_INPUT,
        computation_domain: BuiltinIntegerComputationDomain::Structural,
        output_class: BuiltinIntegerOutputClassRule::FunctionSpecific,
        overflow: BuiltinIntegerOverflowRule::Error,
        backend: BuiltinIntegerBackendRule::HostOnly,
        overload: BuiltinIntegerOverloadKind::StructuralParameter,
        notes: "The default result is exact uint8. Networking executes on the host; resident count controls gather automatically before exact validation.",
    },
    BuiltinIntegerCapabilityDescriptor {
        form: "data = read(client, integer_count, integer_datatype)",
        inputs: &READ_INTEGER_COUNT_INPUT,
        computation_domain: BuiltinIntegerComputationDomain::ExactInteger,
        output_class: BuiltinIntegerOutputClassRule::OptionDependent,
        overflow: BuiltinIntegerOverflowRule::Error,
        backend: BuiltinIntegerBackendRule::HostOnly,
        overload: BuiltinIntegerOverloadKind::FunctionSpecific,
        notes: "An explicit integer datatype returns that exact integer class in a host row vector, with byte order decoded directly into the requested class.",
    },
];
