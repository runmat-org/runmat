use crate::*;
pub const INTEGER_PORT_EXTENSION: BuiltinExtensionDescriptor = BuiltinExtensionDescriptor {
    id: "tcpserver-integer-port",
    mode: BuiltinExtensionMode::RunMatOnly,
    description: "tcpserver with a typed-integer port is a RunMat extension",
    error_identifier: Some("RunMat:compatibility:TcpserverIntegerPortExtension"),
};
pub const INTEGER_TIMEOUT_EXTENSION: BuiltinExtensionDescriptor = BuiltinExtensionDescriptor {
    id: "tcpserver-integer-timeout",
    mode: BuiltinExtensionMode::RunMatOnly,
    description: "tcpserver with a typed-integer Timeout is a RunMat extension",
    error_identifier: Some("RunMat:compatibility:TcpserverIntegerTimeoutExtension"),
};
pub const ZERO_PORT_EXTENSION: BuiltinExtensionDescriptor = BuiltinExtensionDescriptor {
    id: "tcpserver-zero-port",
    mode: BuiltinExtensionMode::RunMatOnly,
    description: "tcpserver ephemeral port selection with port zero is a RunMat extension",
    error_identifier: Some("RunMat:compatibility:TcpserverZeroPortExtension"),
};
pub const LEGACY_OPTIONS_EXTENSION: BuiltinExtensionDescriptor = BuiltinExtensionDescriptor {
    id: "tcpserver-legacy-constructor-options",
    mode: BuiltinExtensionMode::RunMatOnly,
    description: "tcpserver legacy constructor options are a RunMat extension",
    error_identifier: Some("RunMat:compatibility:TcpserverLegacyOptionsExtension"),
};
pub const EXPLICIT_GPU_INPUT_EXTENSION: BuiltinExtensionDescriptor = BuiltinExtensionDescriptor {
    id: "tcpserver-explicit-gpu-input",
    mode: BuiltinExtensionMode::RunMatOnly,
    description: "tcpserver with explicit gpuArray input is a RunMat extension",
    error_identifier: Some("RunMat:compatibility:TcpserverExplicitGpuInputExtension"),
};
pub const TCPSERVER_EXTENSIONS: [BuiltinExtensionDescriptor; 5] = [
    INTEGER_PORT_EXTENSION,
    INTEGER_TIMEOUT_EXTENSION,
    ZERO_PORT_EXTENSION,
    LEGACY_OPTIONS_EXTENSION,
    EXPLICIT_GPU_INPUT_EXTENSION,
];

const INTEGER_PORT_INPUT: [BuiltinIntegerInputCapability; 1] = [BuiltinIntegerInputCapability {
    name: "port",
    classes: &crate::ALL_INTEGER_CLASSES,
    availability: BuiltinIntegerInputAvailability::RunMatOnly,
    scalar_double: BuiltinIntegerScalarDoubleRule::Allowed,
    notes: "The public ServerPort datatype is double. RunMat mode additionally decodes all eight typed integer classes directly into the validated 1..65535 structural range.",
}];
const INTEGER_TIMEOUT_INPUT: [BuiltinIntegerInputCapability; 1] = [BuiltinIntegerInputCapability {
    name: "Timeout",
    classes: &crate::ALL_INTEGER_CLASSES,
    availability: BuiltinIntegerInputAvailability::RunMatOnly,
    scalar_double: BuiltinIntegerScalarDoubleRule::Allowed,
    notes: "The public Timeout datatype is double; typed integers cross a checked binary64-seconds boundary only in RunMat mode.",
}];
pub const TCPSERVER_INTEGER_CAPABILITIES: [BuiltinIntegerCapabilityDescriptor; 2] = [
    BuiltinIntegerCapabilityDescriptor {
        form: "server = tcpserver(address, integer_port, ...)",
        inputs: &INTEGER_PORT_INPUT,
        computation_domain: BuiltinIntegerComputationDomain::Structural,
        output_class: BuiltinIntegerOutputClassRule::FunctionSpecific,
        overflow: BuiltinIntegerOverflowRule::Error,
        backend: BuiltinIntegerBackendRule::GatherFallback,
        overload: BuiltinIntegerOverloadKind::ScalarOnly,
        notes: "The port is range-checked from authoritative integer storage before socket binding. Public object properties remain double-valued.",
    },
    BuiltinIntegerCapabilityDescriptor {
        form: "server = tcpserver(..., Timeout=integer_timeout)",
        inputs: &INTEGER_TIMEOUT_INPUT,
        computation_domain: BuiltinIntegerComputationDomain::FloatingPoint,
        output_class: BuiltinIntegerOutputClassRule::FunctionSpecific,
        overflow: BuiltinIntegerOverflowRule::Error,
        backend: BuiltinIntegerBackendRule::GatherFallback,
        overload: BuiltinIntegerOverloadKind::ScalarOnly,
        notes: "Typed Timeout is gated before provider access and must be exactly representable as binary64 before host networking observes it.",
    },
];

const TCPSERVER_OUTPUT_SERVER: [BuiltinParamDescriptor; 1] = [BuiltinParamDescriptor {
    name: "server",
    ty: BuiltinParamType::Any,
    arity: BuiltinParamArity::Required,
    default: None,
    description: "tcpserver object with sealed authority for accept/close operations.",
}];
const TCPSERVER_INPUTS_ADDRESS_PORT: [BuiltinParamDescriptor; 2] = [
    BuiltinParamDescriptor {
        name: "address",
        ty: BuiltinParamType::StringScalar,
        arity: BuiltinParamArity::Required,
        default: None,
        description: "Bind address or hostname.",
    },
    BuiltinParamDescriptor {
        name: "port",
        ty: BuiltinParamType::NumericScalar,
        arity: BuiltinParamArity::Required,
        default: None,
        description: "Listening TCP port (0..65535).",
    },
];
const TCPSERVER_INPUTS_ADDRESS_PORT_NAME_VALUE: [BuiltinParamDescriptor; 3] = [
    BuiltinParamDescriptor {
        name: "address",
        ty: BuiltinParamType::StringScalar,
        arity: BuiltinParamArity::Required,
        default: None,
        description: "Bind address or hostname.",
    },
    BuiltinParamDescriptor {
        name: "port",
        ty: BuiltinParamType::NumericScalar,
        arity: BuiltinParamArity::Required,
        default: None,
        description: "Listening TCP port (0..65535).",
    },
    BuiltinParamDescriptor {
        name: "name_value_pairs",
        ty: BuiltinParamType::Any,
        arity: BuiltinParamArity::Variadic,
        default: None,
        description: "Name/Value options such as Timeout, UserData, Name, and ByteOrder.",
    },
];
const TCPSERVER_SIGNATURES: [BuiltinSignatureDescriptor; 2] = [
    BuiltinSignatureDescriptor {
        label: "server = tcpserver(address, port)",
        inputs: &TCPSERVER_INPUTS_ADDRESS_PORT,
        outputs: &TCPSERVER_OUTPUT_SERVER,
    },
    BuiltinSignatureDescriptor {
        label: "server = tcpserver(address, port, Name, Value, ...)",
        inputs: &TCPSERVER_INPUTS_ADDRESS_PORT_NAME_VALUE,
        outputs: &TCPSERVER_OUTPUT_SERVER,
    },
];

pub const TCPSERVER_ERROR_INVALID_ADDRESS: BuiltinErrorDescriptor = BuiltinErrorDescriptor {
    code: "RM.TCPSERVER.INVALID_ADDRESS",
    identifier: Some("RunMat:tcpserver:InvalidAddress"),
    when: "Address argument is not a valid string scalar.",
    message: "tcpserver: invalid address argument",
};
pub const TCPSERVER_ERROR_INVALID_PORT: BuiltinErrorDescriptor = BuiltinErrorDescriptor {
    code: "RM.TCPSERVER.INVALID_PORT",
    identifier: Some("RunMat:tcpserver:InvalidPort"),
    when: "Port argument is non-scalar, non-integer, non-finite, or out of range.",
    message: "tcpserver: invalid port argument",
};
pub const TCPSERVER_ERROR_INVALID_NAME_VALUE: BuiltinErrorDescriptor = BuiltinErrorDescriptor {
    code: "RM.TCPSERVER.INVALID_NAME_VALUE",
    identifier: Some("RunMat:tcpserver:InvalidNameValue"),
    when: "Name/Value arguments are malformed, unsupported, or have invalid values.",
    message: "tcpserver: invalid name-value arguments",
};
pub const TCPSERVER_ERROR_BIND_FAILED: BuiltinErrorDescriptor = BuiltinErrorDescriptor {
    code: "RM.TCPSERVER.BIND_FAILED",
    identifier: Some("RunMat:tcpserver:BindFailed"),
    when: "Socket bind operation fails.",
    message: "tcpserver: unable to bind listener",
};
pub const TCPSERVER_ERROR_INTERNAL: BuiltinErrorDescriptor = BuiltinErrorDescriptor {
    code: "RM.TCPSERVER.INTERNAL",
    identifier: Some("RunMat:tcpserver:InternalError"),
    when: "Internal listener metadata retrieval fails.",
    message: "tcpserver: internal error",
};
const TCPSERVER_ERRORS: [BuiltinErrorDescriptor; 7] = [
    super::super::NET_ERROR_UNSUPPORTED_TARGET,
    super::super::capability::NET_ERROR_CANCELLED,
    TCPSERVER_ERROR_INVALID_ADDRESS,
    TCPSERVER_ERROR_INVALID_PORT,
    TCPSERVER_ERROR_INVALID_NAME_VALUE,
    TCPSERVER_ERROR_BIND_FAILED,
    TCPSERVER_ERROR_INTERNAL,
];
pub const TCPSERVER_DESCRIPTOR: BuiltinDescriptor = BuiltinDescriptor {
    signatures: &TCPSERVER_SIGNATURES,
    output_mode: BuiltinOutputMode::Fixed,
    completion_policy: BuiltinCompletionPolicy::Public,
    errors: &TCPSERVER_ERRORS,
};
