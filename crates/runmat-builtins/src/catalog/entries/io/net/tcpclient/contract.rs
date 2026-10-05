use crate::*;
pub const INTEGER_PORT_EXTENSION: BuiltinExtensionDescriptor = BuiltinExtensionDescriptor {
    id: "tcpclient-integer-port",
    mode: BuiltinExtensionMode::RunMatOnly,
    description: "tcpclient with a typed-integer port is a RunMat extension",
    error_identifier: Some("RunMat:compatibility:TcpclientIntegerPortExtension"),
};
pub const INTEGER_TIMEOUT_EXTENSION: BuiltinExtensionDescriptor = BuiltinExtensionDescriptor {
    id: "tcpclient-integer-timeout",
    mode: BuiltinExtensionMode::RunMatOnly,
    description: "tcpclient with a typed-integer Timeout is a RunMat extension",
    error_identifier: Some("RunMat:compatibility:TcpclientIntegerTimeoutExtension"),
};
pub const INTEGER_CONNECT_TIMEOUT_EXTENSION: BuiltinExtensionDescriptor =
    BuiltinExtensionDescriptor {
        id: "tcpclient-integer-connect-timeout",
        mode: BuiltinExtensionMode::RunMatOnly,
        description: "tcpclient with a typed-integer ConnectTimeout is a RunMat extension",
        error_identifier: Some("RunMat:compatibility:TcpclientIntegerConnectTimeoutExtension"),
    };
pub const ZERO_PORT_EXTENSION: BuiltinExtensionDescriptor = BuiltinExtensionDescriptor {
    id: "tcpclient-zero-port",
    mode: BuiltinExtensionMode::RunMatOnly,
    description: "tcpclient port zero is a RunMat extension",
    error_identifier: Some("RunMat:compatibility:TcpclientZeroPortExtension"),
};
pub const LEGACY_OPTIONS_EXTENSION: BuiltinExtensionDescriptor = BuiltinExtensionDescriptor {
    id: "tcpclient-legacy-constructor-options",
    mode: BuiltinExtensionMode::RunMatOnly,
    description: "tcpclient legacy constructor options are a RunMat extension",
    error_identifier: Some("RunMat:compatibility:TcpclientLegacyOptionsExtension"),
};
pub const EXPLICIT_GPU_INPUT_EXTENSION: BuiltinExtensionDescriptor = BuiltinExtensionDescriptor {
    id: "tcpclient-explicit-gpu-input",
    mode: BuiltinExtensionMode::RunMatOnly,
    description: "tcpclient with explicit gpuArray input is a RunMat extension",
    error_identifier: Some("RunMat:compatibility:TcpclientExplicitGpuInputExtension"),
};
pub const TCPCLIENT_EXTENSIONS: [BuiltinExtensionDescriptor; 6] = [
    INTEGER_PORT_EXTENSION,
    INTEGER_TIMEOUT_EXTENSION,
    INTEGER_CONNECT_TIMEOUT_EXTENSION,
    ZERO_PORT_EXTENSION,
    LEGACY_OPTIONS_EXTENSION,
    EXPLICIT_GPU_INPUT_EXTENSION,
];

const INTEGER_PORT_INPUT: [BuiltinIntegerInputCapability; 1] = [BuiltinIntegerInputCapability {
    name: "port",
    classes: &crate::ALL_INTEGER_CLASSES,
    availability: BuiltinIntegerInputAvailability::RunMatOnly,
    scalar_double: BuiltinIntegerScalarDoubleRule::Allowed,
    notes: "The public Port datatype is double. RunMat mode additionally decodes all eight typed integer classes directly into the validated 1..65535 structural range.",
}];
const INTEGER_TIMEOUT_INPUTS: [BuiltinIntegerInputCapability; 2] = [
    BuiltinIntegerInputCapability {
        name: "Timeout",
        classes: &crate::ALL_INTEGER_CLASSES,
        availability: BuiltinIntegerInputAvailability::RunMatOnly,
        scalar_double: BuiltinIntegerScalarDoubleRule::Allowed,
        notes: "The public Timeout datatype is double; typed integers cross a checked binary64-seconds boundary only in RunMat mode.",
    },
    BuiltinIntegerInputCapability {
        name: "ConnectTimeout",
        classes: &crate::ALL_INTEGER_CLASSES,
        availability: BuiltinIntegerInputAvailability::RunMatOnly,
        scalar_double: BuiltinIntegerScalarDoubleRule::Allowed,
        notes: "The public ConnectTimeout datatype is double; typed integers cross a checked binary64-seconds boundary only in RunMat mode.",
    },
];
const INTEGER_BUFFER_INPUTS: [BuiltinIntegerInputCapability; 2] = [
    BuiltinIntegerInputCapability {
        name: "InputBufferSize",
        classes: &crate::ALL_INTEGER_CLASSES,
        availability: BuiltinIntegerInputAvailability::RunMatOnly,
        scalar_double: BuiltinIntegerScalarDoubleRule::Allowed,
        notes: "InputBufferSize is part of RunMat's gated legacy constructor-option surface and is decoded exactly into a bounded host size.",
    },
    BuiltinIntegerInputCapability {
        name: "OutputBufferSize",
        classes: &crate::ALL_INTEGER_CLASSES,
        availability: BuiltinIntegerInputAvailability::RunMatOnly,
        scalar_double: BuiltinIntegerScalarDoubleRule::Allowed,
        notes: "OutputBufferSize is part of RunMat's gated legacy constructor-option surface and is decoded exactly into a bounded host size.",
    },
];
pub const TCPCLIENT_INTEGER_CAPABILITIES: [BuiltinIntegerCapabilityDescriptor; 3] = [
    BuiltinIntegerCapabilityDescriptor {
        form: "client = tcpclient(address, integer_port, ...)",
        inputs: &INTEGER_PORT_INPUT,
        computation_domain: BuiltinIntegerComputationDomain::Structural,
        output_class: BuiltinIntegerOutputClassRule::FunctionSpecific,
        overflow: BuiltinIntegerOverflowRule::Error,
        backend: BuiltinIntegerBackendRule::GatherFallback,
        overload: BuiltinIntegerOverloadKind::ScalarOnly,
        notes: "The port is range-checked from authoritative integer storage before any socket connection. Public object properties remain double-valued.",
    },
    BuiltinIntegerCapabilityDescriptor {
        form: "client = tcpclient(..., Timeout=integer_timeout, ConnectTimeout=integer_connect_timeout)",
        inputs: &INTEGER_TIMEOUT_INPUTS,
        computation_domain: BuiltinIntegerComputationDomain::FloatingPoint,
        output_class: BuiltinIntegerOutputClassRule::FunctionSpecific,
        overflow: BuiltinIntegerOverflowRule::Error,
        backend: BuiltinIntegerBackendRule::GatherFallback,
        overload: BuiltinIntegerOverloadKind::ScalarOnly,
        notes: "Each typed timeout is independently gated and must be exactly representable as binary64 before host networking observes it.",
    },
    BuiltinIntegerCapabilityDescriptor {
        form: "client = tcpclient(..., InputBufferSize=integer_n, OutputBufferSize=integer_n)",
        inputs: &INTEGER_BUFFER_INPUTS,
        computation_domain: BuiltinIntegerComputationDomain::Structural,
        output_class: BuiltinIntegerOutputClassRule::FunctionSpecific,
        overflow: BuiltinIntegerOverflowRule::Error,
        backend: BuiltinIntegerBackendRule::GatherFallback,
        overload: BuiltinIntegerOverloadKind::ScalarOnly,
        notes: "Legacy buffer controls are gated before provider access and decoded exactly without a floating round trip.",
    },
];

const TCPCLIENT_OUTPUT_CLIENT: [BuiltinParamDescriptor; 1] = [BuiltinParamDescriptor {
    name: "client",
    ty: BuiltinParamType::Any,
    arity: BuiltinParamArity::Required,
    default: None,
    description: "tcpclient object with sealed authority for read/write/close operations.",
}];
const TCPCLIENT_INPUTS_HOST_PORT: [BuiltinParamDescriptor; 2] = [
    BuiltinParamDescriptor {
        name: "host",
        ty: BuiltinParamType::StringScalar,
        arity: BuiltinParamArity::Required,
        default: None,
        description: "Server hostname or IP address.",
    },
    BuiltinParamDescriptor {
        name: "port",
        ty: BuiltinParamType::NumericScalar,
        arity: BuiltinParamArity::Required,
        default: None,
        description: "Server TCP port (0..65535).",
    },
];
const TCPCLIENT_INPUTS_HOST_PORT_NAME_VALUE: [BuiltinParamDescriptor; 3] = [
    BuiltinParamDescriptor {
        name: "host",
        ty: BuiltinParamType::StringScalar,
        arity: BuiltinParamArity::Required,
        default: None,
        description: "Server hostname or IP address.",
    },
    BuiltinParamDescriptor {
        name: "port",
        ty: BuiltinParamType::NumericScalar,
        arity: BuiltinParamArity::Required,
        default: None,
        description: "Server TCP port (0..65535).",
    },
    BuiltinParamDescriptor {
        name: "name_value_pairs",
        ty: BuiltinParamType::Any,
        arity: BuiltinParamArity::Variadic,
        default: None,
        description:
            "Name/Value options such as Timeout, ConnectTimeout, ByteOrder, UserData, Name, InputBufferSize, and OutputBufferSize.",
    },
];
const TCPCLIENT_SIGNATURES: [BuiltinSignatureDescriptor; 2] = [
    BuiltinSignatureDescriptor {
        label: "client = tcpclient(host, port)",
        inputs: &TCPCLIENT_INPUTS_HOST_PORT,
        outputs: &TCPCLIENT_OUTPUT_CLIENT,
    },
    BuiltinSignatureDescriptor {
        label: "client = tcpclient(host, port, Name, Value, ...)",
        inputs: &TCPCLIENT_INPUTS_HOST_PORT_NAME_VALUE,
        outputs: &TCPCLIENT_OUTPUT_CLIENT,
    },
];

pub const TCPCLIENT_ERROR_INVALID_ADDRESS: BuiltinErrorDescriptor = BuiltinErrorDescriptor {
    code: "RM.TCPCLIENT.INVALID_ADDRESS",
    identifier: Some("RunMat:tcpclient:InvalidAddress"),
    when: "Host/address argument is not a valid string scalar.",
    message: "tcpclient: invalid host argument",
};
pub const TCPCLIENT_ERROR_INVALID_PORT: BuiltinErrorDescriptor = BuiltinErrorDescriptor {
    code: "RM.TCPCLIENT.INVALID_PORT",
    identifier: Some("RunMat:tcpclient:InvalidPort"),
    when: "Port argument is non-scalar, non-integer, non-finite, or out of range.",
    message: "tcpclient: invalid port argument",
};
pub const TCPCLIENT_ERROR_INVALID_NAME_VALUE: BuiltinErrorDescriptor = BuiltinErrorDescriptor {
    code: "RM.TCPCLIENT.INVALID_NAME_VALUE",
    identifier: Some("RunMat:tcpclient:InvalidNameValue"),
    when: "Name/Value arguments are malformed, unsupported, or have invalid values.",
    message: "tcpclient: invalid name-value arguments",
};
pub const TCPCLIENT_ERROR_CONNECT_FAILED: BuiltinErrorDescriptor = BuiltinErrorDescriptor {
    code: "RM.TCPCLIENT.CONNECT_FAILED",
    identifier: Some("RunMat:tcpclient:ConnectionFailed"),
    when: "Socket connect attempt fails.",
    message: "tcpclient: unable to connect",
};
pub const TCPCLIENT_ERROR_INTERNAL: BuiltinErrorDescriptor = BuiltinErrorDescriptor {
    code: "RM.TCPCLIENT.INTERNAL",
    identifier: Some("RunMat:tcpclient:InternalError"),
    when: "Internal stream setup or metadata query fails.",
    message: "tcpclient: internal error",
};
const TCPCLIENT_ERRORS: [BuiltinErrorDescriptor; 7] = [
    super::super::NET_ERROR_UNSUPPORTED_TARGET,
    super::super::capability::NET_ERROR_CANCELLED,
    TCPCLIENT_ERROR_INVALID_ADDRESS,
    TCPCLIENT_ERROR_INVALID_PORT,
    TCPCLIENT_ERROR_INVALID_NAME_VALUE,
    TCPCLIENT_ERROR_CONNECT_FAILED,
    TCPCLIENT_ERROR_INTERNAL,
];
pub const TCPCLIENT_DESCRIPTOR: BuiltinDescriptor = BuiltinDescriptor {
    signatures: &TCPCLIENT_SIGNATURES,
    output_mode: BuiltinOutputMode::Fixed,
    completion_policy: BuiltinCompletionPolicy::Public,
    errors: &TCPCLIENT_ERRORS,
};
