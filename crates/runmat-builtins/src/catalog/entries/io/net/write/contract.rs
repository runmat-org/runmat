use crate::*;
const WRITE_OUTPUT_COUNT: [BuiltinParamDescriptor; 1] = [BuiltinParamDescriptor {
    name: "count",
    ty: BuiltinParamType::NumericScalar,
    arity: BuiltinParamArity::Required,
    default: None,
    description: "Number of elements written to the socket.",
}];
const WRITE_INPUTS_CLIENT_DATA: [BuiltinParamDescriptor; 2] = [
    BuiltinParamDescriptor {
        name: "client",
        ty: BuiltinParamType::Any,
        arity: BuiltinParamArity::Required,
        default: None,
        description: "tcpclient object with sealed session-owned socket authority.",
    },
    BuiltinParamDescriptor {
        name: "data",
        ty: BuiltinParamType::Any,
        arity: BuiltinParamArity::Required,
        default: None,
        description: "Payload to send.",
    },
];
const WRITE_INPUTS_CLIENT_DATA_DATATYPE: [BuiltinParamDescriptor; 3] = [
    BuiltinParamDescriptor {
        name: "client",
        ty: BuiltinParamType::Any,
        arity: BuiltinParamArity::Required,
        default: None,
        description: "tcpclient object with sealed session-owned socket authority.",
    },
    BuiltinParamDescriptor {
        name: "data",
        ty: BuiltinParamType::Any,
        arity: BuiltinParamArity::Required,
        default: None,
        description: "Payload to send.",
    },
    BuiltinParamDescriptor {
        name: "datatype",
        ty: BuiltinParamType::StringScalar,
        arity: BuiltinParamArity::Optional,
        default: Some("class(data)"),
        description: "Data type label (for example \"uint8\", \"double\", \"char\", \"string\").",
    },
];
const WRITE_SIGNATURES: [BuiltinSignatureDescriptor; 2] = [
    BuiltinSignatureDescriptor {
        label: "count = write(client, data)",
        inputs: &WRITE_INPUTS_CLIENT_DATA,
        outputs: &WRITE_OUTPUT_COUNT,
    },
    BuiltinSignatureDescriptor {
        label: "count = write(client, data, datatype)",
        inputs: &WRITE_INPUTS_CLIENT_DATA_DATATYPE,
        outputs: &WRITE_OUTPUT_COUNT,
    },
];

pub const WRITE_ERROR_INVALID_CLIENT: BuiltinErrorDescriptor = BuiltinErrorDescriptor {
    code: "RM.WRITE.INVALID_CLIENT",
    identifier: Some("RunMat:write:InvalidTcpClient"),
    when: "Client handle is missing, malformed, invalid, or disconnected.",
    message: "write: invalid tcpclient handle",
};
pub const WRITE_ERROR_INVALID_INPUT: BuiltinErrorDescriptor = BuiltinErrorDescriptor {
    code: "RM.WRITE.INVALID_INPUT",
    identifier: Some("RunMat:write:InvalidInput"),
    when: "Argument list shape is unsupported for write.",
    message: "write: invalid argument list",
};
pub const WRITE_ERROR_INVALID_DATA: BuiltinErrorDescriptor = BuiltinErrorDescriptor {
    code: "RM.WRITE.INVALID_DATA",
    identifier: Some("RunMat:write:InvalidData"),
    when: "Payload cannot be converted to the requested datatype.",
    message: "write: invalid data payload",
};
pub const WRITE_ERROR_INVALID_DATATYPE: BuiltinErrorDescriptor = BuiltinErrorDescriptor {
    code: "RM.WRITE.INVALID_DATATYPE",
    identifier: Some("RunMat:write:InvalidDataType"),
    when: "Datatype argument is not a supported scalar text label.",
    message: "write: invalid datatype argument",
};
pub const WRITE_ERROR_NOT_CONNECTED: BuiltinErrorDescriptor = BuiltinErrorDescriptor {
    code: "RM.WRITE.NOT_CONNECTED",
    identifier: Some("RunMat:write:NotConnected"),
    when: "Client has no active socket connection.",
    message: "write: tcpclient is disconnected",
};
pub const WRITE_ERROR_TIMEOUT: BuiltinErrorDescriptor = BuiltinErrorDescriptor {
    code: "RM.WRITE.TIMEOUT",
    identifier: Some("RunMat:write:Timeout"),
    when: "Socket write exceeds configured timeout.",
    message: "write: timed out while sending data",
};
pub const WRITE_ERROR_CONNECTION_CLOSED: BuiltinErrorDescriptor = BuiltinErrorDescriptor {
    code: "RM.WRITE.CONNECTION_CLOSED",
    identifier: Some("RunMat:write:ConnectionClosed"),
    when: "Peer closes socket before payload is fully written.",
    message: "write: connection closed before all data was sent",
};
pub const WRITE_ERROR_INTERNAL: BuiltinErrorDescriptor = BuiltinErrorDescriptor {
    code: "RM.WRITE.INTERNAL",
    identifier: Some("RunMat:write:InternalError"),
    when: "Internal socket/control-flow conversion fails.",
    message: "write: internal socket error",
};
const WRITE_ERRORS: [BuiltinErrorDescriptor; 10] = [
    super::super::NET_ERROR_UNSUPPORTED_TARGET,
    super::super::capability::NET_ERROR_CANCELLED,
    WRITE_ERROR_INVALID_CLIENT,
    WRITE_ERROR_INVALID_INPUT,
    WRITE_ERROR_INVALID_DATA,
    WRITE_ERROR_INVALID_DATATYPE,
    WRITE_ERROR_NOT_CONNECTED,
    WRITE_ERROR_TIMEOUT,
    WRITE_ERROR_CONNECTION_CLOSED,
    WRITE_ERROR_INTERNAL,
];
pub const WRITE_DESCRIPTOR: BuiltinDescriptor = BuiltinDescriptor {
    signatures: &WRITE_SIGNATURES,
    output_mode: BuiltinOutputMode::Fixed,
    completion_policy: BuiltinCompletionPolicy::Public,
    errors: &WRITE_ERRORS,
};

pub const WRITE_EXPLICIT_GPU_EXTENSION: BuiltinExtensionDescriptor = BuiltinExtensionDescriptor {
    id: "write-explicit-gpu-data",
    mode: BuiltinExtensionMode::RunMatOnly,
    description: "write with explicit gpuArray data is a RunMat extension",
    error_identifier: Some("RunMat:compatibility:WriteExplicitGpuDataExtension"),
};
pub const WRITE_NONVECTOR_EXTENSION: BuiltinExtensionDescriptor = BuiltinExtensionDescriptor {
    id: "write-nonvector-data",
    mode: BuiltinExtensionMode::RunMatOnly,
    description: "write with matrix or multidimensional numeric data is a RunMat extension",
    error_identifier: Some("RunMat:compatibility:WriteNonvectorDataExtension"),
};
pub const WRITE_SPARSE_EXTENSION: BuiltinExtensionDescriptor = BuiltinExtensionDescriptor {
    id: "write-sparse-data",
    mode: BuiltinExtensionMode::RunMatOnly,
    description: "write with sparse numeric data is a RunMat extension",
    error_identifier: Some("RunMat:compatibility:WriteSparseDataExtension"),
};
pub const WRITE_LOGICAL_EXTENSION: BuiltinExtensionDescriptor = BuiltinExtensionDescriptor {
    id: "write-logical-data",
    mode: BuiltinExtensionMode::RunMatOnly,
    description: "write with logical data is a RunMat extension",
    error_identifier: Some("RunMat:compatibility:WriteLogicalDataExtension"),
};
pub const WRITE_EXTENSIONS: [BuiltinExtensionDescriptor; 4] = [
    WRITE_EXPLICIT_GPU_EXTENSION,
    WRITE_NONVECTOR_EXTENSION,
    WRITE_SPARSE_EXTENSION,
    WRITE_LOGICAL_EXTENSION,
];

const WRITE_INTEGER_VECTOR_INPUT: [BuiltinIntegerInputCapability; 1] =
    [BuiltinIntegerInputCapability {
        name: "data",
        classes: &crate::ALL_INTEGER_CLASSES,
        availability: BuiltinIntegerInputAvailability::Documented,
        scalar_double: BuiltinIntegerScalarDoubleRule::NotApplicable,
        notes: "The public datatype table explicitly lists all eight integer classes for real row or column vectors. Without a datatype override, the native input datatype selects the wire representation.",
    }];
const WRITE_INTEGER_NONVECTOR_INPUT: [BuiltinIntegerInputCapability; 1] =
    [BuiltinIntegerInputCapability {
        name: "matrix or multidimensional data",
        classes: &crate::ALL_INTEGER_CLASSES,
        availability: BuiltinIntegerInputAvailability::RunMatOnly,
        scalar_double: BuiltinIntegerScalarDoubleRule::NotApplicable,
        notes: "RunMat can serialize nonvector integer arrays in column-major order, but the public interactive contract documents row or column vectors.",
    }];
const WRITE_INTEGER_SPARSE_INPUT: [BuiltinIntegerInputCapability; 1] =
    [BuiltinIntegerInputCapability {
        name: "sparse data",
        classes: &crate::ALL_INTEGER_CLASSES,
        availability: BuiltinIntegerInputAvailability::RunMatOnly,
        scalar_double: BuiltinIntegerScalarDoubleRule::NotApplicable,
        notes: "Sparse payloads are a RunMat-only host convenience and are densified from authoritative integer CSC storage before byte encoding.",
    }];
const WRITE_INTEGER_GPU_INPUT: [BuiltinIntegerInputCapability; 1] =
    [BuiltinIntegerInputCapability {
        name: "explicit gpuArray data",
        classes: &crate::ALL_INTEGER_CLASSES,
        availability: BuiltinIntegerInputAvailability::RunMatOnly,
        scalar_double: BuiltinIntegerScalarDoubleRule::NotApplicable,
        notes: "The public page lists GPU code generation rather than interactive gpuArray input. RunMat gates explicit device intent before gather while leaving automatic residency transparent.",
    }];
pub const WRITE_INTEGER_CAPABILITIES: [BuiltinIntegerCapabilityDescriptor; 4] = [
    BuiltinIntegerCapabilityDescriptor {
        form: "write(client, integer_vector, datatype?)",
        inputs: &WRITE_INTEGER_VECTOR_INPUT,
        computation_domain: BuiltinIntegerComputationDomain::ExactInteger,
        output_class: BuiltinIntegerOutputClassRule::NotApplicable,
        overflow: BuiltinIntegerOverflowRule::Saturate,
        backend: BuiltinIntegerBackendRule::GatherFallback,
        overload: BuiltinIntegerOverloadKind::Multiple,
        notes: "Same-class writes emit exact native bytes with client byte order. An explicit integer datatype performs one exact saturating cast; single/double overrides are deliberate floating wire boundaries.",
    },
    BuiltinIntegerCapabilityDescriptor {
        form: "write(client, integer_matrix, datatype?)",
        inputs: &WRITE_INTEGER_NONVECTOR_INPUT,
        computation_domain: BuiltinIntegerComputationDomain::ExactInteger,
        output_class: BuiltinIntegerOutputClassRule::NotApplicable,
        overflow: BuiltinIntegerOverflowRule::Saturate,
        backend: BuiltinIntegerBackendRule::GatherFallback,
        overload: BuiltinIntegerOverloadKind::Multiple,
        notes: "RunMat mode traverses nonvector payloads in column-major storage order after the independent shape gate.",
    },
    BuiltinIntegerCapabilityDescriptor {
        form: "write(client, sparse_integer_data, datatype?)",
        inputs: &WRITE_INTEGER_SPARSE_INPUT,
        computation_domain: BuiltinIntegerComputationDomain::ExactInteger,
        output_class: BuiltinIntegerOutputClassRule::NotApplicable,
        overflow: BuiltinIntegerOverflowRule::Saturate,
        backend: BuiltinIntegerBackendRule::HostOnly,
        overload: BuiltinIntegerOverloadKind::Multiple,
        notes: "RunMat mode expands sparse integer storage exactly before the selected wire cast; the sparse extension gate precedes socket access.",
    },
    BuiltinIntegerCapabilityDescriptor {
        form: "write(client, explicit_gpu_integer_data, datatype?)",
        inputs: &WRITE_INTEGER_GPU_INPUT,
        computation_domain: BuiltinIntegerComputationDomain::ExactInteger,
        output_class: BuiltinIntegerOutputClassRule::NotApplicable,
        overflow: BuiltinIntegerOverflowRule::Saturate,
        backend: BuiltinIntegerBackendRule::GatherFallback,
        overload: BuiltinIntegerOverloadKind::Multiple,
        notes: "Explicit gpuArray data is available only in RunMat mode and gathers through its owner before host socket encoding.",
    },
];
