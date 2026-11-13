use crate::*;

pub const GPU_DEVICE_LOGICAL_SELECTOR_EXTENSION: BuiltinExtensionDescriptor =
    BuiltinExtensionDescriptor {
        id: "gpudevice-logical-selector",
        mode: BuiltinExtensionMode::RunMatOnly,
        description: "using a logical scalar as a gpuDevice index is a RunMat extension",
        error_identifier: Some("RunMat:compatibility:GpuDeviceLogicalSelectorExtension"),
    };

pub const GPU_DEVICE_EXTENSIONS: [BuiltinExtensionDescriptor; 1] =
    [GPU_DEVICE_LOGICAL_SELECTOR_EXTENSION];

const DOCUMENTED_INTEGER_INDEX: [BuiltinIntegerInputCapability; 1] =
    [BuiltinIntegerInputCapability {
        name: "ind",
        classes: &ALL_INTEGER_CLASSES,
        availability: BuiltinIntegerInputAvailability::Documented,
        scalar_double: BuiltinIntegerScalarDoubleRule::Allowed,
        notes: "Exact positive host signed/unsigned integer selector; no conversion through double. Physical device availability and lifecycle capability are checked at runtime.",
    }];

pub const GPU_DEVICE_INTEGER_CAPABILITIES: [BuiltinIntegerCapabilityDescriptor; 1] = [
    BuiltinIntegerCapabilityDescriptor {
        form: "D = gpuDevice(integer_ind)",
        inputs: &DOCUMENTED_INTEGER_INDEX,
        computation_domain: BuiltinIntegerComputationDomain::Structural,
        output_class: BuiltinIntegerOutputClassRule::NotApplicable,
        overflow: BuiltinIntegerOverflowRule::Error,
        backend: BuiltinIntegerBackendRule::HostOnly,
        overload: BuiltinIntegerOverloadKind::ScalarOnly,
        notes: "The selection form returns an issued GPUDevice handle after physical same-session reset, including when already selected. Unsupported backends/leases fail explicitly before admission; later retirement failure remains fenced, not rolled back.",
    },
];

const OUTPUT: [BuiltinParamDescriptor; 1] = [BuiltinParamDescriptor {
    name: "device",
    ty: BuiltinParamType::Any,
    arity: BuiltinParamArity::Required,
    default: None,
    description:
        "Session-owned parallel.gpu.GPUDevice handle, or empty host double result for deselection.",
}];
const EMPTY_INPUTS: [BuiltinParamDescriptor; 0] = [];
const INDEX_INPUTS: [BuiltinParamDescriptor; 1] = [BuiltinParamDescriptor {
    name: "arg",
    ty: BuiltinParamType::Any,
    arity: BuiltinParamArity::Required,
    default: None,
    description:
        "Exact positive host device index or empty host array for physical reset and deselection.",
}];
const SIGNATURES: [BuiltinSignatureDescriptor; 2] = [
    BuiltinSignatureDescriptor {
        label: "device = gpuDevice()",
        inputs: &EMPTY_INPUTS,
        outputs: &OUTPUT,
    },
    BuiltinSignatureDescriptor {
        label: "device = gpuDevice(arg)",
        inputs: &INDEX_INPUTS,
        outputs: &OUTPUT,
    },
];

pub const GPU_DEVICE_ERROR_NO_PROVIDER: BuiltinErrorDescriptor = BuiltinErrorDescriptor {
    code: "RM.GPU_DEVICE.NO_PROVIDER",
    identifier: Some("RunMat:gpuDevice:NoProvider"),
    when: "No acceleration provider is registered.",
    message: "gpuDevice: no acceleration provider registered",
};
pub const GPU_DEVICE_ERROR_TOO_MANY_INPUTS: BuiltinErrorDescriptor = BuiltinErrorDescriptor {
    code: "RM.GPU_DEVICE.TOO_MANY_INPUTS",
    identifier: Some("RunMat:gpuDevice:TooManyInputs"),
    when: "More than one input argument was supplied.",
    message: "gpuDevice: too many input arguments",
};
pub const GPU_DEVICE_ERROR_UNSUPPORTED_ARGUMENT: BuiltinErrorDescriptor = BuiltinErrorDescriptor {
    code: "RM.GPU_DEVICE.UNSUPPORTED_ARGUMENT",
    identifier: Some("RunMat:gpuDevice:UnsupportedArgument"),
    when: "Argument does not map to a valid device selector/reset operation.",
    message: "gpuDevice: unsupported input argument",
};
pub const GPU_DEVICE_ERROR_RESET_NOT_SUPPORTED: BuiltinErrorDescriptor = BuiltinErrorDescriptor {
    code: "RM.GPU_DEVICE.RESET_NOT_SUPPORTED",
    identifier: Some("RunMat:gpuDevice:ResetNotSupported"),
    when: "Reset was requested but active provider does not support reset.",
    message: "gpuDevice: reset is not supported by the active provider",
};
pub const GPU_DEVICE_ERROR_INVALID_INDEX: BuiltinErrorDescriptor = BuiltinErrorDescriptor {
    code: "RM.GPU_DEVICE.INVALID_INDEX",
    identifier: Some("RunMat:gpuDevice:InvalidIndex"),
    when: "Device index is not a positive integer scalar.",
    message: "gpuDevice: device index must be a positive integer",
};
pub const GPU_DEVICE_ERROR_DEVICE_NOT_AVAILABLE: BuiltinErrorDescriptor = BuiltinErrorDescriptor {
    code: "RM.GPU_DEVICE.DEVICE_NOT_AVAILABLE",
    identifier: Some("RunMat:gpuDevice:DeviceNotAvailable"),
    when: "Requested device index does not match the currently active provider index.",
    message: "gpuDevice: requested device index is not available",
};
const ERRORS: [BuiltinErrorDescriptor; 7] = [
    GPU_DEVICE_ERROR_NO_PROVIDER,
    GPU_DEVICE_ERROR_TOO_MANY_INPUTS,
    GPU_DEVICE_ERROR_UNSUPPORTED_ARGUMENT,
    GPU_DEVICE_ERROR_RESET_NOT_SUPPORTED,
    GPU_DEVICE_ERROR_INVALID_INDEX,
    GPU_DEVICE_ERROR_DEVICE_NOT_AVAILABLE,
    super::methods::LIFECYCLE_ERROR,
];
pub const GPU_DEVICE_DESCRIPTOR: BuiltinDescriptor = BuiltinDescriptor {
    signatures: &SIGNATURES,
    output_mode: BuiltinOutputMode::Fixed,
    completion_policy: BuiltinCompletionPolicy::Public,
    errors: &ERRORS,
};
