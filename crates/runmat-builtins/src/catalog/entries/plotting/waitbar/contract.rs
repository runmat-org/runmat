use crate::*;

pub const WAITBAR_INTEGER_PROGRESS_EXTENSION: BuiltinExtensionDescriptor =
    BuiltinExtensionDescriptor {
        id: "waitbar-integer-progress",
        mode: BuiltinExtensionMode::RunMatOnly,
        description: "waitbar with typed-integer progress storage is a RunMat extension",
        error_identifier: Some("RunMat:compatibility:WaitbarIntegerProgressExtension"),
    };
pub const WAITBAR_INTEGER_HANDLE_EXTENSION: BuiltinExtensionDescriptor =
    BuiltinExtensionDescriptor {
        id: "waitbar-integer-handle",
        mode: BuiltinExtensionMode::RunMatOnly,
        description: "waitbar with a typed-integer numeric figure alias is a RunMat extension",
        error_identifier: Some("RunMat:compatibility:WaitbarIntegerHandleExtension"),
    };
pub const WAITBAR_EXTENSIONS: [BuiltinExtensionDescriptor; 2] = [
    WAITBAR_INTEGER_PROGRESS_EXTENSION,
    WAITBAR_INTEGER_HANDLE_EXTENSION,
];
const INTEGER_PROGRESS_INPUTS: [BuiltinIntegerInputCapability; 1] = [BuiltinIntegerInputCapability {
    name: "x", classes: &ALL_INTEGER_CLASSES,
    availability: BuiltinIntegerInputAvailability::RunMatOnly,
    scalar_double: BuiltinIntegerScalarDoubleRule::Allowed,
    notes: "The public page specifies a real progress number but does not enumerate native integer storage; RunMat gates typed integer progress independently.",
}];
const INTEGER_HANDLE_INPUTS: [BuiltinIntegerInputCapability; 1] = [BuiltinIntegerInputCapability {
    name: "f", classes: &ALL_INTEGER_CLASSES,
    availability: BuiltinIntegerInputAvailability::RunMatOnly,
    scalar_double: BuiltinIntegerScalarDoubleRule::Allowed,
    notes: "The documented argument is a Figure object; typed-integer numeric aliases are a gated RunMat representation extension.",
}];
pub const WAITBAR_INTEGER_CAPABILITIES: [BuiltinIntegerCapabilityDescriptor; 2] = [
    BuiltinIntegerCapabilityDescriptor {
        form: "h = waitbar(integer_x, ...)", inputs: &INTEGER_PROGRESS_INPUTS,
        computation_domain: BuiltinIntegerComputationDomain::FloatingPoint,
        output_class: BuiltinIntegerOutputClassRule::Double,
        overflow: BuiltinIntegerOverflowRule::Error,
        backend: BuiltinIntegerBackendRule::HostOnly,
        overload: BuiltinIntegerOverloadKind::ScalarOnly,
        notes: "The exact scalar crosses one checked binary64 UI-metadata boundary and is clamped to the implemented progress interval before figure mutation.",
    },
    BuiltinIntegerCapabilityDescriptor {
        form: "h = waitbar(x, integer_f, ...)", inputs: &INTEGER_HANDLE_INPUTS,
        computation_domain: BuiltinIntegerComputationDomain::Structural,
        output_class: BuiltinIntegerOutputClassRule::Double,
        overflow: BuiltinIntegerOverflowRule::NotApplicable,
        backend: BuiltinIntegerBackendRule::HostOnly,
        overload: BuiltinIntegerOverloadKind::StructuralParameter,
        notes: "The compatibility gate precedes numeric handle decoding and any waitbar-state access; resident handles are not gathered.",
    },
];
const OUTPUT_HANDLE: [BuiltinParamDescriptor; 1] = [BuiltinParamDescriptor {
    name: "h",
    ty: BuiltinParamType::NumericScalar,
    arity: BuiltinParamArity::Required,
    default: None,
    description: "Waitbar figure handle.",
}];
const PROGRESS: BuiltinParamDescriptor = BuiltinParamDescriptor {
    name: "x",
    ty: BuiltinParamType::NumericScalar,
    arity: BuiltinParamArity::Required,
    default: None,
    description: "Progress fraction. Finite values are clamped into [0, 1].",
};
const MESSAGE: BuiltinParamDescriptor = BuiltinParamDescriptor {
    name: "msg",
    ty: BuiltinParamType::Any,
    arity: BuiltinParamArity::Required,
    default: None,
    description: "Progress message as text scalar, string array, or cell array of text.",
};
const EXISTING_HANDLE: BuiltinParamDescriptor = BuiltinParamDescriptor {
    name: "h",
    ty: BuiltinParamType::NumericScalar,
    arity: BuiltinParamArity::Required,
    default: None,
    description: "Existing waitbar figure handle to update.",
};
const PROPERTY_PAIRS: BuiltinParamDescriptor = BuiltinParamDescriptor {
    name: "Name, Value",
    ty: BuiltinParamType::Any,
    arity: BuiltinParamArity::Variadic,
    default: None,
    description: "Optional waitbar or figure-style property/value pairs.",
};
const INPUTS_PROGRESS: [BuiltinParamDescriptor; 1] = [PROGRESS];
const INPUTS_PROGRESS_MESSAGE: [BuiltinParamDescriptor; 2] = [PROGRESS, MESSAGE];
const INPUTS_PROGRESS_HANDLE: [BuiltinParamDescriptor; 2] = [PROGRESS, EXISTING_HANDLE];
const INPUTS_PROGRESS_HANDLE_MESSAGE: [BuiltinParamDescriptor; 3] =
    [PROGRESS, EXISTING_HANDLE, MESSAGE];
const INPUTS_PROGRESS_HANDLE_MESSAGE_PAIRS: [BuiltinParamDescriptor; 4] =
    [PROGRESS, EXISTING_HANDLE, MESSAGE, PROPERTY_PAIRS];
const SIGNATURES: [BuiltinSignatureDescriptor; 9] = [
    BuiltinSignatureDescriptor {
        label: "waitbar(x)",
        inputs: &INPUTS_PROGRESS,
        outputs: &[],
    },
    BuiltinSignatureDescriptor {
        label: "waitbar(x, msg)",
        inputs: &INPUTS_PROGRESS_MESSAGE,
        outputs: &[],
    },
    BuiltinSignatureDescriptor {
        label: "waitbar(x, h, msg, Name, Value, ...)",
        inputs: &INPUTS_PROGRESS_HANDLE_MESSAGE_PAIRS,
        outputs: &[],
    },
    BuiltinSignatureDescriptor {
        label: "h = waitbar(x)",
        inputs: &INPUTS_PROGRESS,
        outputs: &OUTPUT_HANDLE,
    },
    BuiltinSignatureDescriptor {
        label: "h = waitbar(x, msg)",
        inputs: &INPUTS_PROGRESS_MESSAGE,
        outputs: &OUTPUT_HANDLE,
    },
    BuiltinSignatureDescriptor {
        label: "h = waitbar(x, h)",
        inputs: &INPUTS_PROGRESS_HANDLE,
        outputs: &OUTPUT_HANDLE,
    },
    BuiltinSignatureDescriptor {
        label: "h = waitbar(x, h, msg)",
        inputs: &INPUTS_PROGRESS_HANDLE_MESSAGE,
        outputs: &OUTPUT_HANDLE,
    },
    BuiltinSignatureDescriptor {
        label: "h = waitbar(x, h, msg, Name, Value, ...)",
        inputs: &INPUTS_PROGRESS_HANDLE_MESSAGE_PAIRS,
        outputs: &OUTPUT_HANDLE,
    },
    BuiltinSignatureDescriptor {
        label: "waitbar(x, h, msg)",
        inputs: &INPUTS_PROGRESS_HANDLE_MESSAGE,
        outputs: &[],
    },
];
pub const WAITBAR_ERROR_INVALID_ARGUMENT: BuiltinErrorDescriptor = BuiltinErrorDescriptor {
    code: "RM.WAITBAR.INVALID_ARGUMENT",
    identifier: Some("RunMat:waitbar:InvalidArgument"),
    when: "Progress, handle, message, or property/value arguments are invalid.",
    message: "waitbar: invalid argument",
};
const ERRORS: [BuiltinErrorDescriptor; 1] = [WAITBAR_ERROR_INVALID_ARGUMENT];
pub const WAITBAR_DESCRIPTOR: BuiltinDescriptor = BuiltinDescriptor {
    signatures: &SIGNATURES,
    output_mode: BuiltinOutputMode::Fixed,
    completion_policy: BuiltinCompletionPolicy::Public,
    errors: &ERRORS,
};
