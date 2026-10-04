use crate::*;

pub const NATIVE_FFI_ERROR_UNAVAILABLE: BuiltinErrorDescriptor = BuiltinErrorDescriptor {
    code: "RM.NATIVE_FFI.UNAVAILABLE",
    identifier: Some("RunMat:Foreign:UnsupportedOnWasm"),
    when: "The current host cannot load a native shared library.",
    message: "Native shared libraries are unavailable on this host.",
};
pub const NATIVE_FFI_ERROR_INVALID: BuiltinErrorDescriptor = BuiltinErrorDescriptor {
    code: "RM.NATIVE_FFI.INVALID_CALL",
    identifier: Some("RunMat:Foreign:InvalidCall"),
    when: "Arguments do not satisfy the normalized shared-library contract.",
    message: "Invalid shared-library operation.",
};
pub const NATIVE_FFI_ERROR_HOST_UNAVAILABLE: BuiltinErrorDescriptor = BuiltinErrorDescriptor {
    code: "RM.NATIVE_FFI.HOST_UNAVAILABLE",
    identifier: Some("RunMat:Foreign:HostUnavailable"),
    when: "A native shared-library operation is invoked without an active runtime context.",
    message: "Native shared-library operation has no active runtime context.",
};
const ERRORS: &[BuiltinErrorDescriptor] = &[
    NATIVE_FFI_ERROR_UNAVAILABLE,
    NATIVE_FFI_ERROR_INVALID,
    NATIVE_FFI_ERROR_HOST_UNAVAILABLE,
];
pub const NATIVE_FFI_INTEGER_AUDIT: BuiltinIntegerAuditDescriptor = BuiltinIntegerAuditDescriptor {
    kind: BuiltinIntegerAuditKind::NotApplicable,
    canonical_builtin: None,
    notes: "Typed integer values retain native storage and are range-checked against the declared C ABI type.",
};

const TEXT: BuiltinParamDescriptor = BuiltinParamDescriptor {
    name: "name",
    ty: BuiltinParamType::StringScalar,
    arity: BuiltinParamArity::Required,
    default: None,
    description: "Library, header, alias, or option name.",
};
const ANY: BuiltinParamDescriptor = BuiltinParamDescriptor {
    name: "arguments",
    ty: BuiltinParamType::Any,
    arity: BuiltinParamArity::Variadic,
    default: None,
    description: "Arguments to the native-library operation.",
};
const RESULT: BuiltinParamDescriptor = BuiltinParamDescriptor {
    name: "result",
    ty: BuiltinParamType::Any,
    arity: BuiltinParamArity::Variadic,
    default: None,
    description: "Declared result of the native-library operation.",
};
const LOAD_SIGNATURES: &[BuiltinSignatureDescriptor] = &[BuiltinSignatureDescriptor {
    label: "[notfound, warnings] = loadlibrary(library, header, options)",
    inputs: &[ANY],
    outputs: &[RESULT],
}];
const UNLOAD_SIGNATURES: &[BuiltinSignatureDescriptor] = &[BuiltinSignatureDescriptor {
    label: "unloadlibrary(library)",
    inputs: &[TEXT],
    outputs: &[],
}];
const IS_LOADED_SIGNATURES: &[BuiltinSignatureDescriptor] = &[BuiltinSignatureDescriptor {
    label: "loaded = libisloaded(library)",
    inputs: &[TEXT],
    outputs: &[RESULT],
}];
const FUNCTIONS_SIGNATURES: &[BuiltinSignatureDescriptor] = &[
    BuiltinSignatureDescriptor {
        label: "names = libfunctions(library)",
        inputs: &[TEXT],
        outputs: &[RESULT],
    },
    BuiltinSignatureDescriptor {
        label: "signatures = libfunctions(library, '-full')",
        inputs: &[ANY],
        outputs: &[RESULT],
    },
];
const CALL_SIGNATURES: &[BuiltinSignatureDescriptor] = &[BuiltinSignatureDescriptor {
    label: "[outputs] = calllib(library, function, arguments)",
    inputs: &[ANY],
    outputs: &[RESULT],
}];
const POINTER_SIGNATURES: &[BuiltinSignatureDescriptor] = &[
    BuiltinSignatureDescriptor {
        label: "pointer = libpointer()",
        inputs: &[],
        outputs: &[RESULT],
    },
    BuiltinSignatureDescriptor {
        label: "pointer = libpointer(type)",
        inputs: &[TEXT],
        outputs: &[RESULT],
    },
    BuiltinSignatureDescriptor {
        label: "pointer = libpointer(type, value)",
        inputs: &[ANY],
        outputs: &[RESULT],
    },
];
const STRUCT_SIGNATURES: &[BuiltinSignatureDescriptor] = &[
    BuiltinSignatureDescriptor {
        label: "value = libstruct(type)",
        inputs: &[TEXT],
        outputs: &[RESULT],
    },
    BuiltinSignatureDescriptor {
        label: "value = libstruct(type, structure)",
        inputs: &[ANY],
        outputs: &[RESULT],
    },
];
const SETDATATYPE_SIGNATURES: &[BuiltinSignatureDescriptor] = &[BuiltinSignatureDescriptor {
    label: "setdatatype(pointer, type, dimensions)",
    inputs: &[ANY],
    outputs: &[],
}];

pub const LOADLIBRARY_DESCRIPTOR: BuiltinDescriptor = BuiltinDescriptor {
    signatures: LOAD_SIGNATURES,
    output_mode: BuiltinOutputMode::ByRequestedOutputCount,
    completion_policy: BuiltinCompletionPolicy::Public,
    errors: ERRORS,
};
pub const UNLOADLIBRARY_DESCRIPTOR: BuiltinDescriptor = BuiltinDescriptor {
    signatures: UNLOAD_SIGNATURES,
    output_mode: BuiltinOutputMode::Fixed,
    completion_policy: BuiltinCompletionPolicy::Public,
    errors: ERRORS,
};
pub const LIBISLOADED_DESCRIPTOR: BuiltinDescriptor = BuiltinDescriptor {
    signatures: IS_LOADED_SIGNATURES,
    output_mode: BuiltinOutputMode::Fixed,
    completion_policy: BuiltinCompletionPolicy::Public,
    errors: ERRORS,
};
pub const LIBFUNCTIONS_DESCRIPTOR: BuiltinDescriptor = BuiltinDescriptor {
    signatures: FUNCTIONS_SIGNATURES,
    output_mode: BuiltinOutputMode::Fixed,
    completion_policy: BuiltinCompletionPolicy::Public,
    errors: ERRORS,
};
pub const CALLLIB_DESCRIPTOR: BuiltinDescriptor = BuiltinDescriptor {
    signatures: CALL_SIGNATURES,
    output_mode: BuiltinOutputMode::ByRequestedOutputCount,
    completion_policy: BuiltinCompletionPolicy::Public,
    errors: ERRORS,
};
pub const LIBPOINTER_DESCRIPTOR: BuiltinDescriptor = BuiltinDescriptor {
    signatures: POINTER_SIGNATURES,
    output_mode: BuiltinOutputMode::Fixed,
    completion_policy: BuiltinCompletionPolicy::Public,
    errors: ERRORS,
};
pub const LIBSTRUCT_DESCRIPTOR: BuiltinDescriptor = BuiltinDescriptor {
    signatures: STRUCT_SIGNATURES,
    output_mode: BuiltinOutputMode::Fixed,
    completion_policy: BuiltinCompletionPolicy::Public,
    errors: ERRORS,
};
pub const SETDATATYPE_DESCRIPTOR: BuiltinDescriptor = BuiltinDescriptor {
    signatures: SETDATATYPE_SIGNATURES,
    output_mode: BuiltinOutputMode::Fixed,
    completion_policy: BuiltinCompletionPolicy::Public,
    errors: ERRORS,
};
