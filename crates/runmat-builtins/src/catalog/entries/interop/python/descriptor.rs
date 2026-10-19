use crate::*;

pub const PYTHON_ERROR_UNAVAILABLE: BuiltinErrorDescriptor = BuiltinErrorDescriptor {
    code: "RM.PYTHON.UNAVAILABLE",
    identifier: Some("RunMat:Foreign:UnsupportedOnWasm"),
    when: "The current host has no explicitly provided CPython capability.",
    message: "Python interoperability is unavailable on this host.",
};

pub const PYTHON_ERROR_INVALID: BuiltinErrorDescriptor = BuiltinErrorDescriptor {
    code: "RM.PYTHON.INVALID_CALL",
    identifier: Some("RunMat:Foreign:InvalidCall"),
    when: "Arguments do not satisfy the Python interoperability contract.",
    message: "Invalid Python interoperability operation.",
};

pub const PYTHON_ERROR_HOST_UNAVAILABLE: BuiltinErrorDescriptor = BuiltinErrorDescriptor {
    code: "RM.PYTHON.HOST_UNAVAILABLE",
    identifier: Some("RunMat:Foreign:HostUnavailable"),
    when: "A Python operation is invoked without an active runtime context.",
    message: "Python operation has no active runtime context.",
};

pub(super) const PYTHON_ERROR_ADAPTER_UNAVAILABLE: BuiltinErrorDescriptor =
    BuiltinErrorDescriptor {
        code: "RM.PYTHON.ADAPTER_UNAVAILABLE",
        identifier: Some("RunMat:Foreign:AdapterUnavailable"),
        when: "The session has no Python adapter matching the declared foreign contract.",
        message: "Python adapter is unavailable.",
    };
pub(super) const PYTHON_ERROR_CAPABILITY_UNAVAILABLE: BuiltinErrorDescriptor =
    BuiltinErrorDescriptor {
        code: "RM.PYTHON.CAPABILITY_UNAVAILABLE",
        identifier: Some("RunMat:Foreign:CapabilityUnavailable"),
        when: "The installed foreign host cannot satisfy a required Python capability.",
        message: "Required Python capability is unavailable.",
    };
pub(super) const PYTHON_ERROR_EXCEPTION: BuiltinErrorDescriptor = BuiltinErrorDescriptor {
    code: "RM.PYTHON.EXCEPTION",
    identifier: None,
    when: "Python raises an exception during discovery, conversion, callback, or execution.",
    message: "The runtime retains RunMat:Python:<exception type>, source exception, and traceback.",
};
pub(super) const PYTHON_ERROR_MANIFEST: BuiltinErrorDescriptor = BuiltinErrorDescriptor {
    code: "RM.PYTHON.INVALID_MANIFEST",
    identifier: Some("RunMat:Foreign:InvalidManifest"),
    when: "A declared Python artifact does not match its frozen manifest.",
    message: "Python artifact manifest is invalid.",
};

pub(super) const FOREIGN_ERRORS: &[BuiltinErrorDescriptor] = &[
    PYTHON_ERROR_UNAVAILABLE,
    PYTHON_ERROR_INVALID,
    PYTHON_ERROR_HOST_UNAVAILABLE,
    PYTHON_ERROR_ADAPTER_UNAVAILABLE,
    PYTHON_ERROR_CAPABILITY_UNAVAILABLE,
    PYTHON_ERROR_EXCEPTION,
    PYTHON_ERROR_MANIFEST,
];

pub(super) const LOCAL_ERRORS: &[BuiltinErrorDescriptor] = &[PYTHON_ERROR_INVALID];

pub(super) const ENVIRONMENT: &[BuiltinParamDescriptor] = &[BuiltinParamDescriptor {
    name: "environment",
    ty: BuiltinParamType::Any,
    arity: BuiltinParamArity::Required,
    default: None,
    description: "py.PythonEnvironment status object.",
}];

pub(super) const RESULT: &[BuiltinParamDescriptor] = &[BuiltinParamDescriptor {
    name: "result",
    ty: BuiltinParamType::Any,
    arity: BuiltinParamArity::Required,
    default: None,
    description: "Python result converted through the session foreign boundary.",
}];

pub(super) const NONE: &[BuiltinParamDescriptor] = &[];
