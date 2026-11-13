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
pub(super) const ERRORS: &[BuiltinErrorDescriptor] = &[
    NATIVE_FFI_ERROR_UNAVAILABLE,
    NATIVE_FFI_ERROR_INVALID,
    NATIVE_FFI_ERROR_HOST_UNAVAILABLE,
];
// The public text conversion also accepts scalar numeric and logical values.
// `Any` is the truthful descriptor type until the contract model has a union.
pub(super) const fn scalar_text(
    name: &'static str,
    description: &'static str,
) -> BuiltinParamDescriptor {
    BuiltinParamDescriptor {
        name,
        ty: BuiltinParamType::Any,
        arity: BuiltinParamArity::Required,
        default: None,
        description,
    }
}
