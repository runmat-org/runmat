use crate::BuiltinErrorDescriptor;

pub const ACCUMARRAY_ERROR_INVALID_INPUT: BuiltinErrorDescriptor = BuiltinErrorDescriptor { code: "RM.ACCUMARRAY.INVALID_INPUT", identifier: Some("RunMat:accumarray:InvalidInput"), when: "Indices, data, size, callback, fill, or sparse controls do not satisfy the accumarray contract.", message: "accumarray: invalid input" };
pub const ACCUMARRAY_ERROR_CALLBACK: BuiltinErrorDescriptor = BuiltinErrorDescriptor {
    code: "RM.ACCUMARRAY.CALLBACK",
    identifier: Some("RunMat:accumarray:CallbackFailed"),
    when: "The group function fails or does not return a supported scalar.",
    message: "accumarray: callback failed",
};
pub const ACCUMARRAY_ERROR_TOO_LARGE: BuiltinErrorDescriptor = BuiltinErrorDescriptor {
    code: "RM.ACCUMARRAY.TOO_LARGE",
    identifier: Some("RunMat:accumarray:TooLarge"),
    when: "The requested shape or linear index overflows or exceeds the materialization limit.",
    message: "accumarray: output is too large",
};

pub(super) const ERRORS: &[BuiltinErrorDescriptor] = &[
    ACCUMARRAY_ERROR_INVALID_INPUT,
    ACCUMARRAY_ERROR_CALLBACK,
    ACCUMARRAY_ERROR_TOO_LARGE,
];
