use crate::*;

pub const SCATTER_ERROR_INVALID_ARGUMENT: BuiltinErrorDescriptor = BuiltinErrorDescriptor {
    code: "RM.SCATTER.INVALID_ARGUMENT",
    identifier: Some("RunMat:scatter:InvalidArgument"),
    when: "Input data, axes targeting, or marker style arguments are invalid.",
    message: "scatter: invalid argument",
};

pub const SCATTER_ERROR_INTERNAL: BuiltinErrorDescriptor = BuiltinErrorDescriptor {
    code: "RM.SCATTER.INTERNAL",
    identifier: Some("RunMat:scatter:Internal"),
    when: "Internal scatter construction or rendering fails unexpectedly.",
    message: "scatter: internal operation failed",
};

pub(super) const SCATTER_ERRORS: [BuiltinErrorDescriptor; 3] = [
    SCATTER_ERROR_INVALID_ARGUMENT,
    SCATTER_ERROR_INTERNAL,
    crate::COORDINATE_ERROR_TOO_MANY_OUTPUTS,
];
