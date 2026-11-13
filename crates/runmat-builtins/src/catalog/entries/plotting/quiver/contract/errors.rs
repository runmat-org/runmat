use crate::*;

pub const QUIVER_ERROR_INVALID_ARGUMENT: BuiltinErrorDescriptor = BuiltinErrorDescriptor {
    code: "RM.QUIVER.INVALID_ARGUMENT",
    identifier: Some("RunMat:quiver:InvalidArgument"),
    when: "Input data, axes targeting, or quiver style arguments are invalid.",
    message: "quiver: invalid argument",
};

pub const QUIVER_ERROR_INTERNAL: BuiltinErrorDescriptor = BuiltinErrorDescriptor {
    code: "RM.QUIVER.INTERNAL",
    identifier: Some("RunMat:quiver:Internal"),
    when: "Internal quiver construction or rendering fails unexpectedly.",
    message: "quiver: internal operation failed",
};

pub(super) const QUIVER_ERRORS: [BuiltinErrorDescriptor; 3] = [
    QUIVER_ERROR_INVALID_ARGUMENT,
    QUIVER_ERROR_INTERNAL,
    COORDINATE_ERROR_TOO_MANY_OUTPUTS,
];
