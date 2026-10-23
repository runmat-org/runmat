use crate::*;

pub const QUIVER3_ERROR_INVALID_ARGUMENT: BuiltinErrorDescriptor = BuiltinErrorDescriptor {
    code: "RM.QUIVER3.INVALID_ARGUMENT",
    identifier: Some("RunMat:quiver3:InvalidArgument"),
    when: "Input data, axes targeting, scale, or style arguments are invalid.",
    message: "quiver3: invalid argument",
};

pub const QUIVER3_ERROR_INTERNAL: BuiltinErrorDescriptor = BuiltinErrorDescriptor {
    code: "RM.QUIVER3.INTERNAL",
    identifier: Some("RunMat:quiver3:Internal"),
    when: "Internal quiver3 construction or rendering fails unexpectedly.",
    message: "quiver3: internal operation failed",
};

pub(super) const QUIVER3_ERRORS: [BuiltinErrorDescriptor; 3] = [
    QUIVER3_ERROR_INVALID_ARGUMENT,
    QUIVER3_ERROR_INTERNAL,
    COORDINATE_ERROR_TOO_MANY_OUTPUTS,
];
