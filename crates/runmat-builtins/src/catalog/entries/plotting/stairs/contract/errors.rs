use crate::*;

pub const STAIRS_ERROR_INVALID_ARGUMENT: BuiltinErrorDescriptor = BuiltinErrorDescriptor {
    code: "RM.STAIRS.INVALID_ARGUMENT",
    identifier: Some("RunMat:stairs:InvalidArgument"),
    when: "Input data, axes targeting, or style arguments are invalid.",
    message: "stairs: invalid argument",
};

pub const STAIRS_ERROR_INTERNAL: BuiltinErrorDescriptor = BuiltinErrorDescriptor {
    code: "RM.STAIRS.INTERNAL",
    identifier: Some("RunMat:stairs:Internal"),
    when: "Internal plot construction or rendering fails unexpectedly.",
    message: "stairs: internal operation failed",
};

pub(super) const STAIRS_ERRORS: [BuiltinErrorDescriptor; 3] = [
    STAIRS_ERROR_INVALID_ARGUMENT,
    STAIRS_ERROR_INTERNAL,
    crate::COORDINATE_ERROR_TOO_MANY_OUTPUTS,
];
