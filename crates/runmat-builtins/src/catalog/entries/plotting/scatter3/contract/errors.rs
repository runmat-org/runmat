use crate::*;

pub const SCATTER3_ERROR_INVALID_ARGUMENT: BuiltinErrorDescriptor = BuiltinErrorDescriptor {
    code: "RM.SCATTER3.INVALID_ARGUMENT",
    identifier: Some("RunMat:scatter3:InvalidArgument"),
    when: "Input data, axes targeting, or marker style arguments are invalid.",
    message: "scatter3: invalid argument",
};

pub const SCATTER3_ERROR_INTERNAL: BuiltinErrorDescriptor = BuiltinErrorDescriptor {
    code: "RM.SCATTER3.INTERNAL",
    identifier: Some("RunMat:scatter3:Internal"),
    when: "Internal 3-D scatter construction or rendering fails unexpectedly.",
    message: "scatter3: internal operation failed",
};

pub(super) const SCATTER3_ERRORS: [BuiltinErrorDescriptor; 2] =
    [SCATTER3_ERROR_INVALID_ARGUMENT, SCATTER3_ERROR_INTERNAL];
