use crate::*;

pub const PLOT3_ERROR_INVALID_ARGUMENT: BuiltinErrorDescriptor = BuiltinErrorDescriptor {
    code: "RM.PLOT3.INVALID_ARGUMENT",
    identifier: Some("RunMat:plot3:InvalidArgument"),
    when: "Input data, series grammar, axes targeting, or style arguments are invalid.",
    message: "plot3: invalid argument",
};

pub const PLOT3_ERROR_INTERNAL: BuiltinErrorDescriptor = BuiltinErrorDescriptor {
    code: "RM.PLOT3.INTERNAL",
    identifier: Some("RunMat:plot3:Internal"),
    when: "Internal 3-D line construction or rendering fails unexpectedly.",
    message: "plot3: internal operation failed",
};

pub(super) const PLOT3_ERRORS: [BuiltinErrorDescriptor; 2] =
    [PLOT3_ERROR_INVALID_ARGUMENT, PLOT3_ERROR_INTERNAL];
