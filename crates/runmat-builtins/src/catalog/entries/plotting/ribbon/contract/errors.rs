use crate::*;

pub const RIBBON_ERROR_INVALID_ARGUMENT: BuiltinErrorDescriptor = BuiltinErrorDescriptor {
    code: "RM.RIBBON.INVALID_ARGUMENT",
    identifier: Some("RunMat:ribbon:InvalidArgument"),
    when: "Ribbon data, x coordinates, width, axes target, or style arguments are invalid.",
    message: "ribbon: invalid argument",
};

pub const RIBBON_ERROR_INTERNAL: BuiltinErrorDescriptor = BuiltinErrorDescriptor {
    code: "RM.RIBBON.INTERNAL",
    identifier: Some("RunMat:ribbon:Internal"),
    when: "Internal ribbon surface construction or rendering fails.",
    message: "ribbon: internal operation failed",
};

pub(super) const RIBBON_ERRORS: [BuiltinErrorDescriptor; 3] = [
    RIBBON_ERROR_INVALID_ARGUMENT,
    RIBBON_ERROR_INTERNAL,
    crate::COORDINATE_ERROR_TOO_MANY_OUTPUTS,
];
