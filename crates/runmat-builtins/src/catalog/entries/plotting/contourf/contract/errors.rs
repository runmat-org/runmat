use crate::*;

pub const CONTOURF_ERROR_INVALID_ARGUMENT: BuiltinErrorDescriptor = BuiltinErrorDescriptor {
    code: "RM.CONTOURF.INVALID_ARGUMENT",
    identifier: Some("RunMat:contourf:InvalidArgument"),
    when: "Contour input arrays, level arguments, or name/value options are invalid.",
    message: "contourf: invalid argument",
};

pub const CONTOURF_ERROR_INTERNAL: BuiltinErrorDescriptor = BuiltinErrorDescriptor {
    code: "RM.CONTOURF.INTERNAL",
    identifier: Some("RunMat:contourf:Internal"),
    when: "Internal filled-contour render preparation fails unexpectedly.",
    message: "contourf: internal operation failed",
};

pub(super) const CONTOURF_ERRORS: [BuiltinErrorDescriptor; 3] = [
    CONTOURF_ERROR_INVALID_ARGUMENT,
    CONTOURF_ERROR_INTERNAL,
    COORDINATE_ERROR_TOO_MANY_OUTPUTS,
];
