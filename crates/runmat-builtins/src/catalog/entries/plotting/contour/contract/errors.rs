use crate::*;

pub const CONTOUR_ERROR_INVALID_ARGUMENT: BuiltinErrorDescriptor = BuiltinErrorDescriptor {
    code: "RM.CONTOUR.INVALID_ARGUMENT",
    identifier: Some("RunMat:contour:InvalidArgument"),
    when: "Contour input arrays, level arguments, or name/value options are invalid.",
    message: "contour: invalid argument",
};

pub const CONTOUR_ERROR_INTERNAL: BuiltinErrorDescriptor = BuiltinErrorDescriptor {
    code: "RM.CONTOUR.INTERNAL",
    identifier: Some("RunMat:contour:Internal"),
    when: "Internal contour render preparation fails unexpectedly.",
    message: "contour: internal operation failed",
};

pub(super) const CONTOUR_ERRORS: [BuiltinErrorDescriptor; 3] = [
    CONTOUR_ERROR_INVALID_ARGUMENT,
    CONTOUR_ERROR_INTERNAL,
    COORDINATE_ERROR_TOO_MANY_OUTPUTS,
];
