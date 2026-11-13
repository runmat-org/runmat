use crate::*;

pub const CONTOUR3_ERROR_INVALID_ARGUMENT: BuiltinErrorDescriptor = BuiltinErrorDescriptor {
    code: "RM.CONTOUR3.INVALID_ARGUMENT",
    identifier: Some("RunMat:contour3:InvalidArgument"),
    when: "Contour input arrays, level arguments, or name/value options are invalid.",
    message: "contour3: invalid argument",
};

pub const CONTOUR3_ERROR_INTERNAL: BuiltinErrorDescriptor = BuiltinErrorDescriptor {
    code: "RM.CONTOUR3.INTERNAL",
    identifier: Some("RunMat:contour3:Internal"),
    when: "Internal contour render preparation fails unexpectedly.",
    message: "contour3: internal operation failed",
};

pub(super) const CONTOUR3_ERRORS: [BuiltinErrorDescriptor; 3] = [
    CONTOUR3_ERROR_INVALID_ARGUMENT,
    CONTOUR3_ERROR_INTERNAL,
    COORDINATE_ERROR_TOO_MANY_OUTPUTS,
];
