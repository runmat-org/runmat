use crate::*;

pub const MESHC_ERROR_INVALID_ARGUMENT: BuiltinErrorDescriptor = BuiltinErrorDescriptor {
    code: "RM.MESHC.INVALID_ARGUMENT",
    identifier: Some("RunMat:meshc:InvalidArgument"),
    when: "Surface/grid/axis inputs or style name/value arguments are invalid.",
    message: "meshc: invalid argument",
};

pub const MESHC_ERROR_INTERNAL: BuiltinErrorDescriptor = BuiltinErrorDescriptor {
    code: "RM.MESHC.INTERNAL",
    identifier: Some("RunMat:meshc:Internal"),
    when: "Internal surface/contour construction or render preparation fails unexpectedly.",
    message: "meshc: internal operation failed",
};

pub(super) const MESHC_ERRORS: [BuiltinErrorDescriptor; 3] = [
    MESHC_ERROR_INVALID_ARGUMENT,
    MESHC_ERROR_INTERNAL,
    COORDINATE_ERROR_TOO_MANY_OUTPUTS,
];
