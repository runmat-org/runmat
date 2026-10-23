use crate::*;

pub const SURFC_ERROR_INVALID_ARGUMENT: BuiltinErrorDescriptor = BuiltinErrorDescriptor {
    code: "RM.SURFC.INVALID_ARGUMENT",
    identifier: Some("RunMat:surfc:InvalidArgument"),
    when: "Surface/grid/axis inputs or style name/value arguments are invalid.",
    message: "surfc: invalid argument",
};

pub const SURFC_ERROR_INTERNAL: BuiltinErrorDescriptor = BuiltinErrorDescriptor {
    code: "RM.SURFC.INTERNAL",
    identifier: Some("RunMat:surfc:Internal"),
    when: "Internal surface/contour construction or render preparation fails unexpectedly.",
    message: "surfc: internal operation failed",
};

pub(super) const SURFC_ERRORS: [BuiltinErrorDescriptor; 3] = [
    SURFC_ERROR_INVALID_ARGUMENT,
    SURFC_ERROR_INTERNAL,
    COORDINATE_ERROR_TOO_MANY_OUTPUTS,
];
