use crate::*;

pub const PIE_ERROR_INVALID_ARGUMENT: BuiltinErrorDescriptor = BuiltinErrorDescriptor {
    code: "RM.PIE.INVALID_ARGUMENT",
    identifier: Some("RunMat:pie:InvalidArgument"),
    when: "Pie values, explode vectors, or labels are malformed or incompatible.",
    message: "pie: invalid argument",
};

pub(super) const PIE_ERROR_INTERNAL: BuiltinErrorDescriptor = BuiltinErrorDescriptor {
    code: "RM.PIE.INTERNAL",
    identifier: Some("RunMat:pie:Internal"),
    when: "Internal render preparation fails.",
    message: "pie: internal operation failed",
};

pub(super) const PIE_ERRORS: [BuiltinErrorDescriptor; 3] = [
    PIE_ERROR_INVALID_ARGUMENT,
    PIE_ERROR_INTERNAL,
    crate::COORDINATE_ERROR_TOO_MANY_OUTPUTS,
];
