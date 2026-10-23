use crate::*;

pub const POLARHISTOGRAM_ERROR_INVALID_ARGUMENT: BuiltinErrorDescriptor = BuiltinErrorDescriptor {
    code: "RM.POLARHISTOGRAM.INVALID_ARGUMENT",
    identifier: Some("RunMat:polarhistogram:InvalidArgument"),
    when: "Input data, bin specification, axes targeting, or style arguments are invalid.",
    message: "polarhistogram: invalid argument",
};

pub const POLARHISTOGRAM_ERROR_INTERNAL: BuiltinErrorDescriptor = BuiltinErrorDescriptor {
    code: "RM.POLARHISTOGRAM.INTERNAL",
    identifier: Some("RunMat:polarhistogram:Internal"),
    when: "Internal polar histogram construction or rendering fails.",
    message: "polarhistogram: internal operation failed",
};

pub(super) const POLARHISTOGRAM_ERRORS: [BuiltinErrorDescriptor; 3] = [
    POLARHISTOGRAM_ERROR_INVALID_ARGUMENT,
    POLARHISTOGRAM_ERROR_INTERNAL,
    crate::COORDINATE_ERROR_TOO_MANY_OUTPUTS,
];
