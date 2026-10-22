use crate::*;

pub const PLOTMATRIX_ERR_INVALID_ARGUMENT: BuiltinErrorDescriptor = BuiltinErrorDescriptor {
    code: "RM.PLOTMATRIX.INVALID_ARGUMENT",
    identifier: Some("RunMat:plotmatrix:InvalidArgument"),
    when: "Inputs are nonnumeric, observation counts differ, or style arguments are malformed.",
    message: "plotmatrix: invalid argument",
};

pub const PLOTMATRIX_ERR_INTERNAL: BuiltinErrorDescriptor = BuiltinErrorDescriptor {
    code: "RM.PLOTMATRIX.INTERNAL",
    identifier: Some("RunMat:plotmatrix:Internal"),
    when: "Internal subplot, scatter, or histogram rendering fails.",
    message: "plotmatrix: internal operation failed",
};

pub(super) const ERRORS: [BuiltinErrorDescriptor; 3] = [
    PLOTMATRIX_ERR_INVALID_ARGUMENT,
    PLOTMATRIX_ERR_INTERNAL,
    PLOTMATRIX_ERROR_TOO_MANY_OUTPUTS,
];

pub const PLOTMATRIX_ERROR_TOO_MANY_OUTPUTS: BuiltinErrorDescriptor = BuiltinErrorDescriptor {
    code: "RM.PLOTMATRIX.TOO_MANY_OUTPUTS",
    identifier: Some("RunMat:plotting:TooManyOutputs"),
    when: "More than 5 output arguments are requested.",
    message: "plotmatrix: too many output arguments",
};
