use crate::*;

pub const PLOTYY_ERROR_INVALID_ARGUMENT: BuiltinErrorDescriptor = BuiltinErrorDescriptor {
    code: "RM.PLOTYY.INVALID_ARGUMENT",
    identifier: Some("RunMat:plotyy:InvalidArgument"),
    when: "Input count, plotting function selector, or line data is invalid.",
    message: "plotyy: invalid argument",
};

pub const PLOTYY_ERROR_INTERNAL: BuiltinErrorDescriptor = BuiltinErrorDescriptor {
    code: "RM.PLOTYY.INTERNAL",
    identifier: Some("RunMat:plotyy:Internal"),
    when: "Internal plotting state update fails.",
    message: "plotyy: internal operation failed",
};

pub(super) const ERRORS: [BuiltinErrorDescriptor; 3] = [
    PLOTYY_ERROR_INVALID_ARGUMENT,
    PLOTYY_ERROR_INTERNAL,
    PLOTYY_ERROR_TOO_MANY_OUTPUTS,
];

pub const PLOTYY_ERROR_TOO_MANY_OUTPUTS: BuiltinErrorDescriptor = BuiltinErrorDescriptor {
    code: "RM.PLOTYY.TOO_MANY_OUTPUTS",
    identifier: Some("RunMat:plotting:TooManyOutputs"),
    when: "More than 3 output arguments are requested.",
    message: "plotyy: too many output arguments",
};
