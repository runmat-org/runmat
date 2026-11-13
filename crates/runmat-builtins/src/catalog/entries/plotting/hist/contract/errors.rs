use crate::*;

pub const HIST_ERROR_INVALID_ARGUMENT: BuiltinErrorDescriptor = BuiltinErrorDescriptor {
    code: "RM.HIST.INVALID_ARGUMENT",
    identifier: Some("RunMat:hist:InvalidArgument"),
    when: "Histogram inputs, bins, normalization, weights, or style arguments are invalid.",
    message: "hist: invalid argument",
};

pub const HIST_ERROR_INTERNAL: BuiltinErrorDescriptor = BuiltinErrorDescriptor {
    code: "RM.HIST.INTERNAL",
    identifier: Some("RunMat:hist:Internal"),
    when: "Internal histogram rendering or device conversion fails.",
    message: "hist: internal operation failed",
};

pub(super) const HIST_ERRORS: [BuiltinErrorDescriptor; 3] = [
    HIST_ERROR_INVALID_ARGUMENT,
    HIST_ERROR_INTERNAL,
    HIST_ERROR_TOO_MANY_OUTPUTS,
];

pub const HIST_ERROR_TOO_MANY_OUTPUTS: BuiltinErrorDescriptor = BuiltinErrorDescriptor {
    code: "RM.HIST.TOO_MANY_OUTPUTS",
    identifier: Some("RunMat:plotting:TooManyOutputs"),
    when: "More than 2 output arguments are requested.",
    message: "hist: too many output arguments",
};
