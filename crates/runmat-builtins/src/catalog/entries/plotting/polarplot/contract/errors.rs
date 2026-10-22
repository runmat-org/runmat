use crate::*;

pub const POLARPLOT_ERROR_INVALID_ARGUMENT: BuiltinErrorDescriptor = BuiltinErrorDescriptor {
    code: "RM.POLARPLOT.INVALID_ARGUMENT",
    identifier: Some("RunMat:polarplot:InvalidArgument"),
    when: "Input data, series grammar, axes targeting, or style arguments are invalid.",
    message: "polarplot: invalid argument",
};

pub const POLARPLOT_ERROR_INTERNAL: BuiltinErrorDescriptor = BuiltinErrorDescriptor {
    code: "RM.POLARPLOT.INTERNAL",
    identifier: Some("RunMat:polarplot:Internal"),
    when: "Internal polar line construction or rendering fails unexpectedly.",
    message: "polarplot: internal operation failed",
};

pub(super) const POLARPLOT_ERRORS: [BuiltinErrorDescriptor; 3] = [
    POLARPLOT_ERROR_INVALID_ARGUMENT,
    POLARPLOT_ERROR_INTERNAL,
    crate::COORDINATE_ERROR_TOO_MANY_OUTPUTS,
];
