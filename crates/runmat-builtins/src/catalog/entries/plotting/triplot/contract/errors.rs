use crate::*;

pub const TRIPLOT_ERROR_INVALID_ARGUMENT: BuiltinErrorDescriptor = BuiltinErrorDescriptor {
    code: "RM.TRIPLOT.INVALID_ARGUMENT",
    identifier: Some("RunMat:triplot:InvalidArgument"),
    when: "Connectivity, coordinate, axes, or style inputs are malformed.",
    message: "triplot: invalid argument",
};

pub const TRIPLOT_ERROR_INTERNAL: BuiltinErrorDescriptor = BuiltinErrorDescriptor {
    code: "RM.TRIPLOT.INTERNAL",
    identifier: Some("RunMat:triplot:Internal"),
    when: "Internal line construction or plotting state update fails.",
    message: "triplot: internal operation failed",
};

pub(super) const TRIPLOT_ERRORS: [BuiltinErrorDescriptor; 3] = [
    TRIPLOT_ERROR_INVALID_ARGUMENT,
    TRIPLOT_ERROR_INTERNAL,
    COORDINATE_ERROR_TOO_MANY_OUTPUTS,
];
