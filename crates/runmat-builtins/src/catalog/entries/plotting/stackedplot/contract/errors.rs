use crate::*;

pub const STACKEDPLOT_ERROR_INVALID_ARGUMENT: BuiltinErrorDescriptor = BuiltinErrorDescriptor {
    code: "RM.STACKEDPLOT.INVALID_ARGUMENT",
    identifier: Some("RunMat:stackedplot:InvalidArgument"),
    when: "Input data, table selectors, line style, parent, or StackedLineChart properties are invalid.",
    message: "stackedplot: invalid argument",
};

pub(super) const ERROR_INTERNAL: BuiltinErrorDescriptor = BuiltinErrorDescriptor {
    code: "RM.STACKEDPLOT.INTERNAL",
    identifier: Some("RunMat:stackedplot:Internal"),
    when: "Internal plotting state update fails.",
    message: "stackedplot: internal operation failed",
};

pub(super) const ERRORS: [BuiltinErrorDescriptor; 3] = [
    STACKEDPLOT_ERROR_INVALID_ARGUMENT,
    ERROR_INTERNAL,
    crate::COORDINATE_ERROR_TOO_MANY_OUTPUTS,
];
