use crate::*;

pub(super) const LINE_ERROR_INVALID_ARGUMENT: BuiltinErrorDescriptor = BuiltinErrorDescriptor {
    code: "RM.LINE.INVALID_ARGUMENT",
    identifier: Some("RunMat:line:InvalidArgument"),
    when: "Line data, axes target, style arguments, or Name/Value pairs are invalid.",
    message: "line: invalid argument",
};

pub(super) const LINE_ERROR_INTERNAL: BuiltinErrorDescriptor = BuiltinErrorDescriptor {
    code: "RM.LINE.INTERNAL",
    identifier: Some("RunMat:line:Internal"),
    when: "Internal plotting state update fails.",
    message: "line: internal operation failed",
};

pub(super) const LINE_ERRORS: [BuiltinErrorDescriptor; 3] = [
    LINE_ERROR_INVALID_ARGUMENT,
    LINE_ERROR_INTERNAL,
    crate::COORDINATE_ERROR_TOO_MANY_OUTPUTS,
];
