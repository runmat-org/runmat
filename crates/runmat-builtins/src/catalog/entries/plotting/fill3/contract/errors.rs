use crate::*;

pub const FILL3_ERROR_INVALID_ARGUMENT: BuiltinErrorDescriptor = BuiltinErrorDescriptor {
    code: "RM.FILL3.INVALID_ARGUMENT",
    identifier: Some("RunMat:fill3:InvalidArgument"),
    when: "Input groups, axes targeting, or trailing property pairs are invalid.",
    message: "fill3: invalid argument",
};

pub const FILL3_ERROR_INTERNAL: BuiltinErrorDescriptor = BuiltinErrorDescriptor {
    code: "RM.FILL3.INTERNAL",
    identifier: Some("RunMat:fill3:Internal"),
    when: "Internal patch construction or rendering fails unexpectedly.",
    message: "fill3: internal operation failed",
};

pub(super) const FILL3_ERRORS: [BuiltinErrorDescriptor; 3] = [
    FILL3_ERROR_INVALID_ARGUMENT,
    FILL3_ERROR_INTERNAL,
    COORDINATE_ERROR_TOO_MANY_OUTPUTS,
];
