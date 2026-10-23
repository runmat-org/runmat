use crate::*;

pub const STEM_ERROR_INVALID_ARGUMENT: BuiltinErrorDescriptor = BuiltinErrorDescriptor {
    code: "RM.STEM.INVALID_ARGUMENT",
    identifier: Some("RunMat:stem:InvalidArgument"),
    when: "Input data, axes targeting, or style arguments are invalid.",
    message: "stem: invalid argument",
};

pub const STEM_ERROR_INTERNAL: BuiltinErrorDescriptor = BuiltinErrorDescriptor {
    code: "RM.STEM.INTERNAL",
    identifier: Some("RunMat:stem:Internal"),
    when: "Internal plot construction or rendering fails unexpectedly.",
    message: "stem: internal operation failed",
};

pub(super) const STEM_ERRORS: [BuiltinErrorDescriptor; 3] = [
    STEM_ERROR_INVALID_ARGUMENT,
    STEM_ERROR_INTERNAL,
    crate::COORDINATE_ERROR_TOO_MANY_OUTPUTS,
];
