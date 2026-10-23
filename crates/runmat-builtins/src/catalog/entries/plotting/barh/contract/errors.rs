use crate::*;

pub const BARH_ERROR_INVALID_ARGUMENT: BuiltinErrorDescriptor = BuiltinErrorDescriptor {
    code: "RM.BARH.INVALID_ARGUMENT",
    identifier: Some("RunMat:barh:InvalidArgument"),
    when: "Horizontal bar input arrays, style tokens, or name/value arguments are invalid.",
    message: "barh: invalid argument",
};

pub const BARH_ERROR_INTERNAL: BuiltinErrorDescriptor = BuiltinErrorDescriptor {
    code: "RM.BARH.INTERNAL",
    identifier: Some("RunMat:barh:Internal"),
    when: "Renderer/GPU conversion fails while building horizontal bar geometry.",
    message: "barh: internal operation failed",
};

pub(super) const BARH_ERRORS: [BuiltinErrorDescriptor; 3] = [
    BARH_ERROR_INVALID_ARGUMENT,
    BARH_ERROR_INTERNAL,
    crate::COORDINATE_ERROR_TOO_MANY_OUTPUTS,
];
