use crate::*;

pub const BAR_ERROR_INVALID_ARGUMENT: BuiltinErrorDescriptor = BuiltinErrorDescriptor {
    code: "RM.BAR.INVALID_ARGUMENT",
    identifier: Some("RunMat:bar:InvalidArgument"),
    when: "Bar input arrays, style tokens, or name/value arguments are invalid.",
    message: "bar: invalid argument",
};

pub const BAR_ERROR_INTERNAL: BuiltinErrorDescriptor = BuiltinErrorDescriptor {
    code: "RM.BAR.INTERNAL",
    identifier: Some("RunMat:bar:Internal"),
    when: "Renderer/GPU conversion fails while building bar geometry.",
    message: "bar: internal operation failed",
};

pub(super) const BAR_ERRORS: [BuiltinErrorDescriptor; 3] = [
    BAR_ERROR_INVALID_ARGUMENT,
    BAR_ERROR_INTERNAL,
    crate::COORDINATE_ERROR_TOO_MANY_OUTPUTS,
];
