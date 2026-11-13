use crate::*;

pub const FILL_INVALID: BuiltinErrorDescriptor = BuiltinErrorDescriptor {
    code: "RM.FILL.INVALID_ARGUMENT",
    identifier: Some("RunMat:fill:InvalidArgument"),
    when: "Coordinate/color groups, axes targeting, or properties are invalid.",
    message: "fill: invalid argument",
};

pub(super) const ERRORS: [BuiltinErrorDescriptor; 2] =
    [FILL_INVALID, COORDINATE_ERROR_TOO_MANY_OUTPUTS];
