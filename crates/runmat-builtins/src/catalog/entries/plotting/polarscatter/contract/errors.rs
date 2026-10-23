use crate::*;

pub const POLARSCATTER_ERROR_INVALID_ARGUMENT: BuiltinErrorDescriptor = BuiltinErrorDescriptor {
    code: "RM.POLARSCATTER.INVALID_ARGUMENT",
    identifier: Some("RunMat:polarscatter:InvalidArgument"),
    when: "Input data, size/color/style arguments, or axes targeting are invalid.",
    message: "polarscatter: invalid argument",
};

pub const POLARSCATTER_ERROR_INTERNAL: BuiltinErrorDescriptor = BuiltinErrorDescriptor {
    code: "RM.POLARSCATTER.INTERNAL",
    identifier: Some("RunMat:polarscatter:Internal"),
    when: "Internal polar scatter construction or rendering fails unexpectedly.",
    message: "polarscatter: internal operation failed",
};

pub(super) const POLARSCATTER_ERRORS: [BuiltinErrorDescriptor; 3] = [
    POLARSCATTER_ERROR_INVALID_ARGUMENT,
    POLARSCATTER_ERROR_INTERNAL,
    crate::COORDINATE_ERROR_TOO_MANY_OUTPUTS,
];
