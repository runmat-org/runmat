use crate::*;

pub const IMAGE_ERROR_INVALID_ARGUMENT: BuiltinErrorDescriptor = BuiltinErrorDescriptor {
    code: "RM.IMAGE.INVALID_ARGUMENT",
    identifier: Some("RunMat:image:InvalidArgument"),
    when: "Image data, axis inputs, or name/value style arguments are invalid.",
    message: "image: invalid argument",
};

pub const IMAGE_ERROR_INTERNAL: BuiltinErrorDescriptor = BuiltinErrorDescriptor {
    code: "RM.IMAGE.INTERNAL",
    identifier: Some("RunMat:image:Internal"),
    when: "Internal image/surface construction or rendering fails unexpectedly.",
    message: "image: internal operation failed",
};

pub(super) const IMAGE_ERRORS: [BuiltinErrorDescriptor; 3] = [
    IMAGE_ERROR_INVALID_ARGUMENT,
    IMAGE_ERROR_INTERNAL,
    crate::COORDINATE_ERROR_TOO_MANY_OUTPUTS,
];
