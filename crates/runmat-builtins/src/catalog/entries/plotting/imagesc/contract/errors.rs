use crate::*;

pub const IMAGESC_ERROR_INVALID_ARGUMENT: BuiltinErrorDescriptor = BuiltinErrorDescriptor {
    code: "RM.IMAGESC.INVALID_ARGUMENT",
    identifier: Some("RunMat:imagesc:InvalidArgument"),
    when: "Image data, axis inputs, or name/value style arguments are invalid.",
    message: "imagesc: invalid argument",
};

pub const IMAGESC_ERROR_INTERNAL: BuiltinErrorDescriptor = BuiltinErrorDescriptor {
    code: "RM.IMAGESC.INTERNAL",
    identifier: Some("RunMat:imagesc:Internal"),
    when: "Internal image/surface construction or rendering fails unexpectedly.",
    message: "imagesc: internal operation failed",
};

pub(super) const IMAGESC_ERRORS: [BuiltinErrorDescriptor; 3] = [
    IMAGESC_ERROR_INVALID_ARGUMENT,
    IMAGESC_ERROR_INTERNAL,
    crate::COORDINATE_ERROR_TOO_MANY_OUTPUTS,
];
