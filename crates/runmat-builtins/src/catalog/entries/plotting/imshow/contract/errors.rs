use crate::*;

pub const IMSHOW_ERROR_INVALID_ARGUMENT: BuiltinErrorDescriptor = BuiltinErrorDescriptor {
    code: "RM.IMSHOW.INVALID_ARGUMENT",
    identifier: Some("RunMat:imshow:InvalidArgument"),
    when: "Image input, filename, or display range arguments are malformed or unsupported.",
    message: "imshow: invalid argument",
};

pub const IMSHOW_ERROR_INTERNAL: BuiltinErrorDescriptor = BuiltinErrorDescriptor {
    code: "RM.IMSHOW.INTERNAL",
    identifier: Some("RunMat:imshow:Internal"),
    when: "Internal image decoding or render preparation fails unexpectedly.",
    message: "imshow: internal operation failed",
};

pub(super) const IMSHOW_ERRORS: [BuiltinErrorDescriptor; 3] = [
    IMSHOW_ERROR_INVALID_ARGUMENT,
    IMSHOW_ERROR_INTERNAL,
    crate::COORDINATE_ERROR_TOO_MANY_OUTPUTS,
];
