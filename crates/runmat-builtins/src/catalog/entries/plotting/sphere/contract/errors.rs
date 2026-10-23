use crate::*;

pub const SPHERE_ERROR_OUTPUT_COUNT: BuiltinErrorDescriptor = BuiltinErrorDescriptor {
    code: "RM.SPHERE.OUTPUT_COUNT",
    identifier: Some("RunMat:sphere:OutputCount"),
    when: "More than three outputs are requested.",
    message: "sphere: too many output arguments; maximum is 3",
};

pub const SPHERE_ERROR_ARG_COUNT: BuiltinErrorDescriptor = BuiltinErrorDescriptor {
    code: "RM.SPHERE.ARG_COUNT",
    identifier: Some("RunMat:sphere:InvalidArgument"),
    when: "More than one input argument is supplied.",
    message: "sphere: expected zero or one input argument",
};

pub const SPHERE_ERROR_INVALID_N: BuiltinErrorDescriptor = BuiltinErrorDescriptor {
    code: "RM.SPHERE.INVALID_N",
    identifier: Some("RunMat:sphere:InvalidArgument"),
    when: "The n argument is not a finite non-negative integer scalar.",
    message: "sphere: n must be a finite non-negative integer scalar",
};

pub const SPHERE_ERROR_TOO_LARGE: BuiltinErrorDescriptor = BuiltinErrorDescriptor {
    code: "RM.SPHERE.N_TOO_LARGE",
    identifier: Some("RunMat:sphere:OutOfMemory"),
    when: "The requested coordinate grid is too large for RunMat's bounded allocation policy.",
    message: "sphere: requested grid is too large",
};

pub const SPHERE_ERROR_INTERNAL: BuiltinErrorDescriptor = BuiltinErrorDescriptor {
    code: "RM.SPHERE.INTERNAL",
    identifier: Some("RunMat:sphere:Internal"),
    when: "Coordinate tensor construction or plotting fails unexpectedly.",
    message: "sphere: internal operation failed",
};

pub(super) const SPHERE_ERRORS: [BuiltinErrorDescriptor; 5] = [
    SPHERE_ERROR_OUTPUT_COUNT,
    SPHERE_ERROR_ARG_COUNT,
    SPHERE_ERROR_INVALID_N,
    SPHERE_ERROR_TOO_LARGE,
    SPHERE_ERROR_INTERNAL,
];
