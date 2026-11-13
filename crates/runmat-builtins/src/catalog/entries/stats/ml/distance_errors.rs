//! Common public diagnostics for the distance and nearest-neighbor family.
use crate::BuiltinErrorDescriptor;

pub const DISTANCE_ERROR_INVALID_ARGUMENT: BuiltinErrorDescriptor = BuiltinErrorDescriptor {
    code: "RM.DISTANCE.INVALID_ARGUMENT",
    identifier: Some("RunMat:distance:InvalidArgument"),
    when: "Distance inputs, dimensions, metrics, or controls are malformed.",
    message: "distance helper: invalid argument",
};

pub const DISTANCE_ERROR_INTERNAL: BuiltinErrorDescriptor = BuiltinErrorDescriptor {
    code: "RM.DISTANCE.INTERNAL",
    identifier: Some("RunMat:distance:Internal"),
    when: "RunMat cannot allocate or construct a distance output.",
    message: "distance helper: internal error",
};

pub const DISTANCE_ERRORS: [BuiltinErrorDescriptor; 2] =
    [DISTANCE_ERROR_INVALID_ARGUMENT, DISTANCE_ERROR_INTERNAL];
