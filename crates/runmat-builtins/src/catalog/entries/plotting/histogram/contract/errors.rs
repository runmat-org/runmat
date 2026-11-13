use crate::*;

pub const HISTOGRAM_ERROR_INVALID_ARGUMENT: BuiltinErrorDescriptor = BuiltinErrorDescriptor {
    code: "RM.HISTOGRAM.INVALID_ARGUMENT",
    identifier: Some("RunMat:histogram:InvalidArgument"),
    when: "Histogram input arrays, bins, or name/value options are malformed or incompatible.",
    message: "histogram: invalid argument",
};

pub const HISTOGRAM_ERROR_INTERNAL: BuiltinErrorDescriptor = BuiltinErrorDescriptor {
    code: "RM.HISTOGRAM.INTERNAL",
    identifier: Some("RunMat:histogram:Internal"),
    when: "Internal render preparation or histogram state registration fails.",
    message: "histogram: internal operation failed",
};

pub(super) const HISTOGRAM_ERRORS: [BuiltinErrorDescriptor; 3] = [
    HISTOGRAM_ERROR_INVALID_ARGUMENT,
    HISTOGRAM_ERROR_INTERNAL,
    crate::COORDINATE_ERROR_TOO_MANY_OUTPUTS,
];
