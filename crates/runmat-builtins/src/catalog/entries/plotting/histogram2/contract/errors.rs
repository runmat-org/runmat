use crate::*;

pub const HISTOGRAM2_ERROR_INVALID_ARGUMENT: BuiltinErrorDescriptor = BuiltinErrorDescriptor {
    code: "RM.HISTOGRAM2.INVALID_ARGUMENT",
    identifier: Some("RunMat:histogram2:InvalidArgument"),
    when: "Input arrays, bin controls, or chart options are malformed.",
    message: "histogram2: invalid argument",
};

pub const HISTOGRAM2_ERROR_INTERNAL: BuiltinErrorDescriptor = BuiltinErrorDescriptor {
    code: "RM.HISTOGRAM2.INTERNAL",
    identifier: Some("RunMat:histogram2:Internal"),
    when: "RunMat cannot compute bins, register state, or render the chart.",
    message: "histogram2: internal error",
};

pub(super) const ERRORS: [BuiltinErrorDescriptor; 3] = [
    HISTOGRAM2_ERROR_INVALID_ARGUMENT,
    HISTOGRAM2_ERROR_INTERNAL,
    crate::COORDINATE_ERROR_TOO_MANY_OUTPUTS,
];
