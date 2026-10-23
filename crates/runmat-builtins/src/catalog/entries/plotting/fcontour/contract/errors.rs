use crate::*;

pub const FCONTOUR_ERROR_INVALID_ARGUMENT: BuiltinErrorDescriptor = BuiltinErrorDescriptor {
    code: "RM.FCONTOUR.INVALID_ARGUMENT",
    identifier: Some("RunMat:fcontour:InvalidArgument"),
    when: "Function handle, domain, mesh density, axes target, levels, or contour properties are invalid.",
    message: "fcontour: invalid argument",
};

pub const FCONTOUR_ERROR_EVALUATION: BuiltinErrorDescriptor = BuiltinErrorDescriptor {
    code: "RM.FCONTOUR.EVALUATION",
    identifier: Some("RunMat:fcontour:EvaluationFailed"),
    when: "A sampled function handle fails or does not return a scalar numeric value.",
    message: "fcontour: function evaluation failed",
};

pub const FCONTOUR_ERROR_INTERNAL: BuiltinErrorDescriptor = BuiltinErrorDescriptor {
    code: "RM.FCONTOUR.INTERNAL",
    identifier: Some("RunMat:fcontour:Internal"),
    when: "Contour construction or rendering fails unexpectedly.",
    message: "fcontour: internal operation failed",
};

pub(super) const FCONTOUR_ERRORS: [BuiltinErrorDescriptor; 4] = [
    FCONTOUR_ERROR_INVALID_ARGUMENT,
    FCONTOUR_ERROR_EVALUATION,
    FCONTOUR_ERROR_INTERNAL,
    COORDINATE_ERROR_TOO_MANY_OUTPUTS,
];
