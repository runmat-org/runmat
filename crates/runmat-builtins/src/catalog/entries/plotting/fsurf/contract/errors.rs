use crate::*;

pub const FSURF_ERROR_INVALID_ARGUMENT: BuiltinErrorDescriptor = BuiltinErrorDescriptor {
    code: "RM.FSURF.INVALID_ARGUMENT",
    identifier: Some("RunMat:fsurf:InvalidArgument"),
    when:
        "Function handles, domains, mesh density, axes target, or surface properties are invalid.",
    message: "fsurf: invalid argument",
};

pub const FSURF_ERROR_EVALUATION: BuiltinErrorDescriptor = BuiltinErrorDescriptor {
    code: "RM.FSURF.EVALUATION",
    identifier: Some("RunMat:fsurf:EvaluationFailed"),
    when: "A sampled function handle fails or does not return a scalar numeric value.",
    message: "fsurf: function evaluation failed",
};

pub const FSURF_ERROR_INTERNAL: BuiltinErrorDescriptor = BuiltinErrorDescriptor {
    code: "RM.FSURF.INTERNAL",
    identifier: Some("RunMat:fsurf:Internal"),
    when: "Surface construction or rendering fails unexpectedly.",
    message: "fsurf: internal operation failed",
};

pub(super) const FSURF_ERRORS: [BuiltinErrorDescriptor; 4] = [
    FSURF_ERROR_INVALID_ARGUMENT,
    FSURF_ERROR_EVALUATION,
    FSURF_ERROR_INTERNAL,
    COORDINATE_ERROR_TOO_MANY_OUTPUTS,
];
