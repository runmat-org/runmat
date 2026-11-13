use crate::*;

pub const SEMILOGY_ERROR_PLOT_FAILED: BuiltinErrorDescriptor = BuiltinErrorDescriptor {
    code: "RM.SEMILOGY.PLOT_FAILED",
    identifier: Some("RunMat:semilogy:PlotFailed"),
    when: "Underlying plot rendering/parsing fails while building the semilog plot.",
    message: "semilogy: plot operation failed",
};

pub const SEMILOGY_ERROR_LOG_AXIS_FAILED: BuiltinErrorDescriptor = BuiltinErrorDescriptor {
    code: "RM.SEMILOGY.LOG_AXIS_FAILED",
    identifier: Some("RunMat:semilogy:LogAxisFailed"),
    when: "Applying logarithmic Y-axis mode fails.",
    message: "semilogy: failed to apply logarithmic axis mode",
};

pub(super) const SEMILOGY_ERRORS: [BuiltinErrorDescriptor; 3] = [
    SEMILOGY_ERROR_PLOT_FAILED,
    SEMILOGY_ERROR_LOG_AXIS_FAILED,
    crate::COORDINATE_ERROR_TOO_MANY_OUTPUTS,
];
