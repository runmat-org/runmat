use crate::*;

pub const SEMILOGX_ERROR_PLOT_FAILED: BuiltinErrorDescriptor = BuiltinErrorDescriptor {
    code: "RM.SEMILOGX.PLOT_FAILED",
    identifier: Some("RunMat:semilogx:PlotFailed"),
    when: "Underlying plot rendering/parsing fails while building the semilog plot.",
    message: "semilogx: plot operation failed",
};

pub const SEMILOGX_ERROR_LOG_AXIS_FAILED: BuiltinErrorDescriptor = BuiltinErrorDescriptor {
    code: "RM.SEMILOGX.LOG_AXIS_FAILED",
    identifier: Some("RunMat:semilogx:LogAxisFailed"),
    when: "Applying logarithmic X-axis mode fails.",
    message: "semilogx: failed to apply logarithmic axis mode",
};

pub(super) const SEMILOGX_ERRORS: [BuiltinErrorDescriptor; 3] = [
    SEMILOGX_ERROR_PLOT_FAILED,
    SEMILOGX_ERROR_LOG_AXIS_FAILED,
    crate::COORDINATE_ERROR_TOO_MANY_OUTPUTS,
];
