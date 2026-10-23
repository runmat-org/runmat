use crate::*;

pub const LOGLOG_ERROR_PLOT_FAILED: BuiltinErrorDescriptor = BuiltinErrorDescriptor {
    code: "RM.LOGLOG.PLOT_FAILED",
    identifier: Some("RunMat:loglog:PlotFailed"),
    when: "Underlying plot rendering/parsing fails while building the log-log plot.",
    message: "loglog: plot operation failed",
};

pub const LOGLOG_ERROR_LOG_AXIS_FAILED: BuiltinErrorDescriptor = BuiltinErrorDescriptor {
    code: "RM.LOGLOG.LOG_AXIS_FAILED",
    identifier: Some("RunMat:loglog:LogAxisFailed"),
    when: "Applying logarithmic axis modes fails.",
    message: "loglog: failed to apply logarithmic axis mode",
};

pub(super) const LOGLOG_ERRORS: [BuiltinErrorDescriptor; 3] = [
    LOGLOG_ERROR_PLOT_FAILED,
    LOGLOG_ERROR_LOG_AXIS_FAILED,
    crate::COORDINATE_ERROR_TOO_MANY_OUTPUTS,
];
