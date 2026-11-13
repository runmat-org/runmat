use crate::*;

pub const MOVING_WINDOW_ERROR_HOST_WORKSPACE: BuiltinErrorDescriptor = BuiltinErrorDescriptor {
    code: "RM.MOVING_WINDOW.HOST_WORKSPACE",
    identifier: Some("RunMat:MovingWindow:HostWorkspace"),
    when: "Host numerical or gather scratch has no complete finite owner, or exact provider request-shape metadata cannot be admitted. Cancellation and issuer revocation retain canonical control diagnostics instead.",
    message: "moving-window host workspace admission failed",
};

pub const MOVING_WINDOW_ERROR_INVALID_ARGUMENT: BuiltinErrorDescriptor = BuiltinErrorDescriptor {
    code: "RM.MOVING.INVALID_ARGUMENT",
    identifier: None,
    when: "Window length, dimension, nanflag, or name-value options are invalid.",
    message: "moving window: invalid argument",
};

pub const MOVING_WINDOW_ERROR_INVALID_INPUT: BuiltinErrorDescriptor = BuiltinErrorDescriptor {
    code: "RM.MOVING.INVALID_INPUT",
    identifier: None,
    when: "Input values cannot be converted to a supported moving-window domain.",
    message: "moving window: invalid input",
};

pub const MOVING_WINDOW_ERROR_INTERNAL: BuiltinErrorDescriptor = BuiltinErrorDescriptor {
    code: "RM.MOVING.INTERNAL",
    identifier: None,
    when: "Moving-window execution fails due to allocation, provider, or shape operations.",
    message: "moving window: internal failure",
};
