//! Shared ingress diagnostics for coordinate chart identities.
use crate::BuiltinErrorDescriptor;

pub const COORDINATE_ERROR_TOO_MANY_OUTPUTS: BuiltinErrorDescriptor = BuiltinErrorDescriptor {
    code: "RM.PLOTTING.COORDINATE.TOO_MANY_OUTPUTS",
    identifier: Some("RunMat:plotting:TooManyOutputs"),
    when: "More than one output is requested from a coordinate chart constructor.",
    message: "plotting: too many output arguments",
};
