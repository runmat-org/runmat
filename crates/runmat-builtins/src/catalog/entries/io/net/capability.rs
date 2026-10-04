use crate::BuiltinErrorDescriptor;

/// The current TCP transport is backed by native operating-system sockets.
pub const NET_ERROR_UNSUPPORTED_TARGET: BuiltinErrorDescriptor = BuiltinErrorDescriptor {
    code: "RM.NET.UNSUPPORTED_TARGET",
    identifier: Some("RunMat:network:UnsupportedTarget"),
    when: "A native TCP socket operation is requested in a browser target.",
    message: "native TCP sockets are unavailable on this target",
};
