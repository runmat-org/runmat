use core::ffi::c_void;

/// Host-owned invocation context. Extensions must treat this pointer as opaque.
pub type RunMatExtensionContext = c_void;

/// Extension-owned instance state returned by initialization.
pub type RunMatExtensionInstance = c_void;

/// Host-owned cancellation token. Its state is queried through the host vtable.
pub type RunMatCancellationToken = c_void;
