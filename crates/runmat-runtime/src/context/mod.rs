//! Explicit, session-owned runtime capabilities and invocation state.
//!
//! [`RuntimeContext`] is the one context propagated by Core, the VM, nested
//! calls, and future native/foreign/parallel adapters. Legacy ambient APIs are
//! migrated through [`legacy`] and must not become new semantic dependencies.

mod capability;
mod runtime;
mod scope;
mod services;
mod state;

pub mod legacy;

pub use capability::{RuntimeCapability, RuntimeCapabilityError};
pub use runtime::{
    RuntimeContext, RuntimeLanguageMode, DEFAULT_CALLSTACK_LIMIT, DEFAULT_ERROR_NAMESPACE,
};
pub use scope::{ContextFuture, RuntimeContextGuard};
pub use services::{
    ForeignCall, HostInteraction, NativeCapability, ParallelCapability, RuntimeAccelerationService,
    RuntimeCallRequest, RuntimeCallService, RuntimeErrorService, RuntimeForeignService,
    RuntimeHostService, RuntimeNativeService, RuntimeObjectService, RuntimeParallelService,
    RuntimeServicePorts, RuntimeWorkspaceService,
};
pub(crate) use state::RuntimeContextState;
