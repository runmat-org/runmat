mod cache;
mod compile;
mod deopt;
mod dependencies;
mod error;
mod invoke;
#[cfg(test)]
mod native_frame_tests;
pub(crate) mod placement;
mod super_constructor;

pub(crate) use cache::GenericNativeCache;
pub(crate) use compile::capture_liveness_admission;
pub(crate) use invoke::{invoke, NativeInvocation, RootedNativeExecution};
