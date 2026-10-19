mod cache;
mod compile;
mod deopt;
mod dependencies;
mod error;
mod invoke;
pub(crate) mod placement;
mod super_constructor;
#[cfg(test)]
mod native_frame_tests;

pub(crate) use cache::GenericNativeCache;
pub(crate) use invoke::{invoke, NativeInvocation, RootedNativeExecution};
