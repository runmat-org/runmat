mod abi;
mod async_abi;
mod async_request;
mod boundary;
#[cfg(not(target_family = "wasm"))]
mod gpu_abi;
#[cfg(not(target_family = "wasm"))]
pub(crate) use gpu_abi::native_device_address;
#[cfg(all(test, not(target_family = "wasm")))]
pub(crate) use gpu_abi::NativeInvocationContext;
mod service_slot;
mod services;

pub use abi::{MexCallState, MexDiagnostic, MexHostApiV1, MEX_HOST_ABI_VERSION};
#[doc(hidden)]
pub use async_request::MexAsyncThreadBound;
pub use async_request::{
    MexAsyncHostServices, MexAsyncOperation, MexAsyncResult, MexAsyncResultHandle,
};
pub use boundary::{DirectMexBoundaryHostServices, MexBoundaryHostServices, MxBoundaryCodec};
pub use service_slot::ConcurrentMexBoundaryHostServices;
pub(crate) use service_slot::{BoundaryServiceSlot, LocalBoundaryServiceGuard};
pub use services::{
    MexCallOutput, MexCancellationScope, MexEngineCompletion, MexHostServices,
    UnavailableMexHostServices,
};
