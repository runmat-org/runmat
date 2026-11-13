//! Acceleration builtins: gpuArray, gather, gpuDevice, gpuInfo.
//!
//! These builtins provide explicit GPU array support using the runmat-accelerate-api
//! provider interface. If no provider is registered, calls will return an error.

pub mod arrayfun;
mod device_metadata;
pub mod gather;
pub mod gpuarray;
pub mod gpudevice;
pub mod gpuinfo;
pub mod pagefun;
