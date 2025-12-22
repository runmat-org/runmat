#![cfg_attr(target_arch = "wasm32", allow(clippy::new_without_default))]

#[cfg(not(target_arch = "wasm32"))]
use runmat_runtime as _;

#[cfg(feature = "native")]
pub mod native;

#[cfg(feature = "native")]
pub mod backend;

#[cfg(feature = "wasm")]
pub mod wasm;

pub mod core;
