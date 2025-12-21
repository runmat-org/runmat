#![cfg_attr(target_arch = "wasm32", allow(clippy::new_without_default))]

#[cfg(feature = "native")]
pub mod native;

#[cfg(feature = "native")]
pub mod backend;

#[cfg(feature = "wasm")]
pub mod wasm;

pub mod core;


