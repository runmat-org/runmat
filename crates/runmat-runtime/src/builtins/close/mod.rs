//! Canonical `close` builtin dispatcher.
//!
//! This module owns the single runtime registration for `close` and routes
//! requests to plotting or networking close handlers.

mod dispatch;

#[cfg(test)]
mod tests;

pub use dispatch::close_builtin;
pub use runmat_builtins::{CLOSE_DESCRIPTOR, CLOSE_EXTENSIONS, CLOSE_INTEGER_CAPABILITIES};

#[cfg(target_arch = "wasm32")]
pub(crate) use dispatch::__runmat_wasm_register_builtin_close_builtin;
