//! Shared helpers for modern builtin implementations.
//!
//! This module hosts small utility subsystems that new-style builtins
//! can depend on without reaching back into the legacy runtime modules.

pub mod gpu_helpers;
pub mod random;
pub mod random_args;
pub mod shape;
pub mod spec;
pub mod tensor;

pub use crate::register_builtin_doc_text;
pub use crate::register_builtin_fusion_spec;
pub use crate::register_builtin_gpu_spec;
pub use spec::builtin_doc_texts;
pub use spec::DocTextInventory;

#[cfg(test)]
pub mod test_support;
