//! Shared helpers for modern builtin implementations.
//!
//! This module hosts small utility subsystems that new-style builtins
//! can depend on without reaching back into the legacy runtime modules.

pub mod broadcast;
pub mod format;
pub mod fs;
pub mod gpu_helpers;
pub mod linalg;
pub mod path_search;
pub mod path_state;
pub mod random;
pub mod random_args;
pub mod residency;
pub mod shape;
pub mod spec;
pub mod tensor;

pub use spec::builtin_doc_texts;
pub use spec::DocTextInventory;

#[cfg(test)]
pub mod test_support;
