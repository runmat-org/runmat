//! Shared helpers for modern builtin implementations.
//!
//! This module hosts small utility subsystems that new-style builtins
//! can depend on without reaching back into the legacy runtime modules.

pub mod gpu_helpers;
pub mod spec;
pub mod tensor;

#[cfg(test)]
pub mod test_support;
