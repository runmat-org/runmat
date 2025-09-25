#![allow(unused_imports)]

//! Placeholder module for math builtins. Individual builtin files (e.g. `sin.rs`, `sum.rs`) will
//! live alongside this module as migration progresses.

// Re-export legacy implementation for now to avoid breaking existing callers.
pub use crate::mathematics::*;
