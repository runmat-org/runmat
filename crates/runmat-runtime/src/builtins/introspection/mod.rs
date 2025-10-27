#![allow(unused_imports)]

//! Introspection builtins.

pub mod class;
mod isa;
pub mod which;
pub mod who;
pub mod whos;

pub use crate::introspection::*;
pub use class::*;
