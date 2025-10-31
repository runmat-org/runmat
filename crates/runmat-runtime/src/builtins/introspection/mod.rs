#![allow(unused_imports)]

//! Introspection builtins.

pub mod class;
mod isa;
mod ischar;
mod isstring;
pub mod which;
pub mod who;
pub mod whos;

pub use crate::introspection::*;
pub use class::*;
