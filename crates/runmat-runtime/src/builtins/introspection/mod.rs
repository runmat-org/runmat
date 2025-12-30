#![allow(unused_imports)]

//! Introspection builtins.

pub mod class;
pub(crate) mod isa;
pub(crate) mod ischar;
pub(crate) mod isstring;
pub mod which;
pub mod who;
pub mod whos;
pub use class::*;
