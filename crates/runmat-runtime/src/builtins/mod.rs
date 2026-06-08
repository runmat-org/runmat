//! New builtin set. Builtins are organised by category and re-exported from this module.
#[macro_use]
pub mod common;
pub mod acceleration;
pub mod array;
pub mod cells;
pub mod close;
pub mod comms;
pub mod constants;
pub mod containers;
pub mod control;
pub mod datetime;
pub mod diagnostics;
pub mod duration;
pub mod image;
pub mod introspection;
pub mod io;
pub mod logical;
pub mod math;
#[cfg(feature = "plot-core")]
pub mod plotting;
pub mod stats;
pub mod strings;
pub mod structs;
pub mod timing;
pub mod wasm_registry;
