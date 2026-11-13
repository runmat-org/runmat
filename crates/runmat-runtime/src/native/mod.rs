//! Versioned executor-neutral ABI shared by native code generators and runtime hosts.
//!
//! The ABI deliberately carries opaque value/root tokens rather than Rust `Value`
//! layouts. This keeps numeric storage private and lets the final integer seal evolve
//! without creating a second value system. Pointers are borrowed for one host call;
//! ownership always remains with the documented host-side arena or root table.

mod call;
mod control;
mod environment;
mod frame;
mod host;
mod root;
mod site;
mod source;
mod validation;
mod value;
mod version;
#[cfg(not(target_arch = "wasm32"))]
mod probe;

pub use call::*;
pub use control::*;
pub use environment::*;
pub use frame::*;
pub use host::*;
#[cfg(not(target_arch = "wasm32"))]
pub use probe::*;
pub use root::*;
pub use site::*;
pub use source::*;
pub use validation::*;
pub use value::*;
pub use version::*;
