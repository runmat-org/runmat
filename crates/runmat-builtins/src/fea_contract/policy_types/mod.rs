//! Shared catalog contract for typed FEA policy values: accuracy plans,
//! execution plans, signal knots and generator policies, response-spectrum
//! policies, and observation policy choices.
//!
//! Each catalog identity supplies its own documentation, signature, argument
//! rules, and integer facts; this module only owns the shared layout, the
//! static argument checks, and the class identities those values carry.

pub mod classes;
pub mod descriptor;
pub mod documentation;
pub mod entry;
pub mod inference;
pub mod integer;

#[cfg(test)]
pub(crate) mod conformance;
#[cfg(test)]
mod tests;
