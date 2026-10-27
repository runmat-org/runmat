//! Shared static facts for the typed FEA model-object constructors: boundaries,
//! constraints and their support values, interfaces, domains, damping laws,
//! excitations and cases, initial conditions, procedure steps, step
//! activations, and property-region targets. Each catalog identity owns its
//! signature, documentation, and argument rules; this module owns only the
//! rule interpreter, the shared example fixture, and shared integer facts.

pub mod classes;
pub mod fixtures;
pub mod inference;
pub mod integer;
#[cfg(test)]
pub(crate) mod test_support;
