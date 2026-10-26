//! Shared catalog contract for closed FEA policy choices and typed physical
//! signal constructors (`fea.<Quantity>Signal.constant*`, `.fromTimeSamples`,
//! `.fromFunction`).
//!
//! Zero-argument choices and constant signals use the fixed-arity
//! `fea_contract::policy_types` layout. Time-sampled and callable-backed
//! signals need cross-argument and Name/Value checks, so their descriptors and
//! inference live here. Each catalog identity still supplies its own
//! signature label and documentation.

pub mod classes;
pub mod descriptor;
pub mod entry;
pub mod inference;
pub mod integer;

#[cfg(test)]
mod tests;
