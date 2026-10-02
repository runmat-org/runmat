//! Shared optimization option identities.
mod options;
#[cfg(test)]
mod options_tests;

pub use options::{OptimAlgorithm, OptimDisplay, OptimOptionField, OptimSolver, OptimSwitch};
