//! Training data and terminal-loss preparation for the shared network engine.

pub(super) mod admission;
pub(super) mod dataset;
pub(super) mod factors;
pub(super) mod gradients;
pub(super) mod labels;
pub(super) mod loss;
pub(super) mod optimizer;
pub(super) mod progress;
pub(super) mod resources;
pub(super) mod target;
pub(super) mod trainer;

#[cfg(test)]
mod public_recurrent_tests;
