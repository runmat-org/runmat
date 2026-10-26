//! Evaluation dispatch is the existing bounded population-wave owner.
pub(in crate::builtins::math::optim) mod source;
pub(in crate::builtins::math::optim) mod fitness;
pub(super) mod constraints;
pub(super) mod nonlinear;
pub(super) mod duplicates;
pub(super) mod generation;
#[cfg(test)]
mod tests;
