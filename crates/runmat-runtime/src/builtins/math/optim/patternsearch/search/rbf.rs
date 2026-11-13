//! Cubic model search uses actual recent original callback measurements.
mod bounds;
mod history;
mod model;
mod run;
pub(in crate::builtins::math::optim::patternsearch) use history::{Coordinates, History};
pub(in crate::builtins::math::optim::patternsearch) use run::run;

#[cfg(test)]
mod tests;
