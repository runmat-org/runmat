//! FEA pipeline scaffolding for assembly, solve orchestration, and diagnostics.

pub mod assembly;
pub mod contracts;
pub mod diagnostics;
pub mod fixtures;
pub mod operator;
pub mod parity;
pub mod physics;
pub mod pipeline;
pub mod post;
pub mod solve;
pub(crate) mod thermo;

pub use contracts::*;
pub use pipeline::linear_static::{run_linear_static, run_linear_static_with_options};
pub use pipeline::modal::{run_modal, run_modal_with_options};
pub use pipeline::nonlinear::{run_nonlinear, run_nonlinear_with_options};
pub use pipeline::thermal::{run_thermal, run_thermal_with_options};
pub use pipeline::transient::{run_transient, run_transient_with_options};

#[cfg(test)]
mod tests;
