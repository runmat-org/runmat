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
pub mod progress;
pub mod solve;

pub use contracts::*;
pub use pipeline::electromagnetic::{run_electromagnetic, run_electromagnetic_with_options};
pub use pipeline::linear_static::{run_linear_static, run_linear_static_with_options};
pub use pipeline::modal::{run_modal, run_modal_with_options};
pub use pipeline::nonlinear::{run_nonlinear, run_nonlinear_with_options};
pub use pipeline::thermal::{run_thermal, run_thermal_with_options};
pub use pipeline::transient::{run_transient, run_transient_with_options};
pub use progress::{
    check_cancelled as check_fea_cancelled, emit_phase as emit_fea_progress_phase,
    emit_progress as emit_fea_progress, is_cancelled as is_fea_cancelled,
    replace_fea_progress_context, FeaCancellationPredicate, FeaProgressContextGuard,
    FeaProgressEvent, FeaProgressHandler, FeaProgressPhase, FeaProgressStatus,
};

#[cfg(test)]
mod tests;
