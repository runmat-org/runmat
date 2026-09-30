//! Whole-network host admission and post-admission preparation.

mod controls;
mod geometry;
mod prepare;

pub(in crate::builtins::deep_learning) use controls::Controls;
pub(in crate::builtins::deep_learning) use geometry::{
    admit_training_geometry, HostNetworkPlan, StepDescriptor,
};

impl HostNetworkPlan<'_> {
    pub(in crate::builtins::deep_learning) fn output_axes(&self) -> &super::axes::AxisPlan {
        &self.output_axes
    }
}

use crate::builtins::deep_learning::deep_learning_error;
use crate::builtins::deep_learning::model::execution::host::network::{
    self, HostBackwardResult, HostTrace, HostValue,
};
use crate::builtins::deep_learning::model::state::StateUpdate;
use crate::BuiltinResult;

/// The saved forward trace pins its own admitted parameters and may be
/// differentiated repeatedly without consulting mutable network properties.
pub(in crate::builtins::deep_learning) fn reverse(
    trace: &HostTrace,
    output_gradient: HostValue,
    state_gradients: &[StateUpdate],
) -> BuiltinResult<HostBackwardResult> {
    network::backward(trace, output_gradient, state_gradients)
        .map_err(|error| deep_learning_error("dlgradient", format!("dlgradient: {error}")))
}

#[cfg(test)]
mod default_tests;
#[cfg(test)]
mod tests;
