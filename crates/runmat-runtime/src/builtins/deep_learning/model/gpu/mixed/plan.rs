//! One typed physical stage sequence for resident execution and proposed host
//! uploads. Lowering is pure; provider and live-owner admission is separate.

mod admission;
mod host_admission;
mod lowering;

use super::super::*;
use crate::builtins::deep_learning::model::plan::normalization::NormalizationStage;
use runmat_accelerate_api::DeepLearningConv1DRequest;
use runmat_accelerate_api::GpuTensorLayout;

pub(super) use admission::admit;
pub(super) use admission::admit_for_tape;
pub(super) use host_admission::admit_host;
pub(super) use lowering::lower;

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub(super) enum RoutePurpose {
    ExistingMixedForward,
    ResidentTape,
    HostPlacement,
}

pub(super) enum Stage {
    Materialize,
    Transpose {
        output_axes: AxisPlan,
        output_layout: GpuTensorLayout,
        materialize: bool,
    },
    Dense {
        layer_index: usize,
    },
    Conv {
        layer_index: usize,
        request: DeepLearningConv1DRequest,
    },
    Relu,
    Elu {
        alpha: f64,
    },
    Softmax,
    Normalize {
        stage: NormalizationStage,
    },
    Pool {
        axis: usize,
        input_axes: AxisPlan,
        output_axes: AxisPlan,
    },
}

pub(super) struct Plan {
    pub stages: Vec<Stage>,
    pub output_axes: AxisPlan,
    pub consumed_resident_parameters: Vec<GpuTensorHandle>,
}

/// A complete pure host-source capability query. This is admission, not a
/// placement decision: only the placement consumer may authorize execution.
pub(in crate::builtins::deep_learning) struct AdmittedHostRoute<'h, 'n> {
    pub(super) host: &'h crate::builtins::deep_learning::model::plan::host::HostNetworkPlan<'n>,
    pub(super) plan: Plan,
    pub(super) provider: &'static dyn runmat_accelerate_api::AccelProvider,
    pub(in crate::builtins::deep_learning) route_facts: runmat_execution::Digest,
}

impl<'h, 'n> AdmittedHostRoute<'h, 'n> {
    /// Reproject the exact admitted stage sequence for read-only provider
    /// preparation inspection. Execution lowers the same `plan` after upload.
    pub(in crate::builtins::deep_learning) fn physical_route(
        &self,
    ) -> BuiltinResult<Option<runmat_accelerate_api::DeepLearningForwardRoute<'_>>> {
        super::physical::from_plan(self.host, &self.plan, self.host.function())
    }

    pub(in crate::builtins::deep_learning) fn provider(
        &self,
    ) -> &'static dyn runmat_accelerate_api::AccelProvider {
        self.provider
    }

    /// Exact canonical rows consumed by the selected executable stage plan.
    /// The route owns both the admitted plan and its borrowed network, so a
    /// caller cannot substitute similarly shaped values from another network.
    pub(in crate::builtins::deep_learning) fn consumed_parameter_views(
        &self,
    ) -> Vec<&crate::builtins::deep_learning::model::parameters::ParameterView<'n>> {
        super::parameters::consumed_parameter_rows(self.host.parameter_views(), &self.plan.stages)
    }
}
