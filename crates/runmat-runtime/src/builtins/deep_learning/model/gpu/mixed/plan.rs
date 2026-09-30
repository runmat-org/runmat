//! One typed physical stage sequence for resident execution and proposed host
//! uploads. Lowering is pure; provider and live-owner admission is separate.

mod admission;
mod lowering;

use super::super::*;
use crate::builtins::deep_learning::model::plan::normalization::NormalizationStage;
use runmat_accelerate_api::GpuTensorLayout;

pub(super) use admission::admit;
pub(super) use admission::admit_for_tape;
pub(super) use lowering::lower;

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub(super) enum RoutePurpose {
    ExistingMixedForward,
    ResidentTape,
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
