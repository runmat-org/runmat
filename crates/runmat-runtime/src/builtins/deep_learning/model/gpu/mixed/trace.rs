//! Immutable leases for the stages that actually ran on a resident provider.
//!
//! The trace does not re-derive network operations or retain an execution
//! context.  Each handle is an owning view lease; dropping a failed or
//! cancelled reverse pass releases only its new results, not borrowed inputs.

use crate::builtins::deep_learning::model::gpu::normalization::NormalizationRecord;
use crate::builtins::deep_learning::model::plan::axes::AxisPlan;
use runmat_accelerate_api::DeepLearningConv1DRequest;
use runmat_accelerate_api::GpuTensorHandle;

pub(in crate::builtins::deep_learning) enum ResidentStageKind {
    Materialize,
    Transpose {
        materialize: bool,
    },
    Dense {
        layer_index: usize,
        weights: GpuTensorHandle,
    },
    Conv {
        layer_index: usize,
        request: DeepLearningConv1DRequest,
        weights: GpuTensorHandle,
        bias: GpuTensorHandle,
    },
    Relu,
    Elu {
        alpha: f64,
    },
    Softmax,
    Normalize {
        record: NormalizationRecord,
    },
    Pool {
        axis: usize,
        input_axes: AxisPlan,
        output_axes: AxisPlan,
    },
}

pub(in crate::builtins::deep_learning) struct ResidentStage {
    pub kind: ResidentStageKind,
    pub input: GpuTensorHandle,
    pub output: GpuTensorHandle,
}

pub(in crate::builtins::deep_learning) struct ResidentTrace {
    pub stages: Vec<ResidentStage>,
    pub output: GpuTensorHandle,
}

pub(in crate::builtins::deep_learning) struct ResidentExecution {
    pub output: GpuTensorHandle,
    pub trace: Option<ResidentTrace>,
}
