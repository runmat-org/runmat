//! Immutable leases for the stages that actually ran on a resident provider.
//!
//! The trace does not re-derive network operations or retain an execution
//! context.  Each handle is an owning view lease; dropping a failed or
//! cancelled reverse pass releases only its new results, not borrowed inputs.

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
    Relu,
    Elu {
        alpha: f64,
    },
    Softmax,
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
