//! Typed acceleration requests for deep-learning operations.

use crate::{GpuTensorHandle, GpuTensorView, NumericElementType, ProviderViewStep};

#[path = "deep_learning/route.rs"]
mod route;
pub use route::{DeepLearningForwardGeometry, DeepLearningForwardRouteError};

#[path = "deep_learning/route_signature.rs"]
mod route_signature;

#[path = "deep_learning/convolution.rs"]
mod convolution;
pub use convolution::{
    DeepLearningConv1DCapability, DeepLearningConv1DGradients, DeepLearningConv1DOperands,
    DeepLearningConv1DPaddingValue, DeepLearningConv1DRequest,
};

/// The physical input to a proposed forward route. A host source describes the
/// bytes that preparation would upload; it does not authorize or perform that
/// upload. A resident source retains its real allocation owner for admission.
#[derive(Debug, Clone)]
pub enum DeepLearningForwardSource<'a> {
    Host {
        element_type: NumericElementType,
        shape: Vec<usize>,
    },
    Resident(&'a GpuTensorHandle),
}

/// A parameter consumed by a physical route. `view_steps` are applied before
/// its consuming stage, using the same view-chain semantics as execution.
#[derive(Debug, Clone)]
pub struct DeepLearningForwardParameter<'a> {
    pub source: DeepLearningForwardSource<'a>,
    pub view_steps: Vec<ProviderViewStep>,
}

/// Typed binding roles for operations that consume learned values.
/// Indices address `DeepLearningForwardRoute::parameters` and are checked by
/// provider admission before any effect.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum DeepLearningForwardParameters {
    None,
    Dense { weights: usize, bias: usize },
    Conv1D { weights: usize, bias: usize },
    LayerNorm { scale: usize, offset: usize },
}

/// Already-lowered physical operation. Providers never parse layer names or
/// reinterpret semantic axes. `View` is the existing physical view operation;
/// other variants use the same request types as their individual kernels.
#[derive(Debug, Clone, PartialEq)]
pub enum DeepLearningForwardOperation {
    View(ProviderViewStep),
    FeedForward(DeepLearningFeedForwardStage),
    Pool1D {
        input_shape: Vec<usize>,
        route: DeepLearningPool1DRoute,
    },
    Conv1D(DeepLearningConv1DRequest),
    LayerNorm(DeepLearningLayerNormRequest),
}

#[derive(Debug, Clone, PartialEq)]
pub struct DeepLearningForwardStep {
    pub operation: DeepLearningForwardOperation,
    pub parameters: DeepLearningForwardParameters,
}

/// One proposed physical route. The producer must lower admitted semantic
/// geometry once; a provider rechecks the complete physical sequence and real
/// resident owners without uploading, allocating, or dispatching.
#[derive(Debug, Clone)]
pub struct DeepLearningForwardRoute<'a> {
    pub input: DeepLearningForwardSource<'a>,
    pub parameters: Vec<DeepLearningForwardParameter<'a>>,
    pub steps: Vec<DeepLearningForwardStep>,
    pub output_view: GpuTensorView,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct DeepLearningForwardResources {
    pub output_bytes: u64,
    /// Largest single physical stage value, including input and output. This
    /// is not a newly allocated intermediate or concurrent-memory estimate.
    pub largest_stage_value_bytes: u64,
    /// Known normalization work buffer demand. Other provider-private buffers
    /// are not represented here; shared placement must not treat this as peak.
    pub known_normalization_scratch_bytes: u64,
    pub upload_bytes: u64,
    pub resident_bytes: u64,
    /// Checked upper bound on WGPU buffers explicitly created for this route,
    /// including host uploads, outputs, owned temporaries, and dispatch
    /// uniforms. Resident input allocations are reported separately above.
    /// Driver-private memory and pipeline caches are outside this contract.
    /// `Unavailable` must not be interpreted as zero by placement.
    pub new_buffer_bytes_upper_bound: DeepLearningForwardBufferBound,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum DeepLearningForwardBufferBound {
    Known(u64),
    Unavailable,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum DeepLearningForwardCapability {
    Ready(DeepLearningForwardResources),
    UnsupportedOperation,
    ResourceLimit,
}

/// A resident global-average-pooling reduction over one physical axis.
/// The provider retains that axis with extent one; presentation may remove
/// it later without copying device bytes. Unlike a general mean reduction,
/// finite inputs whose raw sum overflows must still produce a finite mean.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct DeepLearningPool1DRequest {
    pub axis_zero_based: usize,
}

/// The complete device-side presentation required after a pooling reduction.
/// `source_shape` is the logical post-pool shape (possibly with the reduced
/// axis removed); `target_to_source` maps each final axis to that source axis.
/// Providers inspect this without allocating or dispatching any work.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct DeepLearningPool1DRoute {
    pub reduction: DeepLearningPool1DRequest,
    pub result_axis: DeepLearningPool1DResultAxis,
    pub source_shape: Vec<usize>,
    pub target_shape: Vec<usize>,
    pub target_to_source: Vec<usize>,
}

/// Whether the reduced physical axis survives as a singleton in the logical
/// source result, or is removed before output presentation.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum DeepLearningPool1DResultAxis {
    KeepSingleton,
    RemoveReduced,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum DeepLearningPool1DCapability {
    Ready,
    UnsupportedOperation,
    ResourceLimit,
}

/// Physical, column-major layer-normalization geometry admitted by the
/// runtime's shared host/device axis plan. Bits in `reduce_mask` identify
/// axes reduced into each independently normalized group.
#[derive(Debug, Clone, PartialEq)]
pub struct DeepLearningLayerNormRequest {
    pub shape: Vec<usize>,
    pub reduce_mask: u8,
    pub channel_axis: usize,
    pub scale_shape: Vec<usize>,
    pub offset_shape: Vec<usize>,
    pub epsilon: f64,
}

/// The output and native scaled statistics from one executed normalization.
/// Backward consumes the same statistics; it must not recompute the forward
/// reduction from a potentially changed input or parameter snapshot.
#[derive(Debug)]
pub struct DeepLearningLayerNormRecorded {
    pub output: GpuTensorHandle,
    pub statistics: GpuTensorHandle,
}

/// Gradients in the physical shapes of the forward input and Scale/Offset.
#[derive(Debug)]
pub struct DeepLearningLayerNormGradients {
    pub input: GpuTensorHandle,
    pub scale: GpuTensorHandle,
    pub offset: GpuTensorHandle,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum DeepLearningLayerNormCapability {
    Ready,
    UnsupportedOperation,
    ResourceLimit,
}

/// A physical stage in a sequential, real-valued resident network. The
/// runtime supplies geometry from its admitted network plan; providers do not
/// reinterpret layer metadata or infer semantic axes here.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum DeepLearningFeedForwardStage {
    Transpose2D {
        rows: usize,
        columns: usize,
    },
    /// Materialize a physical 2-D coordinate permutation, rather than return
    /// a lazy transposed view consumed by a kernel unaware of view strides.
    Permute2D {
        rows: usize,
        columns: usize,
    },
    Dense {
        batch: usize,
        input_channels: usize,
        output_channels: usize,
    },
    Relu {
        elements: usize,
    },
    Elu {
        elements: usize,
    },
    SoftmaxRows {
        rows: usize,
        columns: usize,
    },
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum DeepLearningFeedForwardCapability {
    Ready,
    UnsupportedOperation,
    ResourceLimit,
}
