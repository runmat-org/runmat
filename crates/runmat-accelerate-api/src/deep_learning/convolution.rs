//! Physical learned 1-D cross-correlation requests and provider results.

use crate::GpuTensorHandle;

/// Physical column-major geometry of one admitted learned 1-D
/// cross-correlation. The runtime derives this from its canonical layer and
/// axis plan; the provider authenticates the descriptor before any effect.
#[derive(Debug, Clone, PartialEq)]
pub struct DeepLearningConv1DRequest {
    pub input_shape: Vec<usize>,
    pub output_shape: Vec<usize>,
    pub weights_shape: [usize; 3],
    pub bias_shape: [usize; 2],
    pub span_axis: usize,
    pub channel_axis: usize,
    pub batch_axis: Option<usize>,
    pub time_axis: Option<usize>,
    pub stride: usize,
    pub dilation: usize,
    pub padding_left: usize,
    pub padding_right: usize,
    pub padding_value: DeepLearningConv1DPaddingValue,
}

/// Sources available at pure admission. The anchor proves provider ownership
/// and precision even when the future stage input has not been produced.
/// Present operands must be authenticated; absence never grants permission to
/// gather an explicitly resident operand.
#[derive(Clone, Copy)]
pub struct DeepLearningConv1DOperands<'a> {
    pub anchor: &'a GpuTensorHandle,
    pub input: Option<&'a GpuTensorHandle>,
    pub weights: Option<&'a GpuTensorHandle>,
    pub bias: Option<&'a GpuTensorHandle>,
    pub upstream: Option<&'a GpuTensorHandle>,
}

/// Boundary values for the physical convolution window, not signal-conv
/// mode. Symmetric variants differ in whether the edge element repeats.
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum DeepLearningConv1DPaddingValue {
    Constant(f64),
    Replicate,
    SymmetricIncludeEdge,
    SymmetricExcludeEdge,
}

#[derive(Debug)]
pub struct DeepLearningConv1DGradients {
    pub input: GpuTensorHandle,
    pub weights: GpuTensorHandle,
    pub bias: GpuTensorHandle,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum DeepLearningConv1DCapability {
    Ready,
    UnsupportedOperation,
    ResourceLimit,
}
