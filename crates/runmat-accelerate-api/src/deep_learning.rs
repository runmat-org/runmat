//! Typed acceleration requests for deep-learning operations.

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
