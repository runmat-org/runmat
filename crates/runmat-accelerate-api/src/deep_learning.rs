//! Typed acceleration requests for deep-learning operations.

/// A resident global-average-pooling reduction over one physical axis.
/// The provider retains that axis with extent one; presentation may remove
/// it later without copying device bytes. Unlike a general mean reduction,
/// finite inputs whose raw sum overflows must still produce a finite mean.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct DeepLearningPool1DRequest {
    pub axis_zero_based: usize,
}
