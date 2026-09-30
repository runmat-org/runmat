//! Layer normalization with native-precision statistics and adjoints.

mod geometry;
mod kernel;
mod backward;

pub(in crate::builtins::deep_learning) use backward::{backward, NormalizationGradients};
pub(in crate::builtins::deep_learning) use geometry::{NormalizationError, NormalizationOptions};
pub(in crate::builtins::deep_learning) use kernel::{
    forward, NormalizationOutput, NormalizationStats,
};

#[cfg(test)]
mod tests;
