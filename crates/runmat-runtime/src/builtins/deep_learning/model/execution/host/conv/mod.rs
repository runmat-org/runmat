//! Learned, multichannel 1-D cross-correlation on authoritative host storage.

mod backward;
mod geometry;
mod kernel;

pub(in crate::builtins::deep_learning) use backward::{backward, ConvGradients};
pub(in crate::builtins::deep_learning) use geometry::{
    ConvError, ConvOptions, Padding, PaddingValue,
};
pub(in crate::builtins::deep_learning) use kernel::{forward, ConvOutput};

#[cfg(test)]
mod tests;
