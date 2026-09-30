//! Native-precision recurrent host kernels. The network owner supplies admitted
//! options and canonical parameters/state; this module never mutates a network.

mod backward;
mod forward;
mod gates;
mod geometry;

pub(in crate::builtins::deep_learning) use backward::{backward, RecurrentGradients};
pub(in crate::builtins::deep_learning) use forward::{
    forward, RecurrentBackwardCache, RecurrentOutput,
};
pub(in crate::builtins::deep_learning) use geometry::{
    Activation, GateActivation, OutputMode, RecurrentError, RecurrentOptions,
};

fn required_slice<'a, T>(
    slice: Option<Option<&'a [T]>>,
) -> Result<Option<&'a [T]>, RecurrentError> {
    slice
        .map(|value| value.ok_or(RecurrentError::UnsupportedPrecision))
        .transpose()
}

#[cfg(test)]
mod tests;
