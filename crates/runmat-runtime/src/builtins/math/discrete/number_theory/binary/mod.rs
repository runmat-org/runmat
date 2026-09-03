//! Shared exact mechanics for `gcd` and `lcm`.

mod arithmetic;
mod class;
mod context;
mod input;
mod output;
mod plan;

pub(in crate::builtins::math::discrete) use arithmetic::{extended_gcd, gcd, lcm};
pub(in crate::builtins::math::discrete) use class::{resolve_output, BinaryOutput};
pub(in crate::builtins::math::discrete) use context::{
    binary_error, BinaryContext, GCD_CONTEXT, LCM_CONTEXT,
};
pub(in crate::builtins::math::discrete) use input::BinaryInput;
pub(in crate::builtins::math::discrete) use output::{coefficient_value, magnitude_value};
pub(in crate::builtins::math::discrete) use plan::SameSizeOrScalarPlan;
