mod common;
mod pair;
mod resident;

#[cfg(test)]
mod tests;

pub(in crate::builtins::math::elementwise::binary_arithmetic) use pair::try_pair;
pub(in crate::builtins::math::elementwise::binary_arithmetic) use resident::{
    try_host_left, try_host_right,
};
