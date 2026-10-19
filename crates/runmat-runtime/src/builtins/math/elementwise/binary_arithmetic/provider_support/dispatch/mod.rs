mod common;
mod pair;
mod resident;

pub(in crate::builtins::math::elementwise::binary_arithmetic) use common::{
    common_owner, compatible_inputs, float_handle_supported,
};

#[cfg(test)]
mod tests;

pub(in crate::builtins::math::elementwise::binary_arithmetic) use pair::try_pair;
pub(in crate::builtins::math::elementwise::binary_arithmetic) use resident::{
    try_host_left, try_host_right,
};
