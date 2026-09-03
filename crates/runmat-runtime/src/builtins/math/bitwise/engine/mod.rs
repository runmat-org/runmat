//! Shared exact mechanics for integer bit operations.

use runmat_builtins::{BinaryBitwiseOperator, BuiltinErrorDescriptor, BuiltinExtensionDescriptor};
use runmat_types::IntegerClass;
use runmat_value::{IntValue, IntegerStorage, LogicalArray, NumericDType, Tensor, Value};

use crate::builtins::common::broadcast::BroadcastPlan;
use crate::builtins::common::integer_value::{value_from_exact_integers, IntegerClassValueExt};
use crate::builtins::common::random_args::keyword_of;
use crate::builtins::common::{gpu_helpers, tensor};
use crate::builtins::math::elementwise::sparse::{
    checked_sparse_result_len, map_sparse_real_values,
};
use crate::{build_runtime_error, BuiltinResult, RuntimeError};

const BITCMP_NAME: &str = "bitcmp";
const BITGET_NAME: &str = "bitget";
const BITSET_NAME: &str = "bitset";
const BITSHIFT_NAME: &str = "bitshift";
#[cfg(test)]
const BITXOR_NAME: &str = "bitxor";
const ERROR_INVALID_INPUT: BuiltinErrorDescriptor = runmat_builtins::BITWISE_ERROR_INVALID_INPUT;
const ERROR_SIZE_MISMATCH: BuiltinErrorDescriptor = runmat_builtins::BITWISE_ERROR_SIZE_MISMATCH;

mod arguments;
mod binary;
mod error;
mod operand;
mod output;
mod position;
mod resident;
mod shift;
mod sparse;

use arguments::*;
use error::*;
use operand::*;
use output::*;
use resident::*;
use sparse::*;

pub(crate) use binary::evaluate_binary_bitwise;
pub(crate) use position::{evaluate_bitcmp, evaluate_bitget, evaluate_bitset};
pub(crate) use shift::evaluate_bitshift;

#[cfg(test)]
use super::binary::{bitand_builtin, bitor_builtin, bitxor_builtin};
#[cfg(test)]
use super::{
    bitcmp::bitcmp_builtin, bitget::bitget_builtin, bitset::bitset_builtin,
    bitshift::bitshift_builtin,
};
#[cfg(test)]
mod tests;
