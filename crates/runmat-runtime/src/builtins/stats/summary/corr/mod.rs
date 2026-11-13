//! Pairwise linear and rank correlations.
mod arguments;
mod columns;
mod compute;
mod error;
mod pvalue;
mod statistics;
#[cfg(test)]
mod tests;
use crate::builtins::common::random_args::keyword_of;
use crate::builtins::common::tensor;
use crate::builtins::stats::summary::distribution_math::{standard_normal_cdf, student_t_cdf};
use crate::{build_runtime_error, gather_if_needed_async, BuiltinResult, RuntimeError};
use arguments::*;
use columns::CorrColumn;
use compute::corr_compute;
use error::*;
use pvalue::correlation_pvalue;
use runmat_builtins::{
    BuiltinErrorDescriptor, CORR_ERROR_INTERNAL as ERROR_INTERNAL,
    CORR_ERROR_INVALID_ARGUMENT as ERROR_INVALID_ARGUMENT, CORR_INTEGER_DATA_EXTENSION,
    CORR_INTEGER_WEIGHTS_EXTENSION,
};
use runmat_macros::runtime_builtin;
use runmat_value::{IntValue, Tensor, Value};
use statistics::corr_pair_masked;
use std::cmp::Ordering;
const NAME: &str = "corr";
#[runtime_builtin(
    name = "corr",
    binding_variant = "default",
    builtin_path = "crate::builtins::stats::summary::corr"
)]
async fn corr_builtin(value: Value, rest: crate::call::BuiltinArguments) -> BuiltinResult<Value> {
    ensure_corr_integer_extensions(&value, rest.view())?;
    let args = parse_args(value, rest).await?;
    corr_compute(args)
}
