//! Confusion matrices with exact typed labels.
mod admission;
mod compute;
mod error;
mod input;
mod integer;
mod labels;
mod order;
mod output;
#[cfg(test)]
mod tests;
use crate::builtins::common::tensor;
use crate::{build_runtime_error, gather_if_needed_async, BuiltinResult, RuntimeError};
use admission::*;
use compute::confusionmat_compute;
use error::*;
use input::*;
use integer::*;
use labels::*;
use order::*;
use output::*;
use runmat_builtins::{
    BuiltinErrorDescriptor, CONFUSIONMAT_ERROR_INTERNAL as ERROR_INTERNAL,
    CONFUSIONMAT_ERROR_INVALID_ARGUMENT as ERROR_INVALID_ARGUMENT,
    CONFUSIONMAT_INTEGER_GROUPHAT_EXTENSION, CONFUSIONMAT_INTEGER_GROUP_EXTENSION,
    CONFUSIONMAT_INTEGER_ORDER_EXTENSION, CONFUSIONMAT_RESIDENT_INPUT_EXTENSION,
};
use runmat_macros::runtime_builtin;
use runmat_value::{
    CellArray, CharArray, IntValue, IntegerStorage, LogicalArray, NumericDType, ObjectInstance,
    StringArray, Tensor, Value,
};
use std::{cmp::Ordering, collections::HashMap};
const NAME: &str = "confusionmat";
const MAX_CONFUSIONMAT_CELLS: usize = 25_000_000;
#[runtime_builtin(
    name = "confusionmat",
    binding_variant = "default",
    context = true,
    builtin_path = "crate::builtins::stats::ml::confusionmat"
)]
async fn confusionmat_builtin(
    runtime: Option<crate::context::RuntimeContext>,
    group: Value,
    grouphat: Value,
    rest: crate::call::BuiltinArguments,
) -> BuiltinResult<Value> {
    let requested_outputs = crate::output_count::current_output_count();
    if matches!(requested_outputs, Some(count) if count > 2) {
        return Err(invalid("confusionmat: too many output arguments"));
    }
    ensure_confusionmat_extensions(&group, &grouphat, rest.view())?;

    let group = gather(group).await?;
    let grouphat = gather(grouphat).await?;
    let rest = gather_values(rest, runtime.as_ref()).await?;
    let outputs =
        confusionmat_compute(group, grouphat, rest.view(), requested_outputs.unwrap_or(1))?;
    match requested_outputs {
        Some(0) => Ok(Value::output_list(Vec::new())),
        Some(1) => Ok(Value::output_list(vec![outputs[0].clone()])),
        Some(count) => Ok(crate::output_count::output_list_with_padding(
            count, outputs,
        )),
        None => Ok(outputs[0].clone()),
    }
}
