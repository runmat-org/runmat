//! Cross-validation partition helpers for Statistics and Machine Learning workflows.

use std::collections::BTreeMap;

use runmat_accelerate_api::GpuTensorHandle;
#[cfg(test)]
use runmat_builtins::CVPARTITION_INTEGER_CAPABILITIES;
use runmat_builtins::{
    parse_stratify_boolean_alias, parse_stratify_numeric_boolean, BuiltinErrorDescriptor,
    BuiltinExtensionDescriptor, CvPartitionMethod as PartitionKind, CvPartitionOption,
    MaskSelectionKeyword, PartitionCountGrammar, CVPARTITION_BOOLEAN_ALIAS_EXTENSION,
    CVPARTITION_ERROR_INTERNAL, CVPARTITION_ERROR_INVALID_ARGUMENT,
    CVPARTITION_INTEGER_CONTROL_EXTENSION, CVPARTITION_INTEGER_CUSTOM_EXTENSION,
    CVPARTITION_INTEGER_N_EXTENSION, CVPARTITION_INTEGER_STRATIFICATION_EXTENSION,
    TEST_INTEGER_INDEX_EXTENSION, TEST_LOGICAL_INDEX_EXTENSION, TRAINING_INTEGER_INDEX_EXTENSION,
    TRAINING_LOGICAL_INDEX_EXTENSION,
};
use runmat_macros::runtime_builtin;
use runmat_value::{LogicalArray, ObjectInstance, StringArray, Tensor, Value};

use super::partition_shared::PartitionLabel;
use crate::builtins::common::{random, tensor};
use crate::{build_runtime_error, gather_if_needed_async, BuiltinResult, RuntimeError};

mod admission;
mod controls;
mod input;
mod mask;
mod object;
mod partition;

use admission::*;
use controls::*;
use input::*;
use mask::*;
use object::*;
use partition::*;

const CVPARTITION_NAME: &str = "cvpartition";
const CVPARTITION_CLASS: runmat_types::StaticClassIdentity = runmat_types::standard::CVPARTITION;
const TEST_SETS_PROPERTY: &str = "__RunMatCvPartitionTestSets";
const ACTIVE_ROWS_PROPERTY: &str = "__RunMatCvPartitionActiveRows";
const MAX_MATERIALIZED_MASK_CELLS: usize = 10_000_000;
const EPS: f64 = 1.0e-12;

fn cvpartition_error(
    message: impl Into<String>,
    descriptor: &'static BuiltinErrorDescriptor,
) -> RuntimeError {
    let mut builder = build_runtime_error(message).with_builtin(CVPARTITION_NAME);
    if let Some(identifier) = descriptor.identifier {
        builder = builder.with_identifier(identifier);
    }
    builder.build()
}

fn invalid_argument(message: impl Into<String>) -> RuntimeError {
    cvpartition_error(message, &CVPARTITION_ERROR_INVALID_ARGUMENT)
}

fn internal_error(message: impl Into<String>) -> RuntimeError {
    cvpartition_error(message, &CVPARTITION_ERROR_INTERNAL)
}

#[derive(Clone, Debug)]
struct PartitionSpec {
    kind: PartitionKind,
    n: usize,
    test_sets: Vec<u8>,
    test_set_count: usize,
    active_rows: Vec<u8>,
    is_custom: bool,
    is_stratified: bool,
}

#[runtime_builtin(
    name = "cvpartition",
    binding_variant = "default",
    context = true,
    builtin_path = "crate::builtins::stats::ml::cvpartition"
)]
async fn cvpartition_builtin(
    runtime: Option<crate::context::RuntimeContext>,
    first: Value,
    second: Value,
    rest: crate::call::BuiltinArguments,
) -> BuiltinResult<Value> {
    let admitted = admit_cvpartition(&first, &second, rest.view())?;
    let first = gather_value(first).await?;
    let second = gather_value(second).await?;
    let rest = gather_values(rest, runtime.as_ref()).await?;
    let mut turn = if matches!(admitted.kind, PartitionKind::KFold | PartitionKind::Holdout) {
        Some(random::acquire_turn(runtime.as_ref(), CVPARTITION_NAME).await?)
    } else {
        None
    };
    let spec = cvpartition_compute(first, second, rest.view(), admitted, turn.as_mut())?;
    partition_object(spec).map(Value::Object)
}

#[runtime_builtin(
    name = "training",
    binding_variant = "default",
    context = true,
    builtin_path = "crate::builtins::stats::ml::cvpartition"
)]
async fn training_builtin(
    runtime: Option<crate::context::RuntimeContext>,
    c: Value,
    rest: crate::call::BuiltinArguments,
) -> BuiltinResult<Value> {
    ensure_mask_extensions(rest.view(), true)?;
    let admitted = admit_mask(&c, rest.view())?;
    let output_source = mask_output_source(rest.view(), "training")?;
    let c = gather_value(c).await?;
    let rest = gather_values(rest, runtime.as_ref()).await?;
    let output = mask_from_partition(c, rest.view(), admitted, true)?;
    restore_mask_output(output, output_source.as_ref(), "training")
}

#[runtime_builtin(
    name = "test",
    binding_variant = "default",
    context = true,
    builtin_path = "crate::builtins::stats::ml::cvpartition"
)]
async fn test_builtin(
    runtime: Option<crate::context::RuntimeContext>,
    c: Value,
    rest: crate::call::BuiltinArguments,
) -> BuiltinResult<Value> {
    ensure_mask_extensions(rest.view(), false)?;
    let admitted = admit_mask(&c, rest.view())?;
    let output_source = mask_output_source(rest.view(), "test")?;
    let c = gather_value(c).await?;
    let rest = gather_values(rest, runtime.as_ref()).await?;
    let output = mask_from_partition(c, rest.view(), admitted, false)?;
    restore_mask_output(output, output_source.as_ref(), "test")
}

async fn gather_value(value: Value) -> BuiltinResult<Value> {
    gather_if_needed_async(&value).await
}

async fn gather_values(
    values: crate::call::BuiltinArguments,
    runtime: Option<&crate::context::RuntimeContext>,
) -> BuiltinResult<crate::call::BuiltinArguments> {
    crate::dispatcher::gather_builtin_arguments_async(values, runtime).await
}

#[cfg(test)]
mod tests;
