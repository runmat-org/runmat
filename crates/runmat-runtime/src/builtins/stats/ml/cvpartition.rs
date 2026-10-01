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
    builtin_path = "crate::builtins::stats::ml::cvpartition"
)]
async fn cvpartition_builtin(
    first: Value,
    second: Value,
    rest: Vec<Value>,
) -> BuiltinResult<Value> {
    let admitted = admit_cvpartition(&first, &second, &rest)?;
    let first = gather_value(first).await?;
    let second = gather_value(second).await?;
    let rest = gather_values(rest).await?;
    let spec = cvpartition_compute(first, second, rest, admitted)?;
    partition_object(spec).map(Value::Object)
}

struct AdmittedPartition {
    kind: PartitionKind,
    custom_first: bool,
    option_start: usize,
    stratify: Vec<AdmittedBoolean>,
}

enum AdmittedBoolean {
    Known(bool),
    Resident(usize),
}

fn admit_cvpartition(
    first: &Value,
    second: &Value,
    rest: &[Value],
) -> BuiltinResult<AdmittedPartition> {
    if crate::output_count::current_output_count().is_some_and(|count| count > 1) {
        return Err(invalid_argument("cvpartition: too many output arguments"));
    }
    let first_method = value_partition_method(first);
    let custom_first = first_method == Some(PartitionKind::Custom);
    let kind = if custom_first {
        PartitionKind::Custom
    } else {
        let text = scalar_text(second, "partition kind")?;
        parse_kind(&text)?
    };
    let option_start = match kind {
        PartitionKind::Custom => {
            if !rest.is_empty() {
                return Err(invalid_argument(
                    "cvpartition: CustomPartition accepts exactly one test-set argument",
                ));
            }
            0
        }
        PartitionKind::KFold | PartitionKind::Holdout => {
            if rest.is_empty() {
                return Err(invalid_argument(format!(
                    "cvpartition: {} requires a partition value",
                    kind.property_name()
                )));
            }
            1
        }
        PartitionKind::Leaveout | PartitionKind::Resubstitution => 0,
    };
    let options = &rest[option_start..];
    if !options.len().is_multiple_of(2) {
        return Err(invalid_argument(
            "cvpartition: name-value options must be supplied in pairs",
        ));
    }
    let mut stratify_indices = Vec::new();
    let mut needs_boolean_extension = false;
    for (pair_index, pair) in options.chunks_exact(2).enumerate() {
        let name = scalar_text(&pair[0], "option name")?;
        let option = CvPartitionOption::parse(&name).ok_or_else(|| {
            invalid_argument(format!(
                "cvpartition: unsupported option '{}'",
                canonical(&name)
            ))
        })?;
        match option {
            CvPartitionOption::Stratify => {
                if !documented_logical_scalar(&pair[1]) {
                    needs_boolean_extension = true;
                }
                stratify_indices.push(option_start + pair_index * 2 + 1);
            }
        }
    }
    ensure_cvpartition_extensions(first, second, rest, kind, custom_first)?;
    if needs_boolean_extension {
        enable_extension(&CVPARTITION_BOOLEAN_ALIAS_EXTENSION)?;
    }
    preflight_host_partition(first, second, rest, kind, custom_first)?;
    let mut stratify = Vec::with_capacity(stratify_indices.len());
    for index in stratify_indices {
        stratify.push(if matches!(&rest[index], Value::GpuTensor(_)) {
            AdmittedBoolean::Resident(index)
        } else {
            AdmittedBoolean::Known(scalar_bool(&rest[index], "Stratify")?)
        });
    }
    Ok(AdmittedPartition {
        kind,
        custom_first,
        option_start,
        stratify,
    })
}

fn preflight_host_partition(
    first: &Value,
    second: &Value,
    rest: &[Value],
    kind: PartitionKind,
    custom_first: bool,
) -> BuiltinResult<()> {
    if kind == PartitionKind::Custom {
        let payload = if custom_first { second } else { first };
        if !matches!(
            payload,
            Value::GpuTensor(_) | Value::LogicalArray(_) | Value::Bool(_) | Value::Tensor(_)
        ) {
            return Err(invalid_argument(
                "cvpartition: CustomPartition requires logical or numeric test sets",
            ));
        }
        return Ok(());
    }
    let n = inspect_host_observation_count(first)?;
    match kind {
        PartitionKind::KFold => {
            if !matches!(&rest[0], Value::GpuTensor(_)) {
                let k = positive_integer(&rest[0], "KFold")?;
                if k < 2 || n.is_some_and(|n| k >= n) {
                    return Err(invalid_argument(
                        "cvpartition: KFold must be an integer in the interval [2, n)",
                    ));
                }
                if let Some(n) = n {
                    materialized_mask_len(n, k, "KFold")?;
                }
            }
        }
        PartitionKind::Holdout => {
            if !matches!(&rest[0], Value::GpuTensor(_)) {
                if let Some(n) = n {
                    holdout_count(&rest[0], n)?;
                } else {
                    validate_holdout_control_without_n(&rest[0])?;
                }
            }
        }
        PartitionKind::Leaveout => {
            if let Some(n) = n {
                materialized_mask_len(n, n, "Leaveout")?;
            }
        }
        PartitionKind::Resubstitution | PartitionKind::Custom => {}
    }
    Ok(())
}

fn validate_holdout_control_without_n(value: &Value) -> BuiltinResult<()> {
    if let Some(integer) = tensor::scalar_integer_value(value) {
        if integer.try_to_usize().is_some_and(|count| count > 0) {
            return Ok(());
        }
    } else {
        let raw = scalar_number(value, "Holdout")?;
        if (raw > 0.0 && raw < 1.0) || (raw.is_finite() && raw.fract() == 0.0 && raw >= 1.0) {
            return Ok(());
        }
    }
    Err(invalid_argument(
        "cvpartition: Holdout must be a fraction in (0,1) or an integer in [1,n)",
    ))
}

fn inspect_host_observation_count(value: &Value) -> BuiltinResult<Option<usize>> {
    let vector_len = |shape: &[usize], len: usize| -> BuiltinResult<usize> {
        if len == 0 || shape.iter().filter(|dim| **dim > 1).count() > 1 {
            return Err(invalid_argument(
                "cvpartition: stratification variable must be a nonempty vector",
            ));
        }
        Ok(len)
    };
    let n = match value {
        Value::GpuTensor(_) => return Ok(None),
        Value::Num(number) => positive_integer_number(*number, "n")?,
        Value::Int(integer) => integer.try_to_usize().filter(|n| *n > 0)
            .ok_or_else(|| invalid_argument("cvpartition: n must be a positive integer scalar"))?,
        Value::Tensor(tensor) if tensor::is_scalar_tensor(tensor) => {
            if let Some(integer) = tensor::scalar_integer_value(value) {
                integer.try_to_usize().filter(|n| *n > 0)
                    .ok_or_else(|| invalid_argument("cvpartition: n must be a positive integer scalar"))?
            } else {
                positive_integer_number(tensor::tensor_value_f64(tensor, 0), "n")?
            }
        }
        Value::Tensor(tensor) => vector_len(&tensor.shape, tensor::tensor_element_len(tensor))?,
        Value::LogicalArray(array) => vector_len(&array.shape, array.data.len())?,
        Value::StringArray(array) => vector_len(&array.shape, array.data.len())?,
        Value::CharArray(chars) if chars.rows > 0 => chars.rows,
        Value::Bool(_) | Value::String(_) => 1,
        other => return Err(invalid_argument(format!(
            "cvpartition: first input must be an observation count or grouping variable, got {other:?}"
        ))),
    };
    Ok(Some(n))
}

fn enable_extension(extension: &BuiltinExtensionDescriptor) -> BuiltinResult<()> {
    crate::compatibility::ensure_builtin_extension_enabled(extension, CVPARTITION_NAME)
}

fn typed_integer_value(value: &Value) -> bool {
    matches!(value, Value::Int(_))
        || matches!(value, Value::Tensor(tensor) if tensor.integer_storage().is_some())
        || matches!(value, Value::GpuTensor(handle) if runmat_accelerate_api::handle_integer_type(handle).is_some())
}

fn scalar_value(value: &Value) -> bool {
    matches!(value, Value::Num(_) | Value::Int(_) | Value::Bool(_))
        || matches!(value, Value::Tensor(value_tensor) if tensor::is_scalar_tensor(value_tensor))
        || matches!(value, Value::LogicalArray(array) if array.data.len() == 1)
        || matches!(value, Value::GpuTensor(handle) if tensor::element_count(handle.shape()) == 1)
}

fn documented_logical_scalar(value: &Value) -> bool {
    matches!(value, Value::Bool(_))
        || matches!(value, Value::LogicalArray(array) if array.data.len() == 1)
        || matches!(value, Value::GpuTensor(handle) if runmat_accelerate_api::handle_is_logical(handle) && tensor::element_count(handle.shape()) == 1)
}

fn ensure_cvpartition_extensions(
    first: &Value,
    second: &Value,
    rest: &[Value],
    kind: PartitionKind,
    custom_first: bool,
) -> BuiltinResult<()> {
    if custom_first {
        if typed_integer_value(second) {
            enable_extension(&CVPARTITION_INTEGER_CUSTOM_EXTENSION)?;
        }
        return Ok(());
    }

    if typed_integer_value(first) {
        if scalar_value(first) {
            enable_extension(&CVPARTITION_INTEGER_N_EXTENSION)?;
        } else {
            enable_extension(&CVPARTITION_INTEGER_STRATIFICATION_EXTENSION)?;
        }
    }

    if matches!(kind, PartitionKind::KFold | PartitionKind::Holdout) {
        if rest.first().is_some_and(typed_integer_value) {
            enable_extension(&CVPARTITION_INTEGER_CONTROL_EXTENSION)?;
        }
    }
    Ok(())
}

#[runtime_builtin(
    name = "training",
    binding_variant = "default",
    builtin_path = "crate::builtins::stats::ml::cvpartition"
)]
async fn training_builtin(c: Value, rest: Vec<Value>) -> BuiltinResult<Value> {
    ensure_mask_extensions(&rest, true)?;
    let admitted = admit_mask(&c, &rest)?;
    let output_source = mask_output_source(&rest, "training")?;
    let c = gather_value(c).await?;
    let rest = gather_values(rest).await?;
    let output = mask_from_partition(c, rest, admitted, true)?;
    restore_mask_output(output, output_source.as_ref(), "training")
}

#[runtime_builtin(
    name = "test",
    binding_variant = "default",
    builtin_path = "crate::builtins::stats::ml::cvpartition"
)]
async fn test_builtin(c: Value, rest: Vec<Value>) -> BuiltinResult<Value> {
    ensure_mask_extensions(&rest, false)?;
    let admitted = admit_mask(&c, &rest)?;
    let output_source = mask_output_source(&rest, "test")?;
    let c = gather_value(c).await?;
    let rest = gather_values(rest).await?;
    let output = mask_from_partition(c, rest, admitted, false)?;
    restore_mask_output(output, output_source.as_ref(), "test")
}

fn mask_output_source(
    rest: &[Value],
    builtin: &'static str,
) -> BuiltinResult<Option<GpuTensorHandle>> {
    crate::builtins::common::gpu_helpers::select_resident_output_source(
        rest.iter().filter_map(|value| match value {
            Value::GpuTensor(handle) => Some(handle.clone()),
            _ => None,
        }),
        builtin,
    )
}

fn restore_mask_output(
    output: Value,
    source: Option<&GpuTensorHandle>,
    builtin: &'static str,
) -> BuiltinResult<Value> {
    match source {
        Some(source) => crate::builtins::common::gpu_helpers::restore_class_preserving_value(
            source, output, builtin,
        ),
        None => Ok(output),
    }
}

fn ensure_mask_extensions(rest: &[Value], training: bool) -> BuiltinResult<()> {
    let Some(index) = rest.first() else {
        return Ok(());
    };
    let (integer, logical) = if training {
        (
            &TRAINING_INTEGER_INDEX_EXTENSION,
            &TRAINING_LOGICAL_INDEX_EXTENSION,
        )
    } else {
        (&TEST_INTEGER_INDEX_EXTENSION, &TEST_LOGICAL_INDEX_EXTENSION)
    };
    if typed_integer_value(index) {
        crate::compatibility::ensure_builtin_extension_enabled(
            integer,
            if training { "training" } else { "test" },
        )?;
    }
    if is_logical_index(index) {
        crate::compatibility::ensure_builtin_extension_enabled(
            logical,
            if training { "training" } else { "test" },
        )?;
    }
    Ok(())
}

fn is_logical_index(value: &Value) -> bool {
    matches!(value, Value::Bool(_) | Value::LogicalArray(_))
        || matches!(value, Value::GpuTensor(handle) if runmat_accelerate_api::handle_is_logical(handle))
}

struct AdmittedMask {
    selected_host: Option<Vec<usize>>,
    resubstitution: bool,
}

fn admit_mask(c: &Value, rest: &[Value]) -> BuiltinResult<AdmittedMask> {
    if crate::output_count::current_output_count().is_some_and(|count| count > 1) {
        return Err(invalid_argument(
            "cvpartition: training/test return one output",
        ));
    }
    if rest.len() > 1 {
        return Err(invalid_argument(
            "cvpartition: training/test accept at most one index argument",
        ));
    }
    let object = match c {
        Value::Object(object) if object.is_class(CVPARTITION_CLASS) => object,
        other => {
            return Err(invalid_argument(format!(
                "cvpartition: expected cvpartition object, got {other:?}"
            )))
        }
    };
    let test_sets = match object.properties.get(TEST_SETS_PROPERTY) {
        Some(Value::LogicalArray(array)) => array,
        _ => {
            return Err(invalid_argument(
                "cvpartition: missing internal test-set metadata",
            ))
        }
    };
    let (rows, cols) = logical_shape(test_sets)?;
    if rows.checked_mul(cols) != Some(test_sets.data.len()) {
        return Err(invalid_argument(
            "cvpartition: object test-set metadata is inconsistent",
        ));
    }
    let active_rows = match object.properties.get(ACTIVE_ROWS_PROPERTY) {
        Some(Value::LogicalArray(array)) => array,
        _ => {
            return Err(invalid_argument(
                "cvpartition: missing internal active-row metadata",
            ))
        }
    };
    let (active_count, active_columns) = logical_shape(active_rows)?;
    if active_count != rows || active_columns != 1 || active_rows.data.len() != rows {
        return Err(invalid_argument(
            "cvpartition: object active-row metadata is inconsistent",
        ));
    }
    let kind = object
        .properties
        .get("Type")
        .and_then(value_partition_method)
        .ok_or_else(|| invalid_argument("cvpartition: object partition type is invalid"))?;
    let selected_host =
        match rest.first() {
            Some(Value::GpuTensor(_)) => None,
            Some(index) => Some(selected_indices(index, cols)?),
            None if cols == 1 => Some(vec![0]),
            None => return Err(invalid_argument(
                "cvpartition: an index or 'all' is required for partitions with multiple test sets",
            )),
        };
    if let Some(selected) = &selected_host {
        materialized_mask_len(rows, selected.len(), "training/test output")?;
    }
    Ok(AdmittedMask {
        selected_host,
        resubstitution: kind == PartitionKind::Resubstitution,
    })
}

async fn gather_value(value: Value) -> BuiltinResult<Value> {
    gather_if_needed_async(&value).await
}

async fn gather_values(values: Vec<Value>) -> BuiltinResult<Vec<Value>> {
    let mut out = Vec::with_capacity(values.len());
    for value in values {
        out.push(gather_value(value).await?);
    }
    Ok(out)
}

fn cvpartition_compute(
    first: Value,
    second: Value,
    rest: Vec<Value>,
    admitted: AdmittedPartition,
) -> BuiltinResult<PartitionSpec> {
    if admitted.custom_first {
        return custom_partition(second);
    }

    let kind = admitted.kind;
    let (value, option_start) = match kind {
        PartitionKind::Leaveout | PartitionKind::Resubstitution => (None, 0usize),
        PartitionKind::KFold | PartitionKind::Holdout => {
            let Some(value) = rest.first() else {
                return Err(invalid_argument(format!(
                    "cvpartition: {} requires a partition value",
                    kind.property_name()
                )));
            };
            (Some(value.clone()), 1usize)
        }
        PartitionKind::Custom => {
            if !rest.is_empty() {
                return Err(invalid_argument(
                    "cvpartition: CustomPartition accepts exactly one test-set argument",
                ));
            }
            return custom_partition(first);
        }
    };
    debug_assert_eq!(option_start, admitted.option_start);
    let mut stratify_option = None;
    for option in admitted.stratify {
        stratify_option = Some(match option {
            AdmittedBoolean::Known(value) => value,
            AdmittedBoolean::Resident(index) => scalar_bool(&rest[index], "Stratify")?,
        });
    }
    let input = PartitionInput::from_value(first)?;
    let stratify = stratify_option.unwrap_or(input.default_stratify());
    if stratify && input.labels.is_none() {
        return Err(invalid_argument(
            "cvpartition: Stratify true requires a grouping variable input",
        ));
    }
    let labels = if stratify {
        input.labels.as_deref()
    } else {
        None
    };
    let n = input.n;
    match kind {
        PartitionKind::KFold => {
            let k = positive_integer(value.as_ref().expect("kfold value"), "KFold")?;
            kfold_partition(n, k, labels)
        }
        PartitionKind::Holdout => {
            let test_count = holdout_count(value.as_ref().expect("holdout value"), n)?;
            holdout_partition(n, test_count, labels)
        }
        PartitionKind::Leaveout => leaveout_partition(n),
        PartitionKind::Resubstitution => resubstitution_partition(n),
        PartitionKind::Custom => unreachable!(),
    }
}

#[derive(Clone, Debug)]
struct PartitionInput {
    n: usize,
    labels: Option<Vec<Option<PartitionLabel>>>,
}

impl PartitionInput {
    fn from_value(value: Value) -> BuiltinResult<Self> {
        match value {
            Value::Num(number) => Ok(Self {
                n: positive_integer_number(number, "n")?,
                labels: None,
            }),
            Value::Int(integer) => Ok(Self {
                n: integer.try_to_usize().filter(|n| *n > 0).ok_or_else(|| {
                    invalid_argument("cvpartition: n must be a positive integer scalar")
                })?,
                labels: None,
            }),
            Value::Tensor(tensor) if tensor::is_scalar_tensor(&tensor) => {
                let n = if let Some(integer) = tensor.integer_storage().and_then(|s| s.value_at(0)) {
                    integer.try_to_usize().filter(|n| *n > 0).ok_or_else(|| {
                        invalid_argument("cvpartition: n must be a positive integer scalar")
                    })?
                } else {
                    positive_integer_number(tensor::tensor_value_f64(&tensor, 0), "n")?
                };
                Ok(Self { n, labels: None })
            }
            Value::Tensor(tensor) => numeric_labels(tensor),
            Value::LogicalArray(array) => logical_labels(array),
            Value::Bool(flag) => Ok(Self {
                n: 1,
                labels: Some(vec![Some(PartitionLabel::Logical(flag))]),
            }),
            Value::String(text) => Ok(Self {
                n: 1,
                labels: Some(vec![Some(PartitionLabel::Text(text))]),
            }),
            Value::StringArray(array) => string_array_labels(array),
            Value::CharArray(chars) => Ok(Self {
                n: chars.rows,
                labels: Some(char_row_labels(chars)),
            }),
            other => Err(invalid_argument(format!(
                "cvpartition: first input must be an observation count or grouping variable, got {other:?}"
            ))),
        }
    }

    fn default_stratify(&self) -> bool {
        self.labels.is_some()
    }
}

fn numeric_labels(tensor: Tensor) -> BuiltinResult<PartitionInput> {
    let len = tensor::tensor_element_len(&tensor);
    if len == 0 {
        return Err(invalid_argument(
            "cvpartition: stratification variable must be nonempty",
        ));
    }
    if tensor.shape.iter().filter(|dim| **dim > 1).count() > 1 {
        return Err(invalid_argument(
            "cvpartition: stratification variable must be a vector",
        ));
    }
    let labels = if let Some(storage) = tensor.integer_storage() {
        storage
            .exact_values()
            .iter()
            .map(|value| Some(PartitionLabel::integer(value)))
            .collect()
    } else {
        tensor::tensor_values_f64_cow(&tensor)
            .iter()
            .map(|value| PartitionLabel::number(*value))
            .collect()
    };
    Ok(PartitionInput {
        n: len,
        labels: Some(labels),
    })
}

fn logical_labels(array: LogicalArray) -> BuiltinResult<PartitionInput> {
    if array.data.is_empty() {
        return Err(invalid_argument(
            "cvpartition: stratification variable must be nonempty",
        ));
    }
    if array.shape.iter().filter(|dim| **dim > 1).count() > 1 {
        return Err(invalid_argument(
            "cvpartition: stratification variable must be a vector",
        ));
    }
    Ok(PartitionInput {
        n: array.data.len(),
        labels: Some(
            array
                .data
                .into_iter()
                .map(|flag| Some(PartitionLabel::Logical(flag != 0)))
                .collect(),
        ),
    })
}

fn string_array_labels(array: StringArray) -> BuiltinResult<PartitionInput> {
    if array.data.is_empty() {
        return Err(invalid_argument(
            "cvpartition: stratification variable must be nonempty",
        ));
    }
    if array.shape.iter().filter(|dim| **dim > 1).count() > 1 {
        return Err(invalid_argument(
            "cvpartition: stratification variable must be a vector",
        ));
    }
    Ok(PartitionInput {
        n: array.data.len(),
        labels: Some(
            array
                .data
                .into_iter()
                .map(|text| Some(PartitionLabel::Text(text)))
                .collect(),
        ),
    })
}

fn char_row_labels(chars: runmat_value::CharArray) -> Vec<Option<PartitionLabel>> {
    let mut labels = Vec::with_capacity(chars.rows);
    for row in 0..chars.rows {
        let mut label = String::with_capacity(chars.cols);
        for col in 0..chars.cols {
            label.push(chars.data[row + col * chars.rows]);
        }
        labels.push(Some(PartitionLabel::Text(label)));
    }
    labels
}

fn parse_kind(text: &str) -> BuiltinResult<PartitionKind> {
    PartitionKind::parse(text).ok_or_else(|| {
        invalid_argument(format!(
            "cvpartition: unsupported partition kind '{}'",
            canonical(text)
        ))
    })
}

fn kfold_partition(
    n: usize,
    k: usize,
    labels: Option<&[Option<PartitionLabel>]>,
) -> BuiltinResult<PartitionSpec> {
    if n == 0 {
        return Err(invalid_argument("cvpartition: n must be positive"));
    }
    if k < 2 || k >= n {
        return Err(invalid_argument(
            "cvpartition: KFold must be an integer in the interval [2, n)",
        ));
    }
    let len = materialized_mask_len(n, k, "KFold")?;
    let mut test_sets = vec![0u8; len];
    let active_rows = active_rows_for(n, labels);
    if let Some(labels) = labels {
        if labels.len() != n {
            return Err(invalid_argument(
                "cvpartition: stratification labels must match n",
            ));
        }
        for indices in grouped_indices(labels).values() {
            for (offset, row) in shuffled_indices(indices)?.iter().enumerate() {
                test_sets[row + n * (offset % k)] = 1;
            }
        }
    } else {
        for (offset, row) in shuffled_indices(&(0..n).collect::<Vec<_>>())?
            .iter()
            .enumerate()
        {
            test_sets[row + n * (offset % k)] = 1;
        }
    }
    Ok(PartitionSpec {
        kind: PartitionKind::KFold,
        n,
        test_sets,
        test_set_count: k,
        active_rows,
        is_custom: false,
        is_stratified: labels.is_some(),
    })
}

fn holdout_partition(
    n: usize,
    test_count: usize,
    labels: Option<&[Option<PartitionLabel>]>,
) -> BuiltinResult<PartitionSpec> {
    if n == 0 {
        return Err(invalid_argument("cvpartition: n must be positive"));
    }
    if test_count == 0 || test_count >= n {
        return Err(invalid_argument(
            "cvpartition: Holdout must select between 1 and n-1 observations",
        ));
    }
    let mut test_sets = vec![0u8; n];
    let active_rows = active_rows_for(n, labels);
    if let Some(labels) = labels {
        let groups = grouped_indices(labels);
        let mut selected = 0usize;
        for indices in groups.values() {
            let group_target = ((indices.len() * test_count) + (n / 2)) / n;
            let group_target = group_target.min(indices.len());
            for row in shuffled_indices(indices)?.iter().take(group_target) {
                if selected < test_count {
                    test_sets[*row] = 1;
                    selected += 1;
                }
            }
        }
        for row in shuffled_indices(&(0..n).collect::<Vec<_>>())? {
            if selected >= test_count {
                break;
            }
            if active_rows[row] != 0 && test_sets[row] == 0 {
                test_sets[row] = 1;
                selected += 1;
            }
        }
    } else {
        for row in shuffled_indices(&(0..n).collect::<Vec<_>>())?
            .into_iter()
            .take(test_count)
        {
            test_sets[row] = 1;
        }
    }
    Ok(PartitionSpec {
        kind: PartitionKind::Holdout,
        n,
        test_sets,
        test_set_count: 1,
        active_rows,
        is_custom: false,
        is_stratified: labels.is_some(),
    })
}

fn shuffled_indices(indices: &[usize]) -> BuiltinResult<Vec<usize>> {
    let mut output = indices.to_vec();
    if output.len() <= 1 {
        return Ok(output);
    }
    let uniforms = random::generate_uniform(output.len() - 1, CVPARTITION_NAME)?;
    for (index, uniform) in uniforms.into_iter().enumerate() {
        let span = output.len() - index;
        let offset = ((uniform * span as f64).floor() as usize).min(span - 1);
        output.swap(index, index + offset);
    }
    Ok(output)
}

fn leaveout_partition(n: usize) -> BuiltinResult<PartitionSpec> {
    if n == 0 {
        return Err(invalid_argument("cvpartition: n must be positive"));
    }
    let len = materialized_mask_len(n, n, "Leaveout")?;
    let mut test_sets = vec![0u8; len];
    for row in 0..n {
        test_sets[row + n * row] = 1;
    }
    Ok(PartitionSpec {
        kind: PartitionKind::Leaveout,
        n,
        test_sets,
        test_set_count: n,
        active_rows: vec![1; n],
        is_custom: false,
        is_stratified: false,
    })
}

fn resubstitution_partition(n: usize) -> BuiltinResult<PartitionSpec> {
    if n == 0 {
        return Err(invalid_argument("cvpartition: n must be positive"));
    }
    Ok(PartitionSpec {
        kind: PartitionKind::Resubstitution,
        n,
        test_sets: vec![1u8; n],
        test_set_count: 1,
        active_rows: vec![1; n],
        is_custom: false,
        is_stratified: false,
    })
}

fn custom_partition(value: Value) -> BuiltinResult<PartitionSpec> {
    let array = match value {
        Value::LogicalArray(array) => array,
        Value::Bool(flag) => LogicalArray::new(vec![u8::from(flag)], vec![1, 1])
            .map_err(|err| internal_error(format!("cvpartition: {err}")))?,
        Value::Tensor(tensor) => {
            return custom_numeric_partition(tensor);
        }
        other => {
            return Err(invalid_argument(format!(
                "cvpartition: CustomPartition test sets must be logical, got {other:?}"
            )))
        }
    };
    let shape = logical_shape(&array)?;
    let n = shape.0;
    let test_set_count = shape.1;
    if n == 0 || test_set_count == 0 {
        return Err(invalid_argument(
            "cvpartition: CustomPartition test sets must be nonempty",
        ));
    }
    Ok(PartitionSpec {
        kind: PartitionKind::Custom,
        n,
        test_sets: array.data.into_vec(),
        test_set_count,
        active_rows: vec![1; n],
        is_custom: true,
        is_stratified: false,
    })
}

fn custom_numeric_partition(tensor: Tensor) -> BuiltinResult<PartitionSpec> {
    if tensor.integer_storage().is_some() {
        return custom_integer_partition(tensor);
    }
    let values = tensor::tensor_values_f64_cow(&tensor);
    if values.is_empty() {
        return Err(invalid_argument(
            "cvpartition: CustomPartition test sets must be nonempty",
        ));
    }
    let is_vector = tensor.shape.iter().filter(|dim| **dim > 1).count() <= 1;
    if !is_vector {
        let data = values
            .iter()
            .map(|value| {
                if (*value - 0.0).abs() <= EPS {
                    Ok(0)
                } else if (*value - 1.0).abs() <= EPS {
                    Ok(1)
                } else {
                    Err(invalid_argument(
                        "cvpartition: numeric custom matrices must contain only 0 or 1",
                    ))
                }
            })
            .collect::<BuiltinResult<Vec<_>>>()?;
        let array = LogicalArray::new(data, tensor.shape)
            .map_err(|err| internal_error(format!("cvpartition: {err}")))?;
        return custom_partition(Value::LogicalArray(array));
    }

    if values.iter().any(|value| (*value - 0.0).abs() <= EPS) {
        let data = values
            .iter()
            .map(|value| {
                if (*value - 0.0).abs() <= EPS {
                    Ok(0)
                } else if (*value - 1.0).abs() <= EPS {
                    Ok(1)
                } else {
                    Err(invalid_argument(
                        "cvpartition: numeric custom logical vectors must contain only 0 or 1",
                    ))
                }
            })
            .collect::<BuiltinResult<Vec<_>>>()?;
        let array = LogicalArray::new(data, vec![values.len(), 1])
            .map_err(|err| internal_error(format!("cvpartition: {err}")))?;
        return custom_partition(Value::LogicalArray(array));
    }

    let mut ids = Vec::with_capacity(values.len());
    let mut max_id = 0usize;
    for value in values.iter() {
        if !value.is_finite()
            || value.fract() != 0.0
            || *value < 1.0
            || *value > usize::MAX as f64
            || (usize::BITS == 64 && *value == usize::MAX as f64)
        {
            return Err(invalid_argument(
                "cvpartition: custom partition ids must be positive integers",
            ));
        }
        let id = *value as usize;
        max_id = max_id.max(id);
        ids.push(id);
    }
    let n = ids.len();
    let len = materialized_mask_len(n, max_id, "CustomPartition")?;
    let mut test_sets = vec![0u8; len];
    let mut nonempty = vec![0usize; max_id];
    for (row, id) in ids.into_iter().enumerate() {
        test_sets[row + n * (id - 1)] = 1;
        nonempty[id - 1] += 1;
    }
    if nonempty.contains(&0) {
        return Err(invalid_argument(
            "cvpartition: custom partition ids must form nonempty test sets",
        ));
    }
    Ok(PartitionSpec {
        kind: PartitionKind::Custom,
        n,
        test_sets,
        test_set_count: max_id,
        active_rows: vec![1; n],
        is_custom: true,
        is_stratified: false,
    })
}

fn custom_integer_partition(tensor: Tensor) -> BuiltinResult<PartitionSpec> {
    let storage = tensor
        .integer_storage()
        .expect("custom_integer_partition requires integer storage");
    let values = storage.exact_values();
    if values.is_empty() {
        return Err(invalid_argument(
            "cvpartition: CustomPartition test sets must be nonempty",
        ));
    }
    let is_vector = tensor.shape.iter().filter(|dim| **dim > 1).count() <= 1;
    let logical_byte = |value: &runmat_value::IntValue| match value.try_to_usize() {
        Some(0) => Ok(0),
        Some(1) => Ok(1),
        _ => Err(invalid_argument(
            "cvpartition: numeric custom logical values must contain only 0 or 1",
        )),
    };
    if !is_vector || values.iter().any(|value| value.is_zero()) {
        let data = values
            .iter()
            .map(logical_byte)
            .collect::<BuiltinResult<Vec<_>>>()?;
        let shape = if is_vector {
            vec![values.len(), 1]
        } else {
            tensor.shape.clone()
        };
        let array = LogicalArray::new(data, shape)
            .map_err(|err| internal_error(format!("cvpartition: {err}")))?;
        return custom_partition(Value::LogicalArray(array));
    }

    let mut ids = Vec::with_capacity(values.len());
    let mut max_id = 0usize;
    for value in values {
        let id = value.try_to_usize().filter(|id| *id > 0).ok_or_else(|| {
            invalid_argument("cvpartition: custom partition ids must be positive integers")
        })?;
        max_id = max_id.max(id);
        ids.push(id);
    }
    custom_partition_from_ids(ids, max_id)
}

fn custom_partition_from_ids(ids: Vec<usize>, max_id: usize) -> BuiltinResult<PartitionSpec> {
    let n = ids.len();
    let len = materialized_mask_len(n, max_id, "CustomPartition")?;
    let mut test_sets = vec![0u8; len];
    let mut nonempty = vec![0usize; max_id];
    for (row, id) in ids.into_iter().enumerate() {
        test_sets[row + n * (id - 1)] = 1;
        nonempty[id - 1] += 1;
    }
    if nonempty.contains(&0) {
        return Err(invalid_argument(
            "cvpartition: custom partition ids must form nonempty test sets",
        ));
    }
    Ok(PartitionSpec {
        kind: PartitionKind::Custom,
        n,
        test_sets,
        test_set_count: max_id,
        active_rows: vec![1; n],
        is_custom: true,
        is_stratified: false,
    })
}

fn partition_object(spec: PartitionSpec) -> BuiltinResult<ObjectInstance> {
    let mut object = ObjectInstance::new(CVPARTITION_CLASS.to_string());
    object
        .properties
        .insert("NumObservations".to_string(), Value::Num(spec.n as f64));
    object.properties.insert(
        "NumTestSets".to_string(),
        Value::Num(spec.test_set_count as f64),
    );
    object.properties.insert(
        "TestSize".to_string(),
        size_vector(&spec.test_sets, spec.n, spec.test_set_count, true)?,
    );
    object.properties.insert(
        "TrainSize".to_string(),
        size_vector(&spec.test_sets, spec.n, spec.test_set_count, false)?,
    );
    object.properties.insert(
        "Type".to_string(),
        Value::String(spec.kind.property_name().to_string()),
    );
    object
        .properties
        .insert("IsCustom".to_string(), Value::Bool(spec.is_custom));
    object
        .properties
        .insert("IsGrouped".to_string(), Value::Bool(false));
    object
        .properties
        .insert("IsStratified".to_string(), Value::Bool(spec.is_stratified));
    object.properties.insert(
        TEST_SETS_PROPERTY.to_string(),
        Value::LogicalArray(
            LogicalArray::new(spec.test_sets, vec![spec.n, spec.test_set_count])
                .map_err(|err| internal_error(format!("cvpartition: {err}")))?,
        ),
    );
    object.properties.insert(
        ACTIVE_ROWS_PROPERTY.to_string(),
        Value::LogicalArray(
            LogicalArray::new(spec.active_rows, vec![spec.n, 1])
                .map_err(|err| internal_error(format!("cvpartition: {err}")))?,
        ),
    );
    Ok(object)
}

fn size_vector(
    test_sets: &[u8],
    rows: usize,
    cols: usize,
    test_size: bool,
) -> BuiltinResult<Value> {
    let mut data = Vec::with_capacity(cols);
    for col in 0..cols {
        let count = test_sets[col * rows..(col + 1) * rows]
            .iter()
            .filter(|flag| **flag != 0)
            .count();
        let size = if test_size { count } else { rows - count };
        data.push(size as f64);
    }
    if cols == 1 {
        Ok(Value::Num(data[0]))
    } else {
        Tensor::new(data, vec![1, cols])
            .map(Value::Tensor)
            .map_err(|err| internal_error(format!("cvpartition: {err}")))
    }
}

fn mask_from_partition(
    c: Value,
    rest: Vec<Value>,
    admitted: AdmittedMask,
    training: bool,
) -> BuiltinResult<Value> {
    if rest.len() > 1 {
        return Err(invalid_argument(
            "cvpartition: training/test accept at most one index argument",
        ));
    }
    let object = match c {
        Value::Object(object) if object.is_class(CVPARTITION_CLASS) => object,
        other => {
            return Err(invalid_argument(format!(
                "cvpartition: expected cvpartition object, got {other:?}"
            )))
        }
    };
    let test_sets = test_sets_property(&object)?;
    let active_rows = active_rows_property(&object)?;
    let resubstitution = admitted.resubstitution;
    let (rows, cols) = logical_shape(&test_sets)?;
    if active_rows.data.len() != rows {
        return Err(invalid_argument(
            "cvpartition: object active-row metadata is inconsistent",
        ));
    }
    let selected = if let Some(selected) = admitted.selected_host {
        selected
    } else if let Some(index) = rest.first() {
        selected_indices(index, cols)?
    } else if cols == 1 {
        vec![0usize]
    } else {
        return Err(invalid_argument(
            "cvpartition: an index or 'all' is required for partitions with multiple test sets",
        ));
    };
    let _ = materialized_mask_len(rows, selected.len(), "training/test output")?;
    let mut data = Vec::with_capacity(rows * selected.len());
    for col in &selected {
        let base = *col * rows;
        for row in 0..rows {
            let active = active_rows.data[row] != 0;
            let is_test = test_sets.data[base + row] != 0;
            data.push(u8::from(if training {
                active && (resubstitution || !is_test)
            } else {
                active && is_test
            }));
        }
    }
    LogicalArray::new(
        data,
        vec![rows, if rest.is_empty() { 1 } else { selected.len() }],
    )
    .map(Value::LogicalArray)
    .map_err(|err| internal_error(format!("cvpartition: {err}")))
}

fn selected_indices(value: &Value, cols: usize) -> BuiltinResult<Vec<usize>> {
    if scalar_text(value, "index")
        .ok()
        .and_then(|text| MaskSelectionKeyword::parse(&text))
        == Some(MaskSelectionKeyword::All)
    {
        return Ok((0..cols).collect());
    }
    if let Value::Int(integer) = value {
        return integer
            .try_to_usize()
            .filter(|index| *index >= 1 && *index <= cols)
            .map(|index| vec![index - 1])
            .ok_or_else(|| {
                invalid_argument(format!(
                    "cvpartition: index must be an integer between 1 and {cols}"
                ))
            });
    }
    if let Value::Tensor(tensor) = value {
        if let Some(storage) = tensor.integer_storage() {
            if storage.is_empty() {
                return Err(invalid_argument("cvpartition: index must be nonempty"));
            }
            return storage
                .exact_values()
                .iter()
                .map(|value| {
                    value
                        .try_to_usize()
                        .filter(|index| *index >= 1 && *index <= cols)
                        .map(|index| index - 1)
                        .ok_or_else(|| {
                            invalid_argument(format!(
                                "cvpartition: index must be an integer between 1 and {cols}"
                            ))
                        })
                })
                .collect();
        }
    }
    let raw = match value {
        Value::Num(number) => vec![*number],
        Value::Tensor(tensor) => tensor::tensor_values_f64(tensor),
        Value::LogicalArray(array) => array
            .data
            .iter()
            .map(|flag| if *flag == 0 { 0.0 } else { 1.0 })
            .collect(),
        other => {
            return Err(invalid_argument(format!(
                "cvpartition: index must be numeric, logical, or 'all', got {other:?}"
            )))
        }
    };
    let mut indices = Vec::with_capacity(raw.len());
    for value in raw {
        if !value.is_finite() || value.fract() != 0.0 || value < 1.0 || value > cols as f64 {
            return Err(invalid_argument(format!(
                "cvpartition: index must be an integer between 1 and {cols}"
            )));
        }
        indices.push(value as usize - 1);
    }
    if indices.is_empty() {
        return Err(invalid_argument("cvpartition: index must be nonempty"));
    }
    Ok(indices)
}

fn test_sets_property(object: &ObjectInstance) -> BuiltinResult<LogicalArray> {
    match object.properties.get(TEST_SETS_PROPERTY) {
        Some(Value::LogicalArray(array)) => Ok(array.clone()),
        _ => Err(invalid_argument(
            "cvpartition: object is missing internal test-set data",
        )),
    }
}

fn active_rows_property(object: &ObjectInstance) -> BuiltinResult<LogicalArray> {
    match object.properties.get(ACTIVE_ROWS_PROPERTY) {
        Some(Value::LogicalArray(array)) => Ok(array.clone()),
        _ => Err(invalid_argument(
            "cvpartition: object is missing internal active-row data",
        )),
    }
}

fn logical_shape(array: &LogicalArray) -> BuiltinResult<(usize, usize)> {
    match array.shape.as_slice() {
        [] => Ok((array.data.len(), 1)),
        [rows] => Ok((*rows, 1)),
        [1, cols] => Ok((*cols, 1)),
        [rows, 1] => Ok((*rows, 1)),
        [rows, cols] => Ok((*rows, *cols)),
        _ => Err(invalid_argument(
            "cvpartition: logical test sets must be a vector or 2-D matrix",
        )),
    }
}

fn grouped_indices(labels: &[Option<PartitionLabel>]) -> BTreeMap<PartitionLabel, Vec<usize>> {
    let mut groups = BTreeMap::<PartitionLabel, Vec<usize>>::new();
    for (idx, label) in labels.iter().enumerate() {
        if let Some(label) = label {
            groups.entry(label.clone()).or_default().push(idx);
        }
    }
    groups
}

fn active_rows_for(n: usize, labels: Option<&[Option<PartitionLabel>]>) -> Vec<u8> {
    match labels {
        Some(labels) => labels
            .iter()
            .map(|label| u8::from(label.is_some()))
            .collect(),
        None => vec![1; n],
    }
}

fn materialized_mask_len(rows: usize, cols: usize, label: &str) -> BuiltinResult<usize> {
    let len = rows
        .checked_mul(cols)
        .ok_or_else(|| internal_error(format!("cvpartition: {label} size overflow")))?;
    if len > MAX_MATERIALIZED_MASK_CELLS {
        return Err(internal_error(format!(
            "cvpartition: {label} would materialize {len} logical entries; limit is {MAX_MATERIALIZED_MASK_CELLS}"
        )));
    }
    Ok(len)
}

fn holdout_count(value: &Value, n: usize) -> BuiltinResult<usize> {
    if n < 2 {
        return Err(invalid_argument(
            "cvpartition: Holdout requires at least two observations",
        ));
    }
    if let Some(integer) = tensor::scalar_integer_value(value) {
        return integer
            .try_to_usize()
            .filter(|count| *count > 0 && *count < n)
            .ok_or_else(|| {
                invalid_argument(
                    "cvpartition: Holdout must be a fraction in (0,1) or an integer in [1,n)",
                )
            });
    }
    let raw = scalar_number(value, "Holdout")?;
    if raw > 0.0 && raw < 1.0 {
        let count = (raw * n as f64).round() as usize;
        return Ok(count.clamp(1, n.saturating_sub(1)));
    }
    if raw.fract() == 0.0 && raw >= 1.0 && raw < n as f64 {
        return Ok(raw as usize);
    }
    Err(invalid_argument(
        "cvpartition: Holdout must be a fraction in (0,1) or an integer in [1,n)",
    ))
}

fn positive_integer(value: &Value, label: &str) -> BuiltinResult<usize> {
    if let Some(integer) = tensor::scalar_integer_value(value) {
        return integer
            .try_to_usize()
            .filter(|value| *value > 0)
            .ok_or_else(|| {
                invalid_argument(format!(
                    "cvpartition: {label} must be a positive integer scalar"
                ))
            });
    }
    positive_integer_number(scalar_number(value, label)?, label)
}

fn positive_integer_number(raw: f64, label: &str) -> BuiltinResult<usize> {
    PartitionCountGrammar::CvPartition
        .positive_from_float(raw)
        .ok_or_else(|| {
            invalid_argument(format!(
                "cvpartition: {label} must be a positive integer scalar"
            ))
        })
}

fn scalar_number(value: &Value, label: &str) -> BuiltinResult<f64> {
    let number = match value {
        Value::Num(number) => *number,
        Value::Int(integer) => integer.to_f64(),
        Value::Bool(flag) => {
            if *flag {
                1.0
            } else {
                0.0
            }
        }
        Value::Tensor(tensor) if tensor::is_scalar_tensor(tensor) => {
            tensor::tensor_value_f64(tensor, 0)
        }
        Value::LogicalArray(array) if array.data.len() == 1 => {
            if array.data[0] == 0 {
                0.0
            } else {
                1.0
            }
        }
        other => {
            return Err(invalid_argument(format!(
                "cvpartition: {label} must be a numeric scalar, got {other:?}"
            )))
        }
    };
    if !number.is_finite() {
        return Err(invalid_argument(format!(
            "cvpartition: {label} must be finite"
        )));
    }
    Ok(number)
}

fn scalar_bool(value: &Value, label: &str) -> BuiltinResult<bool> {
    match value {
        Value::Bool(flag) => Ok(*flag),
        Value::LogicalArray(array) if array.data.len() == 1 => Ok(array.data[0] != 0),
        Value::Num(number) => parse_stratify_numeric_boolean(*number).ok_or_else(|| {
            invalid_argument(format!("cvpartition: {label} must be logical scalar"))
        }),
        Value::Tensor(tensor) if tensor::is_scalar_tensor(tensor) => {
            if let Some(integer) = tensor
                .integer_storage()
                .and_then(|storage| storage.value_at(0))
            {
                match integer.try_to_usize() {
                    Some(0) => Ok(false),
                    Some(1) => Ok(true),
                    _ => Err(invalid_argument(format!(
                        "cvpartition: {label} must be logical scalar"
                    ))),
                }
            } else {
                parse_stratify_numeric_boolean(tensor::tensor_value_f64(tensor, 0)).ok_or_else(
                    || invalid_argument(format!("cvpartition: {label} must be logical scalar")),
                )
            }
        }
        Value::Int(integer) => match integer.try_to_usize() {
            Some(0) => Ok(false),
            Some(1) => Ok(true),
            _ => Err(invalid_argument(format!(
                "cvpartition: {label} must be logical scalar"
            ))),
        },
        Value::String(text) => parse_stratify_boolean_alias(text).ok_or_else(|| {
            invalid_argument(format!("cvpartition: {label} must be logical scalar"))
        }),
        Value::CharArray(chars) => {
            let text = chars.data.iter().collect::<String>();
            parse_stratify_boolean_alias(&text).ok_or_else(|| {
                invalid_argument(format!("cvpartition: {label} must be logical scalar"))
            })
        }
        _ => Err(invalid_argument(format!(
            "cvpartition: {label} must be logical scalar"
        ))),
    }
}

fn scalar_text(value: &Value, label: &str) -> BuiltinResult<String> {
    match value {
        Value::String(text) => Ok(text.clone()),
        Value::CharArray(chars) => Ok(chars.data.iter().collect()),
        Value::StringArray(array) if array.data.len() == 1 => Ok(array.data[0].clone()),
        other => Err(invalid_argument(format!(
            "cvpartition: {label} must be a string scalar, got {other:?}"
        ))),
    }
}

fn value_partition_method(value: &Value) -> Option<PartitionKind> {
    scalar_text(value, "partition kind")
        .ok()
        .and_then(|text| PartitionKind::parse(&text))
}

fn canonical(text: &str) -> String {
    text.chars()
        .filter(|ch| *ch != '_' && *ch != '-' && !ch.is_whitespace())
        .collect::<String>()
        .to_ascii_lowercase()
}

#[cfg(test)]
mod tests;
