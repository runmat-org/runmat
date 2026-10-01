//! Dummy-variable and one-hot encoding helpers.

use runmat_value::{NumericDType, NumericScalar, NumericStorage};
use std::{cmp::Ordering, collections::HashMap};

use runmat_builtins::{
    EncodingOptionName, EncodingOutputKind as OutputKind, DUMMYVAR_GPU_GROUP_EXTENSION,
    DUMMYVAR_INTEGER_GROUP_EXTENSION, ONEHOTDECODE_BROAD_ENCODED_EXTENSION,
    ONEHOTDECODE_EXPLICIT_GPU_EXTENSION, ONEHOTDECODE_LOGICAL_CELL_OUTPUT_EXTENSION,
    ONEHOTDECODE_OUTPUT_TYPE_ALIAS_EXTENSION, ONEHOTENCODE_EXPLICIT_GPU_EXTENSION,
    ONEHOTENCODE_OUTPUT_TYPE_ALIAS_EXTENSION,
};
use runmat_macros::runtime_builtin;
use runmat_types::NumericClass;
use runmat_value::{
    CellArray, CharArray, IntValue, LogicalArray, ObjectInstance, StringArray, Tensor, Value,
};

use crate::builtins::common::tensor;
use crate::{build_runtime_error, gather_if_needed_async, BuiltinResult, RuntimeError};

const MAX_ONEHOT_CELLS: usize = 50_000_000;
const MAX_ENCODING_RANK: usize = 32;

enum Admitted<T> {
    Ready(T),
    Deferred(Value),
}

#[path = "encoding/controls.rs"]
mod controls;
use controls::AdmittedControls;
#[path = "encoding/geometry.rs"]
mod geometry;
#[path = "encoding/labels.rs"]
mod labels;
#[path = "encoding/output.rs"]
mod output;
use self::{geometry::*, labels::*, output::*};

fn error(name: &str, message: impl Into<String>) -> RuntimeError {
    build_runtime_error(message).with_builtin(name).build()
}

fn invalid(name: &str, message: impl Into<String>) -> RuntimeError {
    let mut builder = build_runtime_error(message).with_builtin(name);
    // The catalog descriptor owns the public identifier. The encoding
    // descriptors declare invalid-argument first and internal-error second.
    let identifier = runmat_builtins::builtin_descriptor_by_name(name)
        .and_then(|descriptor| descriptor.errors.first())
        .and_then(|error| error.identifier)
        .unwrap_or("RunMat:encoding:InvalidArgument");
    builder = builder.with_identifier(identifier);
    builder.build()
}

fn internal(name: &str, message: impl Into<String>) -> RuntimeError {
    let mut builder = build_runtime_error(message).with_builtin(name);
    let identifier = runmat_builtins::builtin_descriptor_by_name(name)
        .and_then(|descriptor| descriptor.errors.get(1))
        .and_then(|error| error.identifier)
        .unwrap_or("RunMat:encoding:Internal");
    builder = builder.with_identifier(identifier);
    builder.build()
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
enum LabelKind {
    Numeric,
    Logical,
    String,
    Char,
    Cell,
    Categorical,
}

#[derive(Clone, Debug)]
struct LabelArray {
    labels: Vec<Option<Label>>,
    shape: Vec<usize>,
    kind: LabelKind,
    categories: Option<Vec<String>>,
}

#[derive(Clone, Debug, PartialEq)]
enum Label {
    Numeric(NumericScalar),
    Logical(bool),
    Text(String),
}

/// Exact class identity shared by encoding, decoding, and duplicate admission.
/// Text is presentation data only for text labels, never a numeric dispatch key.
#[derive(Clone, Debug, Eq, Hash, PartialEq)]
enum LabelKey {
    Integer(i128),
    FloatingBits(u64),
    Logical(bool),
    Text(String),
}

#[runtime_builtin(
    name = "dummyvar",
    binding_variant = "default",
    builtin_path = "crate::builtins::stats::summary::encoding"
)]
async fn dummyvar_builtin(group: Value) -> BuiltinResult<Value> {
    if value_contains_typed_integer(&group) {
        crate::compatibility::ensure_builtin_extension_enabled(
            &DUMMYVAR_INTEGER_GROUP_EXTENSION,
            "dummyvar",
        )?;
    }
    if crate::value_contains_gpu(&group) {
        crate::compatibility::ensure_builtin_extension_enabled(
            &DUMMYVAR_GPU_GROUP_EXTENSION,
            "dummyvar",
        )?;
    }
    let group = gather("dummyvar", group).await?;
    validate_dummyvar_integer_levels(&group)?;
    dummyvar_compute(group)
}

fn value_contains_typed_integer(value: &Value) -> bool {
    matches!(value, Value::Int(_))
        || matches!(value, Value::Tensor(tensor) if tensor.integer_storage().is_some())
        || matches!(value, Value::GpuTensor(handle) if runmat_accelerate_api::handle_integer_type(handle).is_some())
        || matches!(value, Value::Cell(cell) if cell.data.iter().any(value_contains_typed_integer))
}

fn validate_dummyvar_integer_levels(value: &Value) -> BuiltinResult<()> {
    match value {
        Value::Int(value) => validate_dummyvar_integer_level(value),
        Value::Tensor(tensor) => {
            if let Some(storage) = tensor.integer_storage() {
                for index in 0..storage.len() {
                    let value = storage.value_at(index).ok_or_else(|| {
                        internal(
                            "dummyvar",
                            "dummyvar: invalid typed-integer grouping storage",
                        )
                    })?;
                    validate_dummyvar_integer_level(&value)?;
                }
            }
            Ok(())
        }
        Value::Cell(cell) => {
            for item in &cell.data {
                validate_dummyvar_integer_levels(item)?;
            }
            Ok(())
        }
        _ => Ok(()),
    }
}

fn validate_dummyvar_integer_level(value: &IntValue) -> BuiltinResult<()> {
    let level = match value {
        IntValue::I8(value) if *value > 0 => u128::from(*value as u8),
        IntValue::I16(value) if *value > 0 => u128::from(*value as u16),
        IntValue::I32(value) if *value > 0 => u128::from(*value as u32),
        IntValue::I64(value) if *value > 0 => u128::from(*value as u64),
        IntValue::U8(value) if *value > 0 => u128::from(*value),
        IntValue::U16(value) if *value > 0 => u128::from(*value),
        IntValue::U32(value) if *value > 0 => u128::from(*value),
        IntValue::U64(value) if *value > 0 => u128::from(*value),
        _ => {
            return Err(invalid(
                "dummyvar",
                "dummyvar: numeric groups must contain positive integer levels or NaN",
            ))
        }
    };
    if level > MAX_ONEHOT_CELLS as u128 {
        return Err(invalid("dummyvar", "dummyvar: output is too large"));
    }
    Ok(())
}

#[runtime_builtin(
    name = "onehotencode",
    binding_variant = "default",
    builtin_path = "crate::builtins::stats::summary::encoding"
)]
async fn onehotencode_builtin(
    input: Value,
    feature_dim: Value,
    rest: Vec<Value>,
) -> BuiltinResult<Value> {
    let controls = AdmittedControls::encode(rest);
    controls.preflight_known_extensions()?;
    if crate::builtins::common::validation::value_contains_explicit_gpu(&input)
        || crate::builtins::common::validation::value_contains_explicit_gpu(&feature_dim)
        || controls.contains_explicit_gpu()
    {
        crate::compatibility::ensure_builtin_extension_enabled(
            &ONEHOTENCODE_EXPLICIT_GPU_EXTENSION,
            "onehotencode",
        )?;
    }
    let host_dim = if crate::value_contains_gpu(&feature_dim) {
        None
    } else {
        Some(positive_dim("onehotencode", &feature_dim)?)
    };
    let admitted_labels = if crate::value_contains_gpu(&input) {
        Admitted::Deferred(input)
    } else {
        Admitted::Ready(labels_from_value("onehotencode", input)?)
    };
    if let (Admitted::Ready(labels), Some(dim)) = (&admitted_labels, host_dim) {
        onehot_encoded_shape("onehotencode", &labels.shape, dim, 1)?;
    }
    let options = controls.resolve_encode().await?;
    let feature_dim = match host_dim {
        Some(dim) => dim,
        None => positive_dim("onehotencode", &gather("onehotencode", feature_dim).await?)?,
    };
    let labels = match admitted_labels {
        Admitted::Ready(labels) => labels,
        Admitted::Deferred(input) => {
            labels_from_value("onehotencode", gather("onehotencode", input).await?)?
        }
    };
    onehotencode_compute(labels, feature_dim, options)
}

#[runtime_builtin(
    name = "onehotdecode",
    binding_variant = "default",
    builtin_path = "crate::builtins::stats::summary::encoding"
)]
async fn onehotdecode_builtin(
    encoded: Value,
    classes: Value,
    feature_dim: Value,
    rest: Vec<Value>,
) -> BuiltinResult<Value> {
    let controls = AdmittedControls::decode(rest);
    controls.preflight_known_extensions()?;
    if encoded_value_needs_runmat_extension(&encoded) {
        crate::compatibility::ensure_builtin_extension_enabled(
            &ONEHOTDECODE_BROAD_ENCODED_EXTENSION,
            "onehotdecode",
        )?;
    }
    if crate::builtins::common::validation::value_contains_explicit_gpu(&encoded)
        || crate::builtins::common::validation::value_contains_explicit_gpu(&classes)
        || crate::builtins::common::validation::value_contains_explicit_gpu(&feature_dim)
        || controls.contains_explicit_gpu()
    {
        crate::compatibility::ensure_builtin_extension_enabled(
            &ONEHOTDECODE_EXPLICIT_GPU_EXTENSION,
            "onehotdecode",
        )?;
    }
    let host_dim = if crate::value_contains_gpu(&feature_dim) {
        None
    } else {
        Some(positive_dim("onehotdecode", &feature_dim)?)
    };
    let admitted_encoded = if crate::value_contains_gpu(&encoded) {
        Admitted::Deferred(encoded)
    } else {
        let host = encoded_values(encoded)?;
        validate_encoded_range(&host.data)?;
        Admitted::Ready(host)
    };
    let admitted_classes = if crate::value_contains_gpu(&classes) {
        Admitted::Deferred(classes)
    } else {
        let labels = labels_from_value("onehotdecode", classes)?;
        Admitted::Ready(nonmissing_classes("onehotdecode", &labels)?)
    };
    if matches!(&admitted_classes, Admitted::Ready(classes) if classes.is_empty()) {
        return Err(invalid(
            "onehotdecode",
            "onehotdecode: classes cannot be empty",
        ));
    }
    let options = controls.resolve_decode().await?;
    let feature_dim = match host_dim {
        Some(dim) => dim,
        None => positive_dim("onehotdecode", &gather("onehotdecode", feature_dim).await?)?,
    };
    let encoded = match admitted_encoded {
        Admitted::Ready(encoded) => encoded,
        Admitted::Deferred(value) => {
            let encoded = encoded_values(gather("onehotdecode", value).await?)?;
            validate_encoded_range(&encoded.data)?;
            encoded
        }
    };
    let classes = match admitted_classes {
        Admitted::Ready(classes) => classes,
        Admitted::Deferred(value) => {
            let labels = labels_from_value("onehotdecode", gather("onehotdecode", value).await?)?;
            nonmissing_classes("onehotdecode", &labels)?
        }
    };
    if classes.is_empty() {
        return Err(invalid(
            "onehotdecode",
            "onehotdecode: classes cannot be empty",
        ));
    }
    onehotdecode_compute(encoded, classes, feature_dim, options)
}

fn encoded_value_needs_runmat_extension(value: &Value) -> bool {
    matches!(
        value,
        Value::Bool(_) | Value::LogicalArray(_) | Value::Int(_)
    ) || matches!(value, Value::Tensor(tensor) if tensor.integer_storage().is_some())
        || matches!(value, Value::GpuTensor(handle) if runmat_accelerate_api::handle_integer_type(handle).is_some() || runmat_accelerate_api::handle_is_logical(handle))
}

async fn gather(name: &str, value: Value) -> BuiltinResult<Value> {
    gather_if_needed_async(&value).await.map_err(|cause| {
        let mut owner = invalid(name, format!("{name}: {cause}"));
        owner.gpu_gather_retry = cause.gpu_gather_retry;
        owner.span = cause.span;
        owner.source = Some(Box::new(cause));
        owner
    })
}

fn dummyvar_compute(group: Value) -> BuiltinResult<Value> {
    let groups = dummyvar_groups(group)?;
    let rows = groups.first().map(|group| group.labels.len()).unwrap_or(0);
    let mut total_cols = 0usize;
    let mut specs = Vec::with_capacity(groups.len());
    for group in groups {
        if group.labels.len() != rows {
            return Err(invalid(
                "dummyvar",
                "dummyvar: all grouping variables must have the same number of rows",
            ));
        }
        let classes = classes_for_dummyvar(&group, rows)?;
        total_cols = total_cols
            .checked_add(classes.len())
            .ok_or_else(|| invalid("dummyvar", "dummyvar: too many dummy variables"))?;
        specs.push((group, classes));
    }
    let cell_count = rows
        .checked_mul(total_cols)
        .ok_or_else(|| invalid("dummyvar", "dummyvar: output is too large"))?;
    if cell_count > MAX_ONEHOT_CELLS {
        return Err(invalid("dummyvar", "dummyvar: output is too large"));
    }
    let mut data = vec![0.0; cell_count];
    let mut col_offset = 0usize;
    for (group, classes) in specs {
        let index = class_index(&classes)?;
        for (row, label) in group.labels.iter().enumerate() {
            if label.is_none() {
                for col in 0..classes.len() {
                    data[row + (col_offset + col) * rows] = f64::NAN;
                }
                continue;
            }
            if let Some(class_idx) = label
                .as_ref()
                .and_then(|label| index.get(&label_key(label)))
            {
                data[row + (col_offset + *class_idx) * rows] = 1.0;
            }
        }
        col_offset += classes.len();
    }
    Tensor::new(data, vec![rows, total_cols])
        .map(Value::Tensor)
        .map_err(|err| internal("dummyvar", format!("dummyvar: {err}")))
}

#[derive(Default)]
struct OnehotEncodeOptions {
    classes: Option<LabelArray>,
    output: Option<OutputKind>,
}

fn onehotencode_compute(
    labels: LabelArray,
    feature_dim: usize,
    options: OnehotEncodeOptions,
) -> BuiltinResult<Value> {
    let classes = if let Some(class_names) = options.classes {
        if !label_kinds_compatible(labels.kind, class_names.kind) {
            return Err(invalid(
                "onehotencode",
                "onehotencode: ClassNames must match input label type",
            ));
        }
        nonmissing_classes("onehotencode", &class_names)?
    } else if labels.kind == LabelKind::Categorical {
        classes_for_labels(&labels)
    } else {
        return Err(invalid(
            "onehotencode",
            "onehotencode: ClassNames is required for noncategorical input",
        ));
    };
    if classes.is_empty() {
        return Err(invalid(
            "onehotencode",
            "onehotencode: class names cannot be empty",
        ));
    }
    let shape = onehot_encoded_shape("onehotencode", &labels.shape, feature_dim, classes.len())?;
    let cell_count = element_count(&shape)?;
    if cell_count > MAX_ONEHOT_CELLS {
        return Err(invalid("onehotencode", "onehotencode: output is too large"));
    }
    let mut values = vec![0.0; cell_count];
    let mut has_nan_vector = false;
    let index = class_index(&classes)?;
    let out_strides = strides(&shape);
    let input_shape = normalized_shape(&labels.shape);
    let input_strides = strides(&input_shape);
    let feature_axis = feature_dim - 1;
    for idx in 0..labels.labels.len() {
        let Some(label) = &labels.labels[idx] else {
            fill_feature_vector(
                &mut values,
                &shape,
                &out_strides,
                &input_shape,
                &input_strides,
                feature_axis,
                idx,
                classes.len(),
                f64::NAN,
            );
            has_nan_vector = true;
            continue;
        };
        let Some(class_idx) = index.get(&label_key(label)).copied() else {
            fill_feature_vector(
                &mut values,
                &shape,
                &out_strides,
                &input_shape,
                &input_strides,
                feature_axis,
                idx,
                classes.len(),
                f64::NAN,
            );
            has_nan_vector = true;
            continue;
        };
        let mut coords = coords_for_linear(idx, &input_shape, &input_strides);
        coords.resize(shape.len(), 0);
        coords[feature_axis] = class_idx;
        let out_idx = linear_for_coords(&coords, &out_strides);
        values[out_idx] = 1.0;
    }
    match options
        .output
        .unwrap_or(OutputKind::Numeric(NumericClass::Double))
    {
        OutputKind::Logical => {
            if has_nan_vector {
                return Err(invalid(
                    "onehotencode",
                    "onehotencode: logical output cannot represent missing or excluded labels",
                ));
            }
            LogicalArray::new(
                values
                    .into_iter()
                    .map(|value| u8::from(value != 0.0))
                    .collect(),
                shape,
            )
            .map(Value::LogicalArray)
            .map_err(|err| internal("onehotencode", format!("onehotencode: {err}")))
        }
        OutputKind::Numeric(dtype) => {
            if has_nan_vector && !matches!(dtype, NumericClass::Double | NumericClass::Single) {
                return Err(invalid(
                    "onehotencode",
                    "onehotencode: integer output cannot represent missing or excluded labels",
                ));
            }
            onehot_numeric_output(values, shape, dtype.into())
        }
        _ => Err(invalid(
            "onehotencode",
            "onehotencode: OutputType must be a numeric or logical type",
        )),
    }
}

struct OnehotDecodeOptions {
    output: Option<OutputKind>,
}

fn onehotdecode_compute(
    encoded: EncodedArray,
    classes: Vec<Label>,
    feature_dim: usize,
    options: OnehotDecodeOptions,
) -> BuiltinResult<Value> {
    let shape = normalized_shape(&encoded.shape);
    let feature_axis = feature_dim - 1;
    if feature_axis >= shape.len() {
        return Err(invalid(
            "onehotdecode",
            "onehotdecode: featureDim exceeds encoded data dimensions",
        ));
    }
    if shape[feature_axis] != classes.len() {
        return Err(invalid(
            "onehotdecode",
            "onehotdecode: classes length must match the feature dimension",
        ));
    }
    let mut out_shape = shape.clone();
    out_shape[feature_axis] = 1;
    let out_count = element_count(&out_shape)?;
    let out_strides = strides(&out_shape);
    let encoded_strides = strides(&shape);
    let mut decoded = Vec::with_capacity(out_count);
    for out_idx in 0..out_count {
        let mut coords = coords_for_linear(out_idx, &out_shape, &out_strides);
        let mut best_idx = 0usize;
        let mut best_value = f64::NEG_INFINITY;
        for class_idx in 0..classes.len() {
            coords[feature_axis] = class_idx;
            let value = encoded.data[linear_for_coords(&coords, &encoded_strides)];
            if value.is_nan() {
                continue;
            }
            if value > best_value {
                best_value = value;
                best_idx = class_idx;
            }
        }
        decoded.push(classes[best_idx].clone());
    }
    labels_to_value(
        "onehotdecode",
        &decoded,
        out_shape,
        options.output.unwrap_or(OutputKind::Categorical),
        Some(&classes),
    )
}

fn parse_output_kind(name: &str, text: &str) -> BuiltinResult<OutputKind> {
    OutputKind::from_public(text).ok_or_else(|| {
        invalid(
            name,
            format!("{name}: unsupported output type '{}'", canonical(text)),
        )
    })
}

fn positive_dim(name: &str, value: &Value) -> BuiltinResult<usize> {
    if let Value::Int(value) = value {
        return value.try_to_usize().filter(|dim| *dim >= 1).ok_or_else(|| {
            invalid(
                name,
                format!("{name}: featureDim must be a positive integer scalar"),
            )
        });
    }
    if let Value::Tensor(tensor) = value {
        if tensor::is_scalar_tensor(tensor) {
            if let Some(storage) = tensor.integer_storage() {
                return storage
                    .value_at(0)
                    .and_then(|value| value.try_to_usize())
                    .filter(|dim| *dim >= 1)
                    .ok_or_else(|| {
                        invalid(
                            name,
                            format!("{name}: featureDim must be a positive integer scalar"),
                        )
                    });
            }
        }
    }
    let dim = match value {
        Value::Num(value) => *value,
        Value::Tensor(tensor) if tensor::is_scalar_tensor(tensor) => {
            tensor::tensor_value_f64(tensor, 0)
        }
        _ => {
            return Err(invalid(
                name,
                format!("{name}: featureDim must be a positive integer scalar"),
            ))
        }
    };
    if dim.fract() != 0.0
        || dim < 1.0
        || dim > usize::MAX as f64
        || (usize::BITS == 64 && dim == usize::MAX as f64)
    {
        return Err(invalid(
            name,
            format!("{name}: featureDim must be a positive integer scalar"),
        ));
    }
    Ok(dim as usize)
}

#[cfg(test)]
#[path = "encoding/tests.rs"]
mod tests;
