//! MATLAB-compatible `csvwrite` builtin for RunMat.
//!
//! `csvwrite` is an older convenience wrapper that persists numeric matrices to
//! comma-separated text files. Modern MATLAB code typically prefers
//! `writematrix`, but many legacy scripts still depend on `csvwrite`'s terse
//! API and zero-based offset arguments. This implementation mirrors those
//! semantics while integrating with RunMat's builtin framework.

use std::io::Write;
use std::path::{Path, PathBuf};

use runmat_builtins::{
    BuiltinErrorDescriptor, CSVWRITE_BYTES_OUTPUT_EXTENSION, CSVWRITE_ERROR_DATA_INPUT,
    CSVWRITE_ERROR_DATA_SHAPE, CSVWRITE_ERROR_FILENAME, CSVWRITE_ERROR_FILENAME_EMPTY,
    CSVWRITE_ERROR_IO_OPEN, CSVWRITE_ERROR_IO_WRITE, CSVWRITE_ERROR_OFFSETS,
    CSVWRITE_RESIDENT_INPUT_EXTENSION,
};
#[cfg(test)]
use runmat_builtins::{CSVWRITE_DESCRIPTOR, CSVWRITE_EXTENSIONS, CSVWRITE_INTEGER_CAPABILITIES};
use runmat_filesystem::OpenOptions;
use runmat_macros::runtime_builtin;
use runmat_value::{ComplexTensor, Tensor, Value};

use crate::builtins::common::fs::expand_user_path;
use crate::builtins::common::tensor;
use crate::{build_runtime_error, gather_if_needed_async, BuiltinResult, RuntimeError};

const BUILTIN_NAME: &str = "csvwrite";

use super::cooperation::{await_io, TabularCooperation};
use super::path::{TabularFsOwner, TabularFsPath};

fn csvwrite_error_with(
    error: &'static BuiltinErrorDescriptor,
    message: impl Into<String>,
) -> RuntimeError {
    let mut builder = build_runtime_error(message).with_builtin(BUILTIN_NAME);
    if let Some(identifier) = error.identifier {
        builder = builder.with_identifier(identifier);
    }
    builder.build()
}

fn csvwrite_error_with_source<E>(
    error: &'static BuiltinErrorDescriptor,
    message: impl Into<String>,
    source: E,
) -> RuntimeError
where
    E: crate::runtime_error::RuntimeErrorSourceBound,
{
    let mut builder = build_runtime_error(message)
        .with_builtin(BUILTIN_NAME)
        .with_source(source);
    if let Some(identifier) = error.identifier {
        builder = builder.with_identifier(identifier);
    }
    builder.build()
}

fn map_control_flow(err: RuntimeError) -> RuntimeError {
    let identifier = err.identifier().map(|value| value.to_string());
    let message = err.message().to_string();
    let mut builder = build_runtime_error(message)
        .with_builtin(BUILTIN_NAME)
        .with_source(err);
    if let Some(identifier) = identifier {
        builder = builder.with_identifier(identifier);
    }
    builder.build()
}

#[runtime_builtin(
    name = "csvwrite",
    binding_variant = "default",
    context = true,
    builtin_path = "crate::builtins::io::tabular::csvwrite"
)]
async fn csvwrite_builtin(
    runtime: Option<crate::context::RuntimeContext>,
    filename: Value,
    data: Value,
    rest: crate::call::BuiltinArguments,
) -> crate::BuiltinResult<Value> {
    let mut cooperation = TabularCooperation::new(runtime.as_ref(), BUILTIN_NAME);
    let requested_outputs = crate::output_count::current_output_count();
    if requested_outputs.is_some_and(|count| count > 1) {
        return Err(csvwrite_error_with(
            &CSVWRITE_ERROR_DATA_INPUT,
            "csvwrite: too many output arguments",
        ));
    }
    if requested_outputs.is_some_and(|count| count > 0) {
        crate::compatibility::ensure_builtin_extension_enabled(
            &CSVWRITE_BYTES_OUTPUT_EXTENSION,
            BUILTIN_NAME,
        )?;
    }
    if matches!(filename, Value::GpuTensor(_))
        || matches!(data, Value::GpuTensor(_))
        || rest
            .iter()
            .any(|value| matches!(value, Value::GpuTensor(_)))
    {
        crate::compatibility::ensure_builtin_extension_enabled(
            &CSVWRITE_RESIDENT_INPUT_EXTENSION,
            BUILTIN_NAME,
        )?;
    }
    if !matches!(rest.len(), 0 | 2) {
        return Err(csvwrite_error_with(
            &CSVWRITE_ERROR_OFFSETS,
            "csvwrite: offsets must be provided as two numeric arguments (row, column)",
        ));
    }
    preflight_data(&data)?;
    for (index, label) in [(0, "row offset"), (1, "column offset")] {
        if let Some(value) = rest.get(index) {
            if !matches!(value, Value::GpuTensor(_)) {
                parse_offset(value, label)?;
            }
        }
    }
    let known_path = if matches!(filename, Value::GpuTensor(_)) {
        None
    } else {
        Some(resolve_path(&filename)?)
    };
    let owner = TabularFsOwner::capture(known_path.as_deref()).map_err(|source| {
        csvwrite_error_with_source(
            &CSVWRITE_ERROR_IO_OPEN,
            "csvwrite: unable to capture filesystem authority",
            source,
        )
    })?;
    let filename_value = await_io(
        runtime.as_ref(),
        BUILTIN_NAME,
        gather_if_needed_async(&filename),
    )
    .await?
    .map_err(map_control_flow)?;
    cooperation.completed(0).await?;
    let path = owner
        .path(resolve_path(&filename_value)?)
        .map_err(|source| {
            csvwrite_error_with_source(
                &CSVWRITE_ERROR_IO_OPEN,
                "csvwrite: captured filesystem authority was revoked",
                source,
            )
        })?;

    let mut gathered_offsets = Vec::with_capacity(rest.len());
    for value in &rest {
        gathered_offsets.push(
            await_io(
                runtime.as_ref(),
                BUILTIN_NAME,
                gather_if_needed_async(value),
            )
            .await?
            .map_err(map_control_flow)?,
        );
        cooperation.completed(0).await?;
    }
    let (row_offset, col_offset) = parse_offsets(&gathered_offsets)?;

    let gathered_data = await_io(
        runtime.as_ref(),
        BUILTIN_NAME,
        gather_if_needed_async(&data),
    )
    .await?
    .map_err(map_control_flow)?;
    cooperation.completed(0).await?;
    let matrix = CsvMatrix::from_value(gathered_data)?;
    matrix.ensure_matrix_shape()?;

    let bytes = write_csv(
        runtime.as_ref(),
        &mut cooperation,
        &path,
        &matrix,
        row_offset,
        col_offset,
    )
    .await?;
    cooperation.completed(0).await?;
    match requested_outputs {
        Some(0) => Ok(Value::output_list(Vec::new())),
        Some(1) => Ok(Value::output_list(vec![Value::Num(bytes as f64)])),
        Some(_) => Err(csvwrite_error_with(
            &CSVWRITE_ERROR_DATA_INPUT,
            "csvwrite: too many output arguments",
        )),
        None => Ok(Value::Num(bytes as f64)),
    }
}

fn preflight_data(value: &Value) -> BuiltinResult<()> {
    match value {
        Value::Num(_) | Value::Int(_) | Value::Bool(_) | Value::Complex(_, _) => Ok(()),
        Value::Tensor(tensor) => ensure_matrix_dims(tensor.shape()),
        Value::LogicalArray(array) => ensure_matrix_dims(&array.shape),
        Value::ComplexTensor(tensor) => {
            if tensor.integer_storage().is_some() {
                return Err(csvwrite_error_with(
                    &CSVWRITE_ERROR_DATA_INPUT,
                    "csvwrite: complex integer input is not supported",
                ));
            }
            ensure_matrix_dims(tensor.shape())
        }
        Value::GpuTensor(handle) => ensure_matrix_dims(handle.shape()),
        _ => Err(csvwrite_error_with(
            &CSVWRITE_ERROR_DATA_INPUT,
            "csvwrite: input must be numeric or logical",
        )),
    }
}

enum CsvMatrix {
    Real(Tensor),
    Complex(ComplexTensor),
}

impl CsvMatrix {
    fn from_value(value: Value) -> BuiltinResult<Self> {
        match value {
            Value::Complex(re, im) => ComplexTensor::new(vec![(re, im)], vec![1, 1])
                .map(Self::Complex)
                .map_err(|e| {
                    csvwrite_error_with(&CSVWRITE_ERROR_DATA_INPUT, format!("csvwrite: {e}"))
                }),
            Value::ComplexTensor(tensor) if tensor.integer_storage().is_none() => {
                Ok(Self::Complex(tensor))
            }
            Value::ComplexTensor(_) => Err(csvwrite_error_with(
                &CSVWRITE_ERROR_DATA_INPUT,
                "csvwrite: complex integer input is not supported",
            )),
            value => tensor::value_into_tensor_for("csvwrite", value)
                .map(Self::Real)
                .map_err(|msg| {
                    csvwrite_error_with(&CSVWRITE_ERROR_DATA_INPUT, format!("csvwrite: {msg}"))
                }),
        }
    }

    fn shape(&self) -> &[usize] {
        match self {
            Self::Real(t) => t.shape(),
            Self::Complex(t) => t.shape(),
        }
    }

    fn ensure_matrix_shape(&self) -> BuiltinResult<()> {
        ensure_matrix_dims(self.shape())
    }

    fn rows(&self) -> usize {
        self.shape().first().copied().unwrap_or(1)
    }
    fn cols(&self) -> usize {
        self.shape().get(1).copied().unwrap_or(1)
    }

    fn format_at(&self, idx: usize) -> String {
        match self {
            Self::Real(tensor) => format_tensor_value(tensor, idx),
            Self::Complex(tensor) => {
                let (re, im) = tensor
                    .numeric_value_at(idx)
                    .expect("index within authoritative complex storage");
                format_complex(re.materialize_f64(), im.materialize_f64())
            }
        }
    }
}

fn resolve_path(value: &Value) -> BuiltinResult<PathBuf> {
    let raw =
        crate::builtins::common::validation::value_text_scalar_string(value).ok_or_else(|| {
            csvwrite_error_with(
                &CSVWRITE_ERROR_FILENAME,
                "csvwrite: filename must be a string scalar or character vector",
            )
        })?;

    if raw.trim().is_empty() {
        return Err(csvwrite_error_with(
            &CSVWRITE_ERROR_FILENAME_EMPTY,
            CSVWRITE_ERROR_FILENAME_EMPTY.message,
        ));
    }

    let expanded = expand_user_path(&raw, BUILTIN_NAME)
        .map_err(|msg| csvwrite_error_with(&CSVWRITE_ERROR_FILENAME, msg))?;
    Ok(Path::new(&expanded).to_path_buf())
}

fn parse_offsets(args: &[Value]) -> BuiltinResult<(usize, usize)> {
    match args.len() {
        0 => Ok((0, 0)),
        2 => {
            let row = parse_offset(&args[0], "row offset")?;
            let col = parse_offset(&args[1], "column offset")?;
            Ok((row, col))
        }
        _ => Err(csvwrite_error_with(
            &CSVWRITE_ERROR_OFFSETS,
            "csvwrite: offsets must be provided as two numeric arguments (row, column)",
        )),
    }
}

fn parse_offset(value: &Value, context: &str) -> BuiltinResult<usize> {
    match value {
        Value::Int(i) => i.try_to_usize().ok_or_else(|| {
            csvwrite_error_with(
                &CSVWRITE_ERROR_OFFSETS,
                format!("csvwrite: {context} must be >= 0"),
            )
        }),
        Value::Num(n) => coerce_offset_from_float(*n, context),
        Value::Bool(b) => Ok(if *b { 1 } else { 0 }),
        Value::Tensor(t) => {
            let len = tensor::tensor_element_len(t);
            if len != 1 {
                return Err(csvwrite_error_with(
                    &CSVWRITE_ERROR_OFFSETS,
                    format!("csvwrite: {context} must be a scalar, got {} elements", len),
                ));
            }
            let value = t
                .numeric_value_at(0)
                .expect("one-element authoritative numeric storage");
            if let Some(value) = value.into_int_value() {
                return value.try_to_usize().ok_or_else(|| {
                    csvwrite_error_with(
                        &CSVWRITE_ERROR_OFFSETS,
                        format!("csvwrite: {context} must be >= 0"),
                    )
                });
            }
            coerce_offset_from_float(value.materialize_f64(), context)
        }
        Value::LogicalArray(logical) => {
            if logical.data.len() != 1 {
                return Err(csvwrite_error_with(
                    &CSVWRITE_ERROR_OFFSETS,
                    format!(
                        "csvwrite: {context} must be a scalar, got {} elements",
                        logical.data.len()
                    ),
                ));
            }
            Ok(if logical.data[0] != 0 { 1 } else { 0 })
        }
        other => Err(csvwrite_error_with(
            &CSVWRITE_ERROR_OFFSETS,
            format!("csvwrite: {context} must be numeric, got {:?}", other),
        )),
    }
}

fn coerce_offset_from_float(value: f64, context: &str) -> BuiltinResult<usize> {
    if !value.is_finite() {
        return Err(csvwrite_error_with(
            &CSVWRITE_ERROR_OFFSETS,
            format!("csvwrite: {context} must be finite"),
        ));
    }
    let rounded = value.round();
    if (rounded - value).abs() > 1e-9 {
        return Err(csvwrite_error_with(
            &CSVWRITE_ERROR_OFFSETS,
            format!("csvwrite: {context} must be an integer"),
        ));
    }
    if rounded < 0.0 {
        return Err(csvwrite_error_with(
            &CSVWRITE_ERROR_OFFSETS,
            format!("csvwrite: {context} must be >= 0"),
        ));
    }
    if rounded > usize::MAX as f64 || (usize::BITS == 64 && rounded == usize::MAX as f64) {
        return Err(csvwrite_error_with(
            &CSVWRITE_ERROR_OFFSETS,
            format!("csvwrite: {context} is too large"),
        ));
    }
    Ok(rounded as usize)
}

fn ensure_matrix_dims(shape: &[usize]) -> BuiltinResult<()> {
    if shape.len() <= 2 {
        return Ok(());
    }
    if shape[2..].iter().all(|&dim| dim == 1) {
        return Ok(());
    }
    Err(csvwrite_error_with(
        &CSVWRITE_ERROR_DATA_SHAPE,
        "csvwrite: input must be 2-D; reshape before writing",
    ))
}

#[path = "csvwrite/writer.rs"]
mod writer;
use writer::*;
#[cfg(test)]
#[path = "csvwrite/tests.rs"]
pub(crate) mod tests;
