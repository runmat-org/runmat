//! MATLAB-compatible `writematrix` builtin for emitting delimited text files.

use std::path::Path;

#[cfg(test)]
use runmat_builtins::WRITEMATRIX_DESCRIPTOR;
use runmat_builtins::{
    BuiltinErrorDescriptor, WRITEMATRIX_BYTES_OUTPUT_EXTENSION, WRITEMATRIX_ERROR_ARG_CONFIG,
    WRITEMATRIX_ERROR_DATA, WRITEMATRIX_ERROR_DATA_SHAPE, WRITEMATRIX_ERROR_FILENAME,
    WRITEMATRIX_ERROR_IO, WRITEMATRIX_ERROR_OPTION, WRITEMATRIX_EXPLICIT_GPU_EXTENSION,
    WRITEMATRIX_QUOTE_ALIAS_EXTENSION, WRITEMATRIX_TEXT_OPTIONS_EXTENSION,
};
use runmat_macros::runtime_builtin;
use runmat_value::{Tensor, Value};

use crate::builtins::common::tensor;
use crate::{build_runtime_error, gather_if_needed_async, BuiltinResult, RuntimeError};

use super::cooperation::{await_io, TabularCooperation};
use super::path::{TabularFsOwner, TabularFsPath};

const BUILTIN_NAME: &str = "writematrix";

fn writematrix_error(error: &'static BuiltinErrorDescriptor) -> RuntimeError {
    writematrix_error_with(error, error.message)
}

fn writematrix_error_with(
    error: &'static BuiltinErrorDescriptor,
    message: impl Into<String>,
) -> RuntimeError {
    let mut builder = build_runtime_error(message).with_builtin(BUILTIN_NAME);
    if let Some(identifier) = error.identifier {
        builder = builder.with_identifier(identifier);
    }
    builder.build()
}

fn writematrix_error_with_source<E>(
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
    name = "writematrix",
    context = true,
    binding_variant = "default",
    builtin_path = "crate::builtins::io::tabular::writematrix"
)]
async fn writematrix_builtin(
    runtime: Option<crate::context::RuntimeContext>,
    data: Value,
    rest: crate::call::BuiltinArguments,
) -> crate::BuiltinResult<Value> {
    let mut cooperation = TabularCooperation::new(runtime.as_ref(), BUILTIN_NAME);
    if rest.is_empty() {
        return Err(writematrix_error(&WRITEMATRIX_ERROR_ARG_CONFIG));
    }
    preflight_extensions(&data, rest.view())?;
    preflight_data(&data)?;
    let path = resolve_path(&rest[0])?;
    let options_args = rest.range(1..)?;
    if !options_args.len().is_multiple_of(2) {
        return Err(writematrix_error(&WRITEMATRIX_ERROR_ARG_CONFIG));
    }
    preflight_host_options(options_args.view())?;
    let host_options = if options_args
        .iter()
        .all(|value| !matches!(value, Value::GpuTensor(_)))
    {
        Some(parse_options(options_args.view())?)
    } else {
        None
    };
    let owner = TabularFsOwner::capture(Some(&path)).map_err(|source| {
        writematrix_error_with_source(
            &WRITEMATRIX_ERROR_IO,
            "writematrix: unable to capture filesystem authority",
            source,
        )
    })?;
    let selected = owner.path(path).map_err(|source| {
        writematrix_error_with_source(
            &WRITEMATRIX_ERROR_IO,
            "writematrix: captured filesystem authority was revoked",
            source,
        )
    })?;
    let options = match host_options {
        Some(options) => options,
        None => {
            let gather = options_args.try_map_preserving_roots_async(
                options_args.len(),
                runtime.as_ref(),
                async |index| {
                    let value = await_io(
                        runtime.as_ref(),
                        BUILTIN_NAME,
                        gather_if_needed_async(&options_args[index]),
                    )
                    .await?
                    .map_err(map_control_flow)?;
                    cooperation.completed(0).await?;
                    Ok(value)
                },
            );
            let gathered_options = match runtime.as_ref() {
                Some(runtime) => runtime.scope(gather).await?,
                None => gather.await?,
            };
            parse_options(gathered_options.view())?
        }
    };
    let gather_data = await_io(
        runtime.as_ref(),
        BUILTIN_NAME,
        gather_if_needed_async(&data),
    );
    let gathered = match runtime.as_ref() {
        Some(runtime) => runtime.scope(gather_data).await?,
        None => gather_data.await?,
    }
    .map_err(map_control_flow)?;
    cooperation.completed(0).await?;
    let matrix = MatrixData::from_value(gathered)?;
    let bytes_written = write_matrix(
        runtime.as_ref(),
        &mut cooperation,
        &selected,
        &matrix,
        &options,
    )
    .await?;
    cooperation.completed(0).await?;
    selected.check().map_err(|source| {
        writematrix_error_with_source(
            &WRITEMATRIX_ERROR_IO,
            "writematrix: captured filesystem authority was revoked",
            source,
        )
    })?;
    match crate::output_count::current_output_count() {
        Some(0) => Ok(Value::output_list(Vec::new())),
        Some(1) => Ok(Value::output_list(vec![Value::Num(bytes_written as f64)])),
        Some(_) => Err(writematrix_error(&WRITEMATRIX_ERROR_ARG_CONFIG)),
        None => Ok(Value::Num(bytes_written as f64)),
    }
}

#[derive(Debug, Clone)]
struct WriteMatrixOptions {
    delimiter: Option<String>,
    write_mode: WriteMode,
    quote_strings: QuoteStrings,
    line_ending: LineEnding,
    decimal_separator: char,
    file_type: FileType,
}

impl Default for WriteMatrixOptions {
    fn default() -> Self {
        Self {
            delimiter: None,
            write_mode: WriteMode::Overwrite,
            quote_strings: QuoteStrings::Minimal,
            line_ending: LineEnding::Auto,
            decimal_separator: '.',
            file_type: FileType::DelimitedText,
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum QuoteStrings {
    Minimal,
    All,
    None,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum WriteMode {
    Overwrite,
    Append,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum LineEnding {
    Auto,
    Unix,
    Windows,
    Mac,
}

impl LineEnding {
    fn as_str(&self) -> &'static str {
        match self {
            LineEnding::Auto | LineEnding::Unix => "\n",
            LineEnding::Windows => "\r\n",
            LineEnding::Mac => "\r",
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum FileType {
    DelimitedText,
    Text,
}

#[path = "writematrix/controls.rs"]
mod controls;
use controls::*;

enum MatrixData {
    Numeric(Tensor),
    Text {
        rows: usize,
        cols: usize,
        data: Vec<String>,
    },
}

impl MatrixData {
    fn from_value(value: Value) -> BuiltinResult<Self> {
        match value {
            Value::StringArray(sa) if sa.data().iter().all(|text| !text.is_missing()) => {
                ensure_matrix_shape(&sa.shape, "string")?;
                let (rows, cols) = if sa.shape.is_empty() {
                    (1, 1)
                } else {
                    (sa.rows, sa.cols)
                };
                Ok(MatrixData::Text {
                    rows,
                    cols,
                    data: sa.data().iter().map(|text| text.as_str().expect("checked nonmissing string").to_owned()).collect(),
                })
            }
            Value::String(runmat_value::StringElement::Text(s)) => Ok(MatrixData::Text {
                rows: 1,
                cols: 1,
                data: vec![s.into_string_unbounded()],
            }),
            Value::CharArray(_) => Err(writematrix_error_with(
                &WRITEMATRIX_ERROR_DATA,
                "writematrix: character arrays are not supported; convert to string arrays or use writecell",
            )),
            Value::Cell(_) => Err(writematrix_error_with(
                &WRITEMATRIX_ERROR_DATA,
                "writematrix: cell arrays are not supported; use writecell for heterogeneous content",
            )),
            Value::ComplexTensor(_) | Value::Complex(_, _) => Err(writematrix_error_with(
                &WRITEMATRIX_ERROR_DATA,
                "writematrix: complex values are not supported; write real and imaginary parts separately",
            )),
            other => {
                let tensor = tensor::value_into_tensor_for("writematrix", other)
                    .map_err(|msg| {
                        writematrix_error_with(
                            &WRITEMATRIX_ERROR_DATA,
                            format!("writematrix: {msg}"),
                        )
                    })?;
                ensure_matrix_shape(tensor.shape(), "numeric")?;
                Ok(MatrixData::Numeric(tensor))
            }
        }
    }

    fn rows(&self) -> usize {
        match self {
            MatrixData::Numeric(t) => t.rows(),
            MatrixData::Text { rows, .. } => *rows,
        }
    }

    fn cols(&self) -> usize {
        match self {
            MatrixData::Numeric(t) => t.cols(),
            MatrixData::Text { cols, .. } => *cols,
        }
    }

    fn cell_at(&self, row: usize, col: usize, options: &WriteMatrixOptions) -> MatrixCell<'_> {
        match self {
            MatrixData::Numeric(tensor) => {
                let rows = tensor.rows();
                let idx = row + col * rows;
                let value = tensor
                    .numeric_value_at(idx)
                    .expect("index within authoritative numeric storage");
                MatrixCell::Numeric(format_numeric(value, options.decimal_separator))
            }
            MatrixData::Text { rows, data, .. } => {
                if *rows == 0 {
                    return MatrixCell::Text("");
                }
                let idx = row + col * rows;
                MatrixCell::Text(&data[idx])
            }
        }
    }
}

enum MatrixCell<'a> {
    Numeric(String),
    Text(&'a str),
}

fn preflight_data(value: &Value) -> BuiltinResult<()> {
    match value {
        Value::Num(_) | Value::Int(_) | Value::Bool(_) | Value::String(runmat_value::StringElement::Text(_)) => Ok(()),
        Value::Tensor(tensor) => ensure_matrix_shape(tensor.shape(), "numeric"),
        Value::LogicalArray(array) => ensure_matrix_shape(&array.shape, "logical"),
        Value::StringArray(array) => ensure_matrix_shape(&array.shape, "string"),
        Value::GpuTensor(handle) => {
            if handle.physical().storage()
                == runmat_accelerate_api::GpuTensorStorage::ComplexInterleaved
            {
                return Err(writematrix_error_with(
                    &WRITEMATRIX_ERROR_DATA,
                    "writematrix: complex values are not supported; write real and imaginary parts separately",
                ));
            }
            ensure_matrix_shape(handle.shape(), "numeric")
        }
        Value::CharArray(_) => Err(writematrix_error_with(
            &WRITEMATRIX_ERROR_DATA,
            "writematrix: character arrays are not supported; convert to string arrays or use writecell",
        )),
        Value::Cell(_) => Err(writematrix_error_with(
            &WRITEMATRIX_ERROR_DATA,
            "writematrix: cell arrays are not supported; use writecell for heterogeneous content",
        )),
        Value::ComplexTensor(_) | Value::Complex(_, _) => Err(writematrix_error_with(
            &WRITEMATRIX_ERROR_DATA,
            "writematrix: complex values are not supported; write real and imaginary parts separately",
        )),
        _ => Err(writematrix_error(&WRITEMATRIX_ERROR_DATA)),
    }
}

fn ensure_matrix_shape(shape: &[usize], context: &str) -> BuiltinResult<()> {
    if shape.len() <= 2 {
        return Ok(());
    }
    if shape[2..].iter().all(|&dim| dim == 1) {
        Ok(())
    } else {
        Err(writematrix_error_with(
            &WRITEMATRIX_ERROR_DATA_SHAPE,
            format!(
                "writematrix: {context} input must be 2-D; reshape or use writecell for higher dimensions"
            ),
        ))
    }
}

#[path = "writematrix/writer.rs"]
mod writer;
use writer::*;

impl WriteMatrixOptions {
    fn resolve_delimiter(&self, path: &Path) -> String {
        if let Some(custom) = &self.delimiter {
            return custom.clone();
        }
        match self.file_type {
            FileType::Text => " ".to_string(),
            FileType::DelimitedText => default_delimiter_for_path(path),
        }
    }
}

#[path = "writematrix/format.rs"]
mod format;
use format::*;

#[path = "writematrix/admission.rs"]
mod admission;
use admission::*;

#[cfg(test)]
#[path = "writematrix/tests.rs"]
pub(crate) mod tests;
