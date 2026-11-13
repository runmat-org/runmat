//! MATLAB-compatible `disp` builtin with GPU-aware formatting semantics.

#[cfg(test)]
use runmat_builtins::DISP_DESCRIPTOR;
use runmat_builtins::{
    BuiltinErrorDescriptor, DISP_ERROR_ARG_CONFIG, DISP_ERROR_GATHER, DISP_ERROR_TOO_MANY_OUTPUTS,
};
use runmat_macros::runtime_builtin;
#[cfg(test)]
use runmat_value::set_display_format;
use runmat_value::{
    get_display_format, CellArray, CharArray, ComplexTensor, FormatMode, IntValue, IntegerStorage,
    LogicalArray, StringArray, StructValue, Tensor, Value,
};

use crate::builtins::common::tensor;
use crate::console::{record_console_line, ConsoleStream};
use crate::gather_if_needed_async;

/// Minimum column width (in characters) for numeric and logical displays.
const NUMERIC_MIN_COLUMN_WIDTH: usize = 6;
/// Number of leading spaces for cell array rows.
const CELL_ROW_INDENT: usize = 4;
/// Number of spaces used for struct field indentation.
const STRUCT_FIELD_INDENT: usize = 4;
/// Continuation indent for multi-line struct field values.
const STRUCT_CONTINUATION_INDENT: usize = 8;

#[derive(Clone, Copy)]
enum RenderMode {
    TopLevel,
    Nested,
}

#[derive(Clone, Copy)]
enum Align {
    Left,
    Right,
}

fn disp_error(error: &'static BuiltinErrorDescriptor) -> crate::RuntimeError {
    disp_error_with(error, error.message)
}

fn disp_error_with(
    error: &'static BuiltinErrorDescriptor,
    message: impl Into<String>,
) -> crate::RuntimeError {
    let mut builder = crate::build_runtime_error(message).with_builtin("disp");
    if let Some(identifier) = error.identifier {
        builder = builder.with_identifier(identifier);
    }
    builder.build()
}

fn disp_gather_error(source: crate::RuntimeError) -> crate::RuntimeError {
    let message = format!("disp: {source}");
    let mut builder = crate::build_runtime_error(message)
        .with_builtin("disp")
        .with_source(source);
    if let Some(identifier) = DISP_ERROR_GATHER.identifier {
        builder = builder.with_identifier(identifier);
    }
    builder.build()
}

#[runtime_builtin(
    name = "disp",
    binding_variant = "default",
    builtin_path = "crate::builtins::io::disp"
)]
async fn disp_builtin(
    value: Value,
    rest: crate::call::BuiltinArguments,
) -> crate::BuiltinResult<Value> {
    if matches!(crate::output_count::current_output_count(), Some(n) if n > 0) {
        return Err(disp_error(&DISP_ERROR_TOO_MANY_OUTPUTS));
    }
    if !rest.is_empty() {
        return Err(disp_error(&DISP_ERROR_ARG_CONFIG));
    }

    let host_value = gather_if_needed_async(&value)
        .await
        .map_err(disp_gather_error)?;
    let lines = format_for_disp(&host_value);

    if !lines.is_empty() {
        let body = lines.join("\n");
        record_console_line(ConsoleStream::Stdout, body)?;
    }

    Ok(empty_return_value())
}

mod render;
#[cfg(test)]
use render::render_value;
pub(crate) use render::{format_for_disp, format_for_display};

pub(crate) fn empty_return_value() -> Value {
    Value::Tensor(Tensor::zeros(vec![0, 0]))
}

#[cfg(test)]
#[path = "disp/tests.rs"]
pub(crate) mod tests;
