//! MATLAB-compatible `jsonencode` builtin for serialising RunMat values to JSON text.

mod convert;
mod options;
mod render;

use convert::value_to_json;
use options::{
    parse_options, preflight_jsonencode_extensions, preflight_option_extensions,
    prepare_option_layout,
};
use render::render_json;

use std::collections::BTreeMap;
use std::fmt::Write as FmtWrite;

use runmat_builtins::{
    BuiltinErrorDescriptor, JsonEncodeOptionName, JSONENCODE_COMPLEX_INPUT_EXTENSION,
    JSONENCODE_ERROR_INF_NAN, JSONENCODE_ERROR_INTERNAL, JSONENCODE_ERROR_NAME_VALUE_PAIRS,
    JSONENCODE_ERROR_OPTIONS_CONFIG, JSONENCODE_ERROR_OPTION_NAME, JSONENCODE_ERROR_OPTION_VALUE,
    JSONENCODE_ERROR_UNEXPECTED_GPU, JSONENCODE_ERROR_UNKNOWN_OPTION,
    JSONENCODE_ERROR_UNSUPPORTED_TYPE, JSONENCODE_NUMERIC_OPTION_EXTENSION,
    JSONENCODE_RESIDENT_INPUT_EXTENSION, JSONENCODE_SPARSE_INPUT_EXTENSION,
    JSONENCODE_TEXT_OPTION_EXTENSION, JSONENCODE_TYPED_INTEGER_OPTION_EXTENSION,
};
#[cfg(test)]
use runmat_builtins::{
    BuiltinIntegerInputAvailability, JSONENCODE_DESCRIPTOR, JSONENCODE_EXTENSIONS,
    JSONENCODE_INTEGER_CAPABILITIES,
};
use runmat_macros::runtime_builtin;
use runmat_value::{
    CellArray, CharArray, ComplexTensor, IntValue, IntegerStorage, LogicalArray, ObjectInstance,
    StringArray, StructValue, SymbolicArray, Tensor, Value,
};

use crate::builtins::common::tensor;
use crate::{build_runtime_error, gather_if_needed_async, BuiltinResult, RuntimeError};

const BUILTIN_NAME: &str = "jsonencode";

fn jsonencode_error(error: &'static BuiltinErrorDescriptor) -> RuntimeError {
    jsonencode_error_with(error, error.message)
}

fn jsonencode_error_with(
    error: &'static BuiltinErrorDescriptor,
    message: impl Into<String>,
) -> RuntimeError {
    let mut builder = build_runtime_error(message).with_builtin(BUILTIN_NAME);
    if let Some(identifier) = error.identifier {
        builder = builder.with_identifier(identifier);
    }
    builder.build()
}

fn jsonencode_flow_with_context(err: RuntimeError) -> RuntimeError {
    let mut builder = build_runtime_error(err.message().to_string()).with_builtin(BUILTIN_NAME);
    if let Some(identifier) = err.identifier() {
        builder = builder.with_identifier(identifier.to_string());
    }
    builder.with_source(err).build()
}

#[derive(Debug, Clone)]
struct JsonEncodeOptions {
    pretty_print: bool,
    convert_inf_and_nan: bool,
}

impl Default for JsonEncodeOptions {
    fn default() -> Self {
        Self {
            pretty_print: false,
            convert_inf_and_nan: true,
        }
    }
}

#[derive(Debug, Clone)]
enum JsonValue {
    Null,
    Bool(bool),
    Number(JsonNumber),
    String(runmat_value::Utf8Text),
    Array(Vec<JsonValue>),
    Object(Vec<(String, JsonValue)>),
}

#[derive(Debug, Clone)]
enum JsonNumber {
    Float(f64),
    I64(i64),
    U64(u64),
}

#[runtime_builtin(
    name = "jsonencode",
    context = true,
    binding_variant = "default",
    builtin_path = "crate::builtins::io::json::jsonencode"
)]
async fn jsonencode_builtin(
    runtime: Option<crate::context::RuntimeContext>,
    value: Value,
    rest: crate::call::BuiltinArguments,
) -> crate::BuiltinResult<Value> {
    let layout = prepare_option_layout(rest.view())?;
    preflight_jsonencode_extensions(&value, rest.view(), &layout)?;
    let gather = gather_if_needed_async(&value);
    let host_value = match runtime.as_ref() {
        Some(runtime) => runtime.scope(gather).await,
        None => gather.await,
    }
    .map_err(jsonencode_flow_with_context)?;
    let gather = rest.try_map_preserving_roots_async(rest.len(), runtime.as_ref(), async |index| {
        gather_if_needed_async(&rest[index])
            .await
            .map_err(jsonencode_flow_with_context)
    });
    let gathered_args = match runtime.as_ref() {
        Some(runtime) => runtime.scope(gather).await?,
        None => gather.await?,
    };

    preflight_option_extensions(&layout, gathered_args.view())?;
    let options = parse_options(&layout, gathered_args.view())?;
    let json_value = value_to_json(&host_value, &options)?;
    let json_string = render_json(&json_value, &options);

    Ok(Value::CharArray(CharArray::new_row(&json_string)))
}

#[cfg(test)]
pub(crate) mod tests;
