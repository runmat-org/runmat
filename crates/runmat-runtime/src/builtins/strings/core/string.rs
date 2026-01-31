//! MATLAB-compatible `string` builtin with GPU-aware conversion semantics for RunMat.

use runmat_builtins::{
    CharArray, ComplexTensor, IntValue, LogicalArray, StringArray, Tensor, Value,
};
use runmat_macros::runtime_builtin;

use crate::builtins::common::format::format_variadic;
use crate::builtins::common::map_control_flow_with_builtin;
use crate::builtins::common::spec::{
    BroadcastSemantics, BuiltinFusionSpec, BuiltinGpuSpec, ConstantStrategy, GpuOpKind,
    ReductionNaN, ResidencyPolicy, ShapeRequirements,
};
use crate::builtins::common::tensor;
use crate::builtins::strings::type_resolvers::string_array_type;
use crate::{build_runtime_error, gather_if_needed_async, BuiltinResult, RuntimeError};

#[runmat_macros::register_gpu_spec(builtin_path = "crate::builtins::strings::core::string")]
pub const GPU_SPEC: BuiltinGpuSpec = BuiltinGpuSpec {
    name: "string",
    op_kind: GpuOpKind::Custom("conversion"),
    supported_precisions: &[],
    broadcast: BroadcastSemantics::None,
    provider_hooks: &[],
    constant_strategy: ConstantStrategy::InlineLiteral,
    residency: ResidencyPolicy::GatherImmediately,
    nan_mode: ReductionNaN::Include,
    two_pass_threshold: None,
    workgroup_size: None,
    accepts_nan_mode: false,
    notes: "Always converts on the CPU; GPU tensors are gathered to host memory before conversion.",
};

#[runmat_macros::register_fusion_spec(builtin_path = "crate::builtins::strings::core::string")]
pub const FUSION_SPEC: BuiltinFusionSpec = BuiltinFusionSpec {
    name: "string",
    shape: ShapeRequirements::Any,
    constant_strategy: ConstantStrategy::InlineLiteral,
    elementwise: None,
    reduction: None,
    emits_nan: false,
    notes:
        "Conversion builtin; not eligible for fusion and always materialises host string arrays.",
};

#[runtime_builtin(
    name = "string",
    category = "strings/core",
    summary = "Convert numeric, logical, and text inputs into MATLAB string arrays.",
    keywords = "string,convert,text,char,gpu",
    accel = "sink",
    type_resolver(string_array_type),
    builtin_path = "crate::builtins::strings::core::string"
)]
async fn string_builtin(value: Value, rest: Vec<Value>) -> crate::BuiltinResult<Value> {
    if rest.is_empty() {
        let gathered = gather_if_needed_async(&value)
            .await
            .map_err(|flow| remap_string_flow(flow))?;
        let array = convert_to_string_array(gathered, StringEncoding::Utf8).await?;
        return Ok(Value::StringArray(array));
    }

    let mut args = rest;
    let format_value = gather_if_needed_async(&value)
        .await
        .map_err(|flow| remap_string_flow(flow))?;

    if args.len() == 1 {
        let arg = args.pop().unwrap();
        let gathered_arg = gather_if_needed_async(&arg)
            .await
            .map_err(|flow| remap_string_flow(flow))?;
        if let Some(encoding) = try_encoding_argument(&format_value, &gathered_arg)? {
            let array = convert_to_string_array(format_value, encoding).await?;
            return Ok(Value::StringArray(array));
        }
        let formatted = format_from_spec(format_value, vec![gathered_arg]).await?;
        return Ok(Value::StringArray(formatted));
    }

    let mut gathered_args = Vec::with_capacity(args.len());
    for arg in args {
        gathered_args.push(
            gather_if_needed_async(&arg)
                .await
                .map_err(|flow| remap_string_flow(flow))?,
        );
    }
    let formatted = format_from_spec(format_value, gathered_args).await?;
    Ok(Value::StringArray(formatted))
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum StringEncoding {
    Utf8,
}

fn try_encoding_argument(
    first: &Value,
    candidate: &Value,
) -> BuiltinResult<Option<StringEncoding>> {
    if !matches!(
        first,
        Value::CharArray(_) | Value::String(_) | Value::StringArray(_) | Value::Cell(_)
    ) {
        return Ok(None);
    }
    if has_format_placeholders(first) {
        return Ok(None);
    }
    if let Value::Cell(cell) = first {
        if !cell_contains_only_text_scalars(cell) {
            return Ok(None);
        }
    }
    let Some(text) = value_to_scalar_text(candidate) else {
        return Ok(None);
    };
    parse_encoding_text(&text).map(Some)
}

fn parse_encoding_text(raw: &str) -> BuiltinResult<StringEncoding> {
    let trimmed = raw.trim();
    let lowered = trimmed.to_ascii_lowercase();
    match lowered.as_str() {
        "utf-8" | "utf8" | "unicode" | "system" => Ok(StringEncoding::Utf8),
        _ => Err(string_flow(format!(
            "string: unsupported character encoding '{trimmed}'; only UTF-8 is available"
        ))),
    }
}

fn cell_contains_only_text_scalars(cell: &runmat_builtins::CellArray) -> bool {
    cell.data.iter().all(|ptr| match &**ptr {
        Value::String(_) => true,
        Value::StringArray(sa) => sa.data.len() <= 1,
        Value::CharArray(ca) => ca.rows <= 1,
        _ => false,
    })
}

fn text_has_format_placeholder(text: &str) -> bool {
    let mut chars = text.chars().peekable();
    while let Some(ch) = chars.next() {
        if ch != '%' {
            continue;
        }
        if let Some('%') = chars.peek() {
            chars.next();
            continue;
        }
        while matches!(chars.peek(), Some(flag) if matches!(flag, '+' | '-' | '0' | '#')) {
            chars.next();
        }
        while matches!(chars.peek(), Some(digit) if digit.is_ascii_digit()) {
            chars.next();
        }
        if let Some('.') = chars.peek() {
            chars.next();
            while matches!(chars.peek(), Some(digit) if digit.is_ascii_digit()) {
                chars.next();
            }
        }
        if let Some(conv) = chars.peek() {
            if conv.is_ascii_alphabetic() {
                return true;
            }
        }
    }
    false
}

fn has_format_placeholders(value: &Value) -> bool {
    match value {
        Value::String(s) => text_has_format_placeholder(s),
        Value::StringArray(sa) => sa.data.iter().any(|s| text_has_format_placeholder(s)),
        Value::CharArray(ca) => {
            for row in 0..ca.rows {
                let mut row_str = String::with_capacity(ca.cols);
                for col in 0..ca.cols {
                    row_str.push(ca.data[row * ca.cols + col]);
                }
                if text_has_format_placeholder(&row_str) {
                    return true;
                }
            }
            false
        }
        Value::Cell(cell) => {
            for ptr in &cell.data {
                let element = (**ptr).clone();
                if has_format_placeholders(&element) {
                    return true;
                }
            }
            false
        }
        _ => false,
    }
}

pub(crate) struct FormatSpecData {
    pub(crate) specs: Vec<String>,
    pub(crate) shape: Vec<usize>,
}

struct ArgumentData {
    values: Vec<Value>,
    shape: Vec<usize>,
}

fn string_flow(message: impl Into<String>) -> RuntimeError {
    build_runtime_error(message).with_builtin("string").build()
}

fn remap_string_flow(err: RuntimeError) -> RuntimeError {
    map_control_flow_with_builtin(err, "string")
}

pub(crate) async fn format_from_spec(
    format_value: Value,
    args: Vec<Value>,
) -> crate::BuiltinResult<StringArray> {
    let spec = extract_format_spec(format_value).await?;
    let mut arguments = Vec::with_capacity(args.len());
    for arg in args {
        arguments.push(extract_argument_data(arg).await?);
    }

    let (target_len, mut target_shape) = resolve_target_shape(&spec, &arguments)?;

    if target_len == 0 {
        let shape = if target_shape.is_empty() {
            if spec.shape.is_empty() {
                vec![0, 0]
            } else {
                spec.shape.clone()
            }
        } else {
            target_shape
        };
        return StringArray::new(Vec::new(), shape)
            .map_err(|e| string_flow(format!("string: {e}")));
    }

    let spec_len = spec.specs.len();
    if spec_len == 0 {
        return Err(string_flow(
            "string: formatSpec must contain at least one element when formatting with data",
        ));
    }

    for arg in &arguments {
        if target_len > 0 && arg.values.is_empty() {
            return Err(string_flow(
                "string: format data arguments must be scalars or match formatSpec size",
            ));
        }
    }

    let mut output = Vec::with_capacity(target_len);
    for idx in 0..target_len {
        let spec_idx = if spec_len == 1 { 0 } else { idx };
        let spec_str = &spec.specs[spec_idx];
        let mut per_call = Vec::with_capacity(arguments.len());
        for arg in &arguments {
            let value =
                match arg.values.len() {
                    0 => continue,
                    1 => arg.values[0].clone(),
                    len if len == target_len => arg.values[idx].clone(),
                    _ => return Err(string_flow(
                        "string: format data arguments must be scalars or match formatSpec size",
                    )),
                };
            per_call.push(value);
        }
        let formatted =
            format_variadic(spec_str, &per_call).map_err(|flow| remap_string_flow(flow))?;
        output.push(formatted);
    }

    if target_shape.is_empty() {
        target_shape = if spec_len > 1 {
            spec.shape.clone()
        } else {
            vec![target_len, 1]
        };
    }

    if tensor::element_count(&target_shape) != target_len {
        target_shape = vec![target_len, 1];
    }

    StringArray::new(output, target_shape).map_err(|e| string_flow(format!("string: {e}")))
}

fn resolve_target_shape(
    spec: &FormatSpecData,
    args: &[ArgumentData],
) -> BuiltinResult<(usize, Vec<usize>)> {
    let mut target_len = spec.specs.len();
    let mut target_shape = if target_len > 1 || (target_len == 1 && !spec.shape.is_empty()) {
        spec.shape.clone()
    } else {
        Vec::new()
    };

    for arg in args {
        let len = arg.values.len();
        if len == 0 {
            continue;
        }
        if target_len == 0 {
            target_len = len;
            target_shape = arg.shape.clone();
            continue;
        }
        if len == 1 {
            continue;
        }
        if target_len == 1 {
            target_len = len;
            target_shape = arg.shape.clone();
            continue;
        }
        if len != target_len {
            return Err(string_flow(
                "string: format data arguments must be scalars or match formatSpec size",
            ));
        }
        if target_shape.is_empty() && len > 1 {
            target_shape = arg.shape.clone();
        }
    }

    if target_len == 0 {
        let shape = if spec.shape.is_empty() {
            vec![0, 0]
        } else {
            spec.shape.clone()
        };
        return Ok((0, shape));
    }

    if target_shape.is_empty() {
        target_shape = if spec.shape.is_empty() {
            vec![target_len, 1]
        } else {
            spec.shape.clone()
        };
        if spec.specs.len() == 1 && tensor::element_count(&target_shape) != target_len {
            target_shape = vec![target_len, 1];
        }
    }

    if tensor::element_count(&target_shape) != target_len {
        target_shape = vec![target_len, 1];
    }

    Ok((target_len, target_shape))
}

pub(crate) async fn extract_format_spec(value: Value) -> BuiltinResult<FormatSpecData> {
    match value {
        Value::String(s) => Ok(FormatSpecData {
            specs: vec![s],
            shape: vec![1, 1],
        }),
        Value::StringArray(sa) => Ok(FormatSpecData {
            specs: sa.data.clone(),
            shape: sa.shape.clone(),
        }),
        Value::CharArray(ca) => {
            let array = char_array_to_string_array(ca, StringEncoding::Utf8)?;
            Ok(FormatSpecData {
                specs: array.data,
                shape: array.shape,
            })
        }
        Value::Cell(cell) => {
            let mut specs = Vec::with_capacity(cell.data.len());
            for col in 0..cell.cols {
                for row in 0..cell.rows {
                    let idx = row * cell.cols + col;
                    let element = &cell.data[idx];
                    let value = (**element).clone();
                    let gathered = gather_if_needed_async(&value)
                        .await
                        .map_err(|flow| remap_string_flow(flow))?;
                    let text = value_to_scalar_text(&gathered).ok_or_else(|| {
                        string_flow("string: formatSpec cell elements must be text scalars")
                    })?;
                    specs.push(text);
                }
            }
            Ok(FormatSpecData {
                specs,
                shape: vec![cell.rows, cell.cols],
            })
        }
        _ => Err(string_flow(
            "string: formatSpec must be text (string, char, or cellstr)",
        )),
    }
}

#[async_recursion::async_recursion(?Send)]
async fn extract_argument_data(value: Value) -> BuiltinResult<ArgumentData> {
    match value {
        Value::String(s) => Ok(ArgumentData {
            values: vec![Value::String(s)],
            shape: vec![1, 1],
        }),
        Value::StringArray(sa) => Ok(ArgumentData {
            values: sa.data.into_iter().map(Value::String).collect(),
            shape: sa.shape,
        }),
        Value::CharArray(ca) => {
            let array = char_array_to_string_array(ca, StringEncoding::Utf8)?;
            Ok(ArgumentData {
                values: array.data.into_iter().map(Value::String).collect(),
                shape: array.shape,
            })
        }
        Value::Num(n) => Ok(ArgumentData {
            values: vec![Value::Num(n)],
            shape: vec![1, 1],
        }),
        Value::Int(i) => Ok(ArgumentData {
            values: vec![Value::Int(i)],
            shape: vec![1, 1],
        }),
        Value::Bool(b) => Ok(ArgumentData {
            values: vec![Value::Num(if b { 1.0 } else { 0.0 })],
            shape: vec![1, 1],
        }),
        Value::Tensor(t) => Ok(ArgumentData {
            values: t.data.into_iter().map(Value::Num).collect(),
            shape: t.shape,
        }),
        Value::Complex(re, im) => Ok(ArgumentData {
            values: vec![Value::String(Value::Complex(re, im).to_string())],
            shape: vec![1, 1],
        }),
        Value::ComplexTensor(t) => Ok(ArgumentData {
            values: t
                .data
                .into_iter()
                .map(|(re, im)| Value::String(Value::Complex(re, im).to_string()))
                .collect(),
            shape: t.shape,
        }),
        Value::LogicalArray(la) => Ok(ArgumentData {
            values: la
                .data
                .into_iter()
                .map(|byte| Value::Num(if byte != 0 { 1.0 } else { 0.0 }))
                .collect(),
            shape: la.shape,
        }),
        Value::Cell(cell) => {
            let mut values = Vec::with_capacity(cell.data.len());
            for col in 0..cell.cols {
                for row in 0..cell.rows {
                    let idx = row * cell.cols + col;
                    let element = &cell.data[idx];
                    let value = (**element).clone();
                    let gathered = gather_if_needed_async(&value)
                        .await
                        .map_err(|flow| remap_string_flow(flow))?;
                    let value = match gathered {
                        Value::String(s) => Value::String(s),
                        Value::StringArray(sa) if sa.data.len() == 1 => {
                            Value::String(sa.data[0].clone())
                        }
                        Value::CharArray(ca) => {
                            if ca.rows != 1 {
                                return Err(string_flow(
                                    "string: cell format arguments must contain char row vectors",
                                ));
                            }
                            let mut row_str = String::with_capacity(ca.cols);
                            for ch in ca.data {
                                row_str.push(ch);
                            }
                            Value::String(row_str)
                        }
                        Value::Num(n) => Value::Num(n),
                        Value::Int(i) => Value::Int(i),
                        Value::Bool(b) => Value::Num(if b { 1.0 } else { 0.0 }),
                        Value::Tensor(t) => {
                            if t.data.len() != 1 {
                                return Err(string_flow(
                                    "string: cell format arguments must contain scalar values",
                                ));
                            }
                            Value::Num(t.data[0])
                        }
                        Value::LogicalArray(la) => {
                            if la.data.len() != 1 {
                                return Err(string_flow(
                                    "string: cell format arguments must contain scalar values",
                                ));
                            }
                            Value::Num(if la.data[0] != 0 { 1.0 } else { 0.0 })
                        }
                        Value::Complex(re, im) => {
                            Value::String(Value::Complex(re, im).to_string())
                        }
                        Value::ComplexTensor(t) => {
                            if t.data.len() != 1 {
                                return Err(string_flow(
                                    "string: cell format arguments must contain scalar values",
                                ));
                            }
                            let (re, im) = t.data[0];
                            Value::String(Value::Complex(re, im).to_string())
                        }
                        other => {
                            return Err(string_flow(format!(
                                "string: unsupported cell format argument {other:?}; expected scalar text or numeric values"
                            )))
                        }
                    };
                    values.push(value);
                }
            }
            Ok(ArgumentData {
                values,
                shape: vec![cell.rows, cell.cols],
            })
        }
        Value::GpuTensor(handle) => {
            let gathered = gather_if_needed_async(&Value::GpuTensor(handle))
                .await
                .map_err(|flow| remap_string_flow(flow))?;
            extract_argument_data(gathered).await
        }
        Value::MException(_)
        | Value::HandleObject(_)
        | Value::Object(_)
        | Value::Listener(_)
        | Value::Struct(_) => Err(string_flow("string: unsupported format argument type")),
        Value::FunctionHandle(_) | Value::Closure(_) | Value::ClassRef(_) => {
            Err(string_flow("string: unsupported format argument type"))
        }
    }
}

#[async_recursion::async_recursion(?Send)]
async fn convert_to_string_array(
    value: Value,
    encoding: StringEncoding,
) -> BuiltinResult<StringArray> {
    match value {
        Value::String(s) => string_scalar(s),
        Value::StringArray(sa) => Ok(sa),
        Value::CharArray(ca) => char_array_to_string_array(ca, encoding),
        Value::Tensor(tensor) => tensor_to_string_array(tensor),
        Value::ComplexTensor(tensor) => complex_tensor_to_string_array(tensor),
        Value::LogicalArray(logical) => logical_array_to_string_array(logical),
        Value::Cell(cell) => cell_array_to_string_array(cell, encoding).await,
        Value::Num(n) => string_scalar(Value::Num(n).to_string()),
        Value::Int(i) => string_scalar(int_value_to_string(&i)),
        Value::Bool(b) => string_scalar(bool_to_string(b).to_string()),
        Value::Complex(re, im) => string_scalar(Value::Complex(re, im).to_string()),
        Value::GpuTensor(handle) => {
            // Defensive fallback: gather and retry.
            let gathered = gather_if_needed_async(&Value::GpuTensor(handle))
                .await
                .map_err(|flow| remap_string_flow(flow))?;
            convert_to_string_array(gathered, encoding).await
        }
        Value::Object(_) | Value::HandleObject(_) | Value::Listener(_) => Err(string_flow(
            "string: unsupported conversion from handle-based objects. Use class-specific formatters.",
        )),
        Value::Struct(_) => Err(string_flow(
            "string: structs are not supported for automatic conversion",
        )),
        Value::FunctionHandle(_) | Value::Closure(_) | Value::ClassRef(_) | Value::MException(_) => Err(
            string_flow("string: unsupported conversion for function or exception handles"),
        ),
    }
}

fn string_scalar<S: Into<String>>(text: S) -> BuiltinResult<StringArray> {
    StringArray::new(vec![text.into()], vec![1, 1]).map_err(|e| string_flow(format!("string: {e}")))
}

fn value_to_scalar_text(value: &Value) -> Option<String> {
    match value {
        Value::String(s) => Some(s.clone()),
        Value::StringArray(sa) if sa.data.len() == 1 => Some(sa.data[0].clone()),
        Value::CharArray(ca) if ca.rows == 1 => Some(ca.data.iter().collect()),
        _ => None,
    }
}

fn char_array_to_string_array(
    array: CharArray,
    _encoding: StringEncoding,
) -> BuiltinResult<StringArray> {
    let mut rows: Vec<String> = Vec::with_capacity(array.rows);
    for r in 0..array.rows {
        let mut row = String::with_capacity(array.cols);
        for c in 0..array.cols {
            row.push(array.data[r * array.cols + c]);
        }
        rows.push(row);
    }
    let shape = if array.rows == 0 {
        vec![0, 1]
    } else {
        vec![array.rows, 1]
    };
    StringArray::new(rows, shape).map_err(|e| string_flow(format!("string: {e}")))
}

fn tensor_to_string_array(tensor: Tensor) -> BuiltinResult<StringArray> {
    let mut strings = Vec::with_capacity(tensor.data.len());
    for &value in &tensor.data {
        strings.push(Value::Num(value).to_string());
    }
    StringArray::new(strings, tensor.shape).map_err(|e| string_flow(format!("string: {e}")))
}

fn complex_tensor_to_string_array(tensor: ComplexTensor) -> BuiltinResult<StringArray> {
    let mut strings = Vec::with_capacity(tensor.data.len());
    for &(re, im) in &tensor.data {
        strings.push(Value::Complex(re, im).to_string());
    }
    StringArray::new(strings, tensor.shape).map_err(|e| string_flow(format!("string: {e}")))
}

fn logical_array_to_string_array(logical: LogicalArray) -> BuiltinResult<StringArray> {
    let mut strings = Vec::with_capacity(logical.data.len());
    for &byte in &logical.data {
        strings.push(bool_to_string(byte != 0).to_string());
    }
    StringArray::new(strings, logical.shape).map_err(|e| string_flow(format!("string: {e}")))
}

async fn cell_array_to_string_array(
    cell: runmat_builtins::CellArray,
    _encoding: StringEncoding,
) -> BuiltinResult<StringArray> {
    let mut strings = Vec::with_capacity(cell.data.len());
    for col in 0..cell.cols {
        for row in 0..cell.rows {
            let idx = row * cell.cols + col;
            let element = &cell.data[idx];
            let value = (**element).clone();
            let gathered = gather_if_needed_async(&value)
                .await
                .map_err(|flow| remap_string_flow(flow))?;
            strings.push(cell_element_to_string(&gathered)?);
        }
    }
    StringArray::new(strings, vec![cell.rows, cell.cols])
        .map_err(|e| string_flow(format!("string: {e}")))
}

fn cell_element_to_string(value: &Value) -> BuiltinResult<String> {
    match value {
        Value::String(s) => Ok(s.clone()),
        Value::StringArray(sa) => {
            if sa.data.len() == 1 {
                Ok(sa.data[0].clone())
            } else {
                Err(string_flow(
                    "string: cell elements must contain string scalars, not string arrays",
                ))
            }
        }
        Value::CharArray(ca) => {
            if ca.rows == 1 {
                Ok(ca.data.iter().collect())
            } else {
                Err(string_flow(
                    "string: cell character arrays must be row vectors",
                ))
            }
        }
        Value::Num(n) => Ok(Value::Num(*n).to_string()),
        Value::Int(i) => Ok(int_value_to_string(i)),
        Value::Bool(b) => Ok(bool_to_string(*b).to_string()),
        Value::LogicalArray(array) => {
            if array.data.len() == 1 {
                Ok(bool_to_string(array.data[0] != 0).to_string())
            } else {
                Err(string_flow("string: cell logical values must be scalar"))
            }
        }
        Value::Tensor(t) => {
            if t.data.len() == 1 {
                Ok(Value::Num(t.data[0]).to_string())
            } else {
                Err(string_flow("string: cell numeric values must be scalar"))
            }
        }
        Value::Complex(re, im) => Ok(Value::Complex(*re, *im).to_string()),
        Value::ComplexTensor(t) => {
            if t.data.len() == 1 {
                let (re, im) = t.data[0];
                Ok(Value::Complex(re, im).to_string())
            } else {
                Err(string_flow("string: cell complex values must be scalar"))
            }
        }
        other => Err(string_flow(format!(
            "string: unsupported cell element type {:?}; expected text or scalar values",
            other
        ))),
    }
}

fn bool_to_string(value: bool) -> &'static str {
    if value {
        "true"
    } else {
        "false"
    }
}

fn int_value_to_string(value: &IntValue) -> String {
    match value {
        IntValue::I8(v) => v.to_string(),
        IntValue::I16(v) => v.to_string(),
        IntValue::I32(v) => v.to_string(),
        IntValue::I64(v) => v.to_string(),
        IntValue::U8(v) => v.to_string(),
        IntValue::U16(v) => v.to_string(),
        IntValue::U32(v) => v.to_string(),
        IntValue::U64(v) => v.to_string(),
    }
}

#[cfg(test)]
pub(crate) mod tests {
    use super::*;
    use crate::builtins::common::test_support;
    use runmat_builtins::{CellArray, IntValue, StringArray, StructValue, Type};

    fn string_builtin(value: Value, rest: Vec<Value>) -> BuiltinResult<Value> {
        futures::executor::block_on(super::string_builtin(value, rest))
    }

    fn error_message(err: crate::RuntimeError) -> String {
        err.message().to_string()
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn string_from_numeric_scalar() {
        let out = string_builtin(Value::Num(42.0), Vec::new()).expect("string");
        match out {
            Value::StringArray(sa) => {
                assert_eq!(sa.shape, vec![1, 1]);
                assert_eq!(sa.data, vec!["42".to_string()]);
            }
            other => panic!("expected string array, got {other:?}"),
        }
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn string_from_numeric_tensor_preserves_shape() {
        let tensor = Tensor::new(vec![1.0, 2.0, 3.0, 4.0], vec![2, 2]).unwrap();
        let out = string_builtin(Value::Tensor(tensor), Vec::new()).expect("string");
        match out {
            Value::StringArray(sa) => {
                assert_eq!(sa.shape, vec![2, 2]);
                assert_eq!(sa.data, vec!["1", "2", "3", "4"]);
            }
            other => panic!("expected string array, got {other:?}"),
        }
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn string_from_logical_array_uses_boolean_text() {
        let logical = LogicalArray::new(vec![1, 0, 1], vec![1, 3]).unwrap();
        let out = string_builtin(Value::LogicalArray(logical), Vec::new()).expect("string");
        match out {
            Value::StringArray(sa) => {
                assert_eq!(sa.shape, vec![1, 3]);
                assert_eq!(sa.data, vec!["true", "false", "true"]);
            }
            other => panic!("expected string array, got {other:?}"),
        }
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn string_from_char_array_produces_column_vector() {
        let chars = CharArray::new("abc".chars().collect(), 1, 3).unwrap();
        let out = string_builtin(Value::CharArray(chars), Vec::new()).expect("string");
        match out {
            Value::StringArray(sa) => {
                assert_eq!(sa.shape, vec![1, 1]);
                assert_eq!(sa.data, vec!["abc"]);
            }
            other => panic!("expected string array, got {other:?}"),
        }
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn string_from_cell_array() {
        let cell = CellArray::new(vec![Value::Bool(true), Value::Int(IntValue::I32(7))], 1, 2)
            .expect("cell array");
        let out = string_builtin(Value::Cell(cell), Vec::new()).expect("string");
        match out {
            Value::StringArray(sa) => {
                assert_eq!(sa.shape, vec![1, 2]);
                assert_eq!(sa.data, vec!["true", "7"]);
            }
            other => panic!("expected string array, got {other:?}"),
        }
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn string_from_cell_array_column_major() {
        let cell = CellArray::new(
            vec![
                Value::Int(IntValue::I32(1)),
                Value::Int(IntValue::I32(2)),
                Value::Int(IntValue::I32(3)),
                Value::Int(IntValue::I32(4)),
            ],
            2,
            2,
        )
        .expect("cell array");
        let out = string_builtin(Value::Cell(cell), Vec::new()).expect("string");
        match out {
            Value::StringArray(sa) => {
                assert_eq!(sa.shape, vec![2, 2]);
                assert_eq!(sa.data, vec!["1", "3", "2", "4"]);
            }
            other => panic!("expected string array, got {other:?}"),
        }
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn string_cell_element_requires_scalar_numeric() {
        let tensor = Tensor::new(vec![1.0, 2.0], vec![2, 1]).unwrap();
        let cell =
            CellArray::new(vec![Value::Tensor(tensor)], 1, 1).expect("cell with numeric tensor");
        let err = error_message(string_builtin(Value::Cell(cell), Vec::new()).unwrap_err());
        assert!(err.contains("cell numeric values must be scalar"));
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn string_rejects_struct_input() {
        let err = error_message(
            string_builtin(Value::Struct(StructValue::new()), Vec::new()).expect_err("string"),
        );
        assert!(err.contains("structs are not supported"));
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn string_errors_on_unsupported_encoding() {
        let err = error_message(
            string_builtin(
                Value::CharArray(CharArray::new_row("abc")),
                vec![Value::from("UTF-16")],
            )
            .unwrap_err(),
        );
        assert!(
            err.contains("unsupported character encoding"),
            "unexpected error message: {err}"
        );
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn string_accepts_system_encoding_alias() {
        let out = string_builtin(
            Value::CharArray(CharArray::new_row("hello")),
            vec![Value::from("system")],
        )
        .expect("string");
        match out {
            Value::StringArray(sa) => {
                assert_eq!(sa.shape, vec![1, 1]);
                assert_eq!(sa.data, vec!["hello"]);
            }
            other => panic!("expected string array, got {other:?}"),
        }
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn string_encoding_allows_percent_literal() {
        let out = string_builtin(
            Value::CharArray(CharArray::new_row("100% Done")),
            vec![Value::from("utf8")],
        )
        .expect("string");
        match out {
            Value::StringArray(sa) => {
                assert_eq!(sa.shape, vec![1, 1]);
                assert_eq!(sa.data, vec!["100% Done"]);
            }
            other => panic!("expected string array, got {other:?}"),
        }
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn string_format_spec_cell_requires_text_scalars() {
        let cell = CellArray::new(vec![Value::Num(1.0)], 1, 1).expect("cell");
        let err = error_message(
            string_builtin(Value::Cell(cell), vec![Value::from("data")]).expect_err("string"),
        );
        assert!(
            err.contains("formatSpec cell elements must be text scalars"),
            "unexpected error: {err}"
        );
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn string_format_cell_argument_requires_scalar_values() {
        let tensor = Tensor::new(vec![1.0, 2.0], vec![2, 1]).unwrap();
        let cell = CellArray::new(vec![Value::Tensor(tensor)], 1, 1).expect("cell argument values");
        let err = error_message(
            string_builtin(Value::from("%d"), vec![Value::Cell(cell)]).expect_err("string"),
        );
        assert!(err.contains("cell format arguments must contain scalar values"));
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn string_handles_large_unsigned_int() {
        let value = Value::Int(IntValue::U64(u64::MAX));
        let out = string_builtin(value, Vec::new()).expect("string");
        match out {
            Value::StringArray(sa) => {
                assert_eq!(sa.shape, vec![1, 1]);
                assert_eq!(sa.data, vec![u64::MAX.to_string()]);
            }
            other => panic!("expected string array, got {other:?}"),
        }
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn string_format_numeric_scalar() {
        let out = string_builtin(Value::from("%d"), vec![Value::Num(7.0)]).expect("string");
        match out {
            Value::StringArray(sa) => {
                assert_eq!(sa.shape, vec![1, 1]);
                assert_eq!(sa.data, vec!["7"]);
            }
            other => panic!("expected string array, got {other:?}"),
        }
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn string_format_broadcast_over_tensor() {
        let tensor = Tensor::new(vec![1.0, 2.0, 3.0], vec![1, 3]).unwrap();
        let out =
            string_builtin(Value::from("Trial %d"), vec![Value::Tensor(tensor)]).expect("string");
        match out {
            Value::StringArray(sa) => {
                assert_eq!(sa.shape, vec![1, 3]);
                assert_eq!(sa.data, vec!["Trial 1", "Trial 2", "Trial 3"]);
            }
            other => panic!("expected string array, got {other:?}"),
        }
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn string_format_string_array_spec_alignment() {
        let spec = StringArray::new(vec!["[%d]".into(), "Value %d".into()], vec![1, 2]).unwrap();
        let tensor = Tensor::new(vec![5.0, 6.0], vec![1, 2]).unwrap();
        let out =
            string_builtin(Value::StringArray(spec), vec![Value::Tensor(tensor)]).expect("string");
        match out {
            Value::StringArray(sa) => {
                assert_eq!(sa.shape, vec![1, 2]);
                assert_eq!(sa.data, vec!["[5]", "Value 6"]);
            }
            other => panic!("expected string array, got {other:?}"),
        }
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn string_format_prefers_placeholders_over_encoding_hint() {
        let out = string_builtin(Value::from("%s"), vec![Value::from("UTF-8")]).expect("string");
        match out {
            Value::StringArray(sa) => {
                assert_eq!(sa.shape, vec![1, 1]);
                assert_eq!(sa.data, vec!["UTF-8"]);
            }
            other => panic!("expected string array, got {other:?}"),
        }
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn string_format_mismatched_lengths_errors() {
        let spec = StringArray::new(vec!["%d".into(), "%d".into()], vec![2, 1]).unwrap();
        let tensor = Tensor::new(vec![1.0, 2.0, 3.0], vec![3, 1]).unwrap();
        let err = error_message(
            string_builtin(Value::StringArray(spec), vec![Value::Tensor(tensor)]).unwrap_err(),
        );
        assert!(err.contains("must be scalars or match formatSpec size"));
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn string_gpu_numeric_tensor() {
        test_support::with_test_provider(|provider| {
            let tensor = Tensor::new(vec![10.0, 20.0], vec![1, 2]).unwrap();
            let view = runmat_accelerate_api::HostTensorView {
                data: &tensor.data,
                shape: &tensor.shape,
            };
            let handle = provider.upload(&view).expect("upload");
            let result = string_builtin(Value::GpuTensor(handle), Vec::new())
                .expect("gpu string conversion");
            match result {
                Value::StringArray(sa) => {
                    assert_eq!(sa.shape, vec![1, 2]);
                    assert_eq!(sa.data, vec!["10", "20"]);
                }
                other => panic!("expected string array, got {other:?}"),
            }
        });
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    #[cfg(feature = "wgpu")]
    fn string_wgpu_numeric_tensor_matches_cpu() {
        let _ = runmat_accelerate::backend::wgpu::provider::register_wgpu_provider(
            runmat_accelerate::backend::wgpu::provider::WgpuProviderOptions::default(),
        );
        let tensor = Tensor::new(vec![4.0, 5.0, 6.0], vec![1, 3]).unwrap();
        let cpu = string_builtin(Value::Tensor(tensor.clone()), Vec::new())
            .expect("cpu string conversion");
        let view = runmat_accelerate_api::HostTensorView {
            data: &tensor.data,
            shape: &tensor.shape,
        };
        let handle = runmat_accelerate_api::provider()
            .unwrap()
            .upload(&view)
            .expect("gpu upload");
        let gpu =
            string_builtin(Value::GpuTensor(handle), Vec::new()).expect("gpu string conversion");
        match (cpu, gpu) {
            (Value::StringArray(expect), Value::StringArray(actual)) => {
                assert_eq!(actual.shape, expect.shape);
                assert_eq!(actual.data, expect.data);
            }
            other => panic!("unexpected results {other:?}"),
        }
    }

    #[test]
    fn string_type_is_string_array() {
        assert_eq!(string_array_type(&[Type::Num]), Type::cell_of(Type::String));
    }
}
