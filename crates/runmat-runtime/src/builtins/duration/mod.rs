use std::collections::HashMap;
use std::sync::OnceLock;

use runmat_builtins::{
    Access, CharArray, ClassDef, MethodDef, ObjectInstance, PropertyDef, StringArray, Tensor, Value,
};

use crate::builtins::common::tensor;
use crate::{build_runtime_error, gather_if_needed_async, BuiltinResult, RuntimeError};

const BUILTIN_NAME: &str = "duration";
const DURATION_CLASS: &str = "duration";
const DAYS_FIELD: &str = "__days";
const FORMAT_FIELD: &str = "Format";
pub(crate) const DEFAULT_DURATION_FORMAT: &str = "hh:mm:ss";
const SECONDS_PER_DAY: f64 = 86_400.0;

static DURATION_CLASS_REGISTERED: OnceLock<()> = OnceLock::new();

fn duration_error(message: impl Into<String>) -> RuntimeError {
    build_runtime_error(message)
        .with_builtin(BUILTIN_NAME)
        .build()
}

fn ensure_duration_class_registered() {
    DURATION_CLASS_REGISTERED.get_or_init(|| {
        let mut properties = HashMap::new();
        properties.insert(
            FORMAT_FIELD.to_string(),
            PropertyDef {
                name: FORMAT_FIELD.to_string(),
                is_static: false,
                is_dependent: false,
                get_access: Access::Public,
                set_access: Access::Public,
                default_value: Some(Value::String(DEFAULT_DURATION_FORMAT.to_string())),
            },
        );

        let mut methods = HashMap::new();
        for name in [
            "subsref", "subsasgn", "plus", "minus", "eq", "ne", "lt", "le", "gt", "ge",
        ] {
            methods.insert(
                name.to_string(),
                MethodDef {
                    name: name.to_string(),
                    is_static: false,
                    access: Access::Public,
                    function_name: format!("{DURATION_CLASS}.{name}"),
                },
            );
        }

        runmat_builtins::register_class(ClassDef {
            name: DURATION_CLASS.to_string(),
            parent: None,
            properties,
            methods,
        });
    });
}

pub fn is_duration_object(value: &Value) -> bool {
    matches!(value, Value::Object(obj) if obj.is_class(DURATION_CLASS))
}

async fn gather_args(args: &[Value]) -> BuiltinResult<Vec<Value>> {
    let mut out = Vec::with_capacity(args.len());
    for arg in args {
        out.push(
            gather_if_needed_async(arg)
                .await
                .map_err(|err| duration_error(format!("duration: {}", err.message())))?,
        );
    }
    Ok(out)
}

fn scalar_text(value: &Value, context: &str) -> BuiltinResult<String> {
    match value {
        Value::String(text) => Ok(text.clone()),
        Value::StringArray(array) if array.data.len() == 1 => Ok(array.data[0].clone()),
        Value::CharArray(array) if array.rows == 1 => Ok(array.data.iter().collect()),
        _ => Err(duration_error(format!(
            "duration: {context} must be a string scalar or character vector"
        ))),
    }
}

fn parse_trailing_format(args: &[Value]) -> BuiltinResult<(usize, Option<String>)> {
    let mut positional_end = args.len();
    let mut format = None;

    while positional_end >= 2 {
        let name = match scalar_text(&args[positional_end - 2], "option name") {
            Ok(text) => text,
            Err(_) => break,
        };
        if !name.trim().eq_ignore_ascii_case("format") {
            break;
        }
        format = Some(scalar_text(&args[positional_end - 1], "Format option")?);
        positional_end -= 2;
    }

    Ok((positional_end, format))
}

fn tensor_from_numeric(value: Value, context: &str) -> BuiltinResult<Tensor> {
    tensor::value_into_tensor_for(context, value)
        .map_err(|message| duration_error(format!("duration: {message}")))
}

fn default_shape_for(shape: &[usize], len: usize) -> Vec<usize> {
    if len == 0 {
        vec![0, 1]
    } else if shape.is_empty() {
        vec![1, 1]
    } else {
        shape.to_vec()
    }
}

fn component_tensor(value: Value, context: &str) -> BuiltinResult<Tensor> {
    let tensor = tensor_from_numeric(value, context)?;
    Tensor::new(
        tensor.data.clone(),
        default_shape_for(&tensor.shape, tensor.data.len()),
    )
    .map_err(|err| duration_error(format!("duration: {err}")))
}

fn format_for_object(obj: &ObjectInstance) -> String {
    match obj.properties.get(FORMAT_FIELD) {
        Some(Value::String(text)) => text.clone(),
        Some(Value::StringArray(array)) if array.data.len() == 1 => array.data[0].clone(),
        Some(Value::CharArray(array)) if array.rows == 1 => array.data.iter().collect(),
        _ => DEFAULT_DURATION_FORMAT.to_string(),
    }
}

pub(crate) fn duration_tensor_from_duration_value(value: &Value) -> BuiltinResult<Tensor> {
    match value {
        Value::Object(obj) if obj.is_class(DURATION_CLASS) => {
            match obj.properties.get(DAYS_FIELD) {
                Some(Value::Tensor(tensor)) => Ok(tensor.clone()),
                Some(Value::Num(value)) => Tensor::new(vec![*value], vec![1, 1])
                    .map_err(|err| duration_error(format!("duration: {err}"))),
                Some(other) => Err(duration_error(format!(
                    "duration: invalid internal day storage {other:?}"
                ))),
                None => Err(duration_error("duration: missing internal day storage")),
            }
        }
        _ => Err(duration_error("duration: expected a duration value")),
    }
}

pub(crate) fn duration_format_from_value(value: &Value) -> String {
    match value {
        Value::Object(obj) if obj.is_class(DURATION_CLASS) => format_for_object(obj),
        _ => DEFAULT_DURATION_FORMAT.to_string(),
    }
}

pub(crate) fn duration_object_from_days_tensor(
    days: Tensor,
    format: impl Into<String>,
) -> BuiltinResult<Value> {
    ensure_duration_class_registered();
    let mut object = ObjectInstance::new(DURATION_CLASS.to_string());
    object
        .properties
        .insert(DAYS_FIELD.to_string(), Value::Tensor(days));
    object
        .properties
        .insert(FORMAT_FIELD.to_string(), Value::String(format.into()));
    Ok(Value::Object(object))
}

fn duration_object_from_days(
    days: Vec<f64>,
    shape: Vec<usize>,
    format: impl Into<String>,
) -> BuiltinResult<Value> {
    let tensor =
        Tensor::new(days, shape).map_err(|err| duration_error(format!("duration: {err}")))?;
    duration_object_from_days_tensor(tensor, format)
}

fn broadcast_component_data(
    arrays: &[Tensor],
    labels: &[&str],
) -> BuiltinResult<(Vec<Vec<f64>>, Vec<usize>)> {
    let mut target_shape = vec![1, 1];
    let mut target_len = 1usize;

    for array in arrays {
        let len = array.data.len();
        if len > 1 {
            let shape = default_shape_for(&array.shape, len);
            if target_len == 1 {
                target_len = len;
                target_shape = shape;
            } else if len != target_len || shape != target_shape {
                return Err(duration_error(
                    "duration: non-scalar component inputs must have matching sizes",
                ));
            }
        }
    }

    let mut broadcasted = Vec::with_capacity(arrays.len());
    for (idx, array) in arrays.iter().enumerate() {
        if array.data.len() == 1 {
            broadcasted.push(vec![array.data[0]; target_len]);
        } else if array.data.len() == target_len {
            broadcasted.push(array.data.clone());
        } else {
            return Err(duration_error(format!(
                "duration: {} input size does not match the other components",
                labels[idx]
            )));
        }
    }

    Ok((broadcasted, target_shape))
}

fn build_from_components(args: Vec<Value>, format: Option<String>) -> BuiltinResult<Value> {
    let labels = ["hours", "minutes", "seconds"];
    let mut arrays = Vec::with_capacity(args.len());
    for (idx, arg) in args.into_iter().enumerate() {
        arrays.push(component_tensor(arg, labels[idx])?);
    }
    while arrays.len() < 3 {
        arrays.push(Tensor::new(vec![0.0], vec![1, 1]).unwrap());
    }

    let (broadcasted, shape) = broadcast_component_data(&arrays, &labels)?;
    let len = broadcasted[0].len();
    let mut days = Vec::with_capacity(len);
    for idx in 0..len {
        let total_seconds =
            broadcasted[0][idx] * 3600.0 + broadcasted[1][idx] * 60.0 + broadcasted[2][idx];
        if !total_seconds.is_finite() {
            return Err(duration_error("duration: component values must be finite"));
        }
        days.push(total_seconds / SECONDS_PER_DAY);
    }

    duration_object_from_days(
        days,
        shape,
        format.unwrap_or_else(|| DEFAULT_DURATION_FORMAT.to_string()),
    )
}

fn binary_numeric_tensors(
    lhs: &Tensor,
    rhs: &Tensor,
    context: &str,
) -> BuiltinResult<(Vec<f64>, Vec<f64>, Vec<usize>)> {
    let lhs_shape = default_shape_for(&lhs.shape, lhs.data.len());
    let rhs_shape = default_shape_for(&rhs.shape, rhs.data.len());
    match (lhs.data.len(), rhs.data.len()) {
        (1, 1) => Ok((vec![lhs.data[0]], vec![rhs.data[0]], vec![1, 1])),
        (1, len) => Ok((vec![lhs.data[0]; len], rhs.data.clone(), rhs_shape)),
        (len, 1) => Ok((lhs.data.clone(), vec![rhs.data[0]; len], lhs_shape)),
        (left, right) if left == right && lhs_shape == rhs_shape => {
            Ok((lhs.data.clone(), rhs.data.clone(), lhs_shape))
        }
        _ => Err(duration_error(format!(
            "{context}: operands must be scalar or have matching sizes"
        ))),
    }
}

fn format_seconds_field(seconds: f64) -> String {
    let whole = seconds.floor();
    let fractional = seconds - whole;
    if fractional.abs() <= 1e-9 {
        format!("{:02}", whole as i64)
    } else {
        let mut text = format!("{:06.3}", seconds);
        while text.contains('.') && text.ends_with('0') {
            text.pop();
        }
        if text.ends_with('.') {
            text.pop();
        }
        text
    }
}

fn format_duration_value(days: f64, format: &str) -> BuiltinResult<String> {
    if !days.is_finite() {
        return Err(duration_error("duration: values must be finite"));
    }

    let total_seconds = days * SECONDS_PER_DAY;
    let sign = if total_seconds < 0.0 { "-" } else { "" };
    let total_seconds = total_seconds.abs();
    let total_hours = (total_seconds / 3600.0).floor();
    let total_minutes = (total_seconds / 60.0).floor();
    let hours = total_hours as i64;
    let minutes_component = ((total_seconds / 60.0).floor() as i64) % 60;
    let seconds_component =
        total_seconds - (hours as f64 * 3600.0) - (minutes_component as f64 * 60.0);

    let rendered = match format {
        "hh:mm:ss" => format!(
            "{sign}{hours:02}:{minutes_component:02}:{}",
            format_seconds_field(seconds_component)
        ),
        "hh:mm" => format!("{sign}{hours:02}:{minutes_component:02}"),
        "mm:ss" => format!(
            "{sign}{:02}:{}",
            total_minutes as i64,
            format_seconds_field(total_seconds - total_minutes * 60.0)
        ),
        "s" | "ss" => {
            let mut text = format!("{:.3}", total_seconds);
            while text.contains('.') && text.ends_with('0') {
                text.pop();
            }
            if text.ends_with('.') {
                text.pop();
            }
            format!("{sign}{text}")
        }
        other => {
            return Err(duration_error(format!(
                "duration: unsupported Format value '{other}'"
            )))
        }
    };

    Ok(rendered)
}

pub fn duration_string_array(value: &Value) -> BuiltinResult<Option<StringArray>> {
    let Value::Object(obj) = value else {
        return Ok(None);
    };
    if !obj.is_class(DURATION_CLASS) {
        return Ok(None);
    }
    let days = duration_tensor_from_duration_value(value)?;
    let format = format_for_object(obj);
    let mut strings = Vec::with_capacity(days.data.len());
    for value in &days.data {
        strings.push(format_duration_value(*value, &format)?);
    }
    let shape = default_shape_for(&days.shape, days.data.len());
    let array = StringArray::new(strings, shape)
        .map_err(|err| duration_error(format!("duration: {err}")))?;
    Ok(Some(array))
}

pub fn duration_display_text(value: &Value) -> BuiltinResult<Option<String>> {
    let Some(array) = duration_string_array(value)? else {
        return Ok(None);
    };
    if array.data.len() == 1 {
        return Ok(Some(array.data[0].clone()));
    }

    let rows = array.rows;
    let cols = array.cols;
    let mut widths = vec![0usize; cols];
    for col in 0..cols {
        for row in 0..rows {
            let idx = row + col * rows;
            widths[col] = widths[col].max(array.data[idx].len());
        }
    }

    let mut lines = Vec::with_capacity(rows);
    for row in 0..rows {
        let mut line = String::new();
        for col in 0..cols {
            if col > 0 {
                line.push_str("  ");
            }
            let idx = row + col * rows;
            let text = &array.data[idx];
            line.push_str(text);
            let padding = widths[col].saturating_sub(text.len());
            if padding > 0 {
                line.push_str(&" ".repeat(padding));
            }
        }
        lines.push(line);
    }

    Ok(Some(lines.join("\n")))
}

pub fn duration_summary(value: &Value) -> BuiltinResult<Option<String>> {
    let Value::Object(obj) = value else {
        return Ok(None);
    };
    if !obj.is_class(DURATION_CLASS) {
        return Ok(None);
    }
    let days = duration_tensor_from_duration_value(value)?;
    if days.data.len() == 1 {
        return duration_display_text(value);
    }
    let shape = default_shape_for(&days.shape, days.data.len());
    Ok(Some(format!(
        "[{} duration]",
        shape
            .iter()
            .map(|dim| dim.to_string())
            .collect::<Vec<_>>()
            .join("x")
    )))
}

pub fn duration_char_array(value: &Value) -> BuiltinResult<Option<CharArray>> {
    let Some(array) = duration_string_array(value)? else {
        return Ok(None);
    };
    let width = array.data.iter().map(String::len).max().unwrap_or(0);
    let rows = array.data.len();
    let mut data = vec![' '; rows * width];
    for (row, text) in array.data.iter().enumerate() {
        for (col, ch) in text.chars().enumerate() {
            data[row * width + col] = ch;
        }
    }
    let out = CharArray::new(data, rows, width)
        .map_err(|err| duration_error(format!("duration: {err}")))?;
    Ok(Some(out))
}

fn compare_duration(
    lhs: Value,
    rhs: Value,
    op: &str,
    cmp: impl Fn(f64, f64) -> bool,
) -> BuiltinResult<Value> {
    let lhs_days = duration_tensor_from_duration_value(&lhs)?;
    let rhs_days = duration_tensor_from_duration_value(&rhs)?;
    let (left, right, shape) = binary_numeric_tensors(&lhs_days, &rhs_days, op)?;
    let out = left
        .iter()
        .zip(right.iter())
        .map(|(a, b)| if cmp(*a, *b) { 1.0 } else { 0.0 })
        .collect::<Vec<_>>();
    if out.len() == 1 {
        Ok(Value::Num(out[0]))
    } else {
        Ok(Value::Tensor(Tensor::new(out, shape).map_err(|err| {
            duration_error(format!("duration: {err}"))
        })?))
    }
}

async fn duration_indexing(obj: Value, payload: Value) -> BuiltinResult<Value> {
    let Value::Object(object) = obj else {
        return Err(duration_error(
            "duration.subsref: receiver must be a duration object",
        ));
    };
    let format = format_for_object(&object);
    let days = duration_tensor_from_duration_value(&Value::Object(object.clone()))?;

    let Value::Cell(cell) = payload else {
        return Err(duration_error(
            "duration.subsref: indexing payload must be a cell array",
        ));
    };
    if cell.data.is_empty() {
        return duration_object_from_days_tensor(days, format);
    }
    if cell.data.len() != 1 {
        return Err(duration_error(
            "duration.subsref: only linear duration indexing is currently supported",
        ));
    }
    let selector = (*cell.data[0]).clone();
    let selector = match selector {
        Value::Tensor(tensor) => tensor,
        Value::Num(value) => Tensor::new(vec![value], vec![1, 1])
            .map_err(|err| duration_error(format!("duration.subsref: {err}")))?,
        Value::Int(value) => Tensor::new(vec![value.to_f64()], vec![1, 1])
            .map_err(|err| duration_error(format!("duration.subsref: {err}")))?,
        Value::LogicalArray(logical) => tensor::logical_to_tensor(&logical)
            .map_err(|err| duration_error(format!("duration.subsref: {err}")))?,
        other => {
            return Err(duration_error(format!(
                "duration.subsref: unsupported index value {other:?}"
            )))
        }
    };
    let indexed = crate::perform_indexing(&Value::Tensor(days), &selector.data)
        .await
        .map_err(|err| duration_error(format!("duration.subsref: {}", err.message())))?;
    let indexed_days = match indexed {
        Value::Num(value) => Tensor::new(vec![value], vec![1, 1])
            .map_err(|err| duration_error(format!("duration.subsref: {err}")))?,
        Value::Tensor(tensor) => tensor,
        other => {
            return Err(duration_error(format!(
                "duration.subsref: unexpected indexing result {other:?}"
            )))
        }
    };
    duration_object_from_days_tensor(indexed_days, format)
}

#[runmat_macros::runtime_builtin(
    name = "duration",
    builtin_path = "crate::builtins::duration",
    category = "datetime",
    summary = "Create MATLAB-compatible duration arrays from hour, minute, and second components.",
    keywords = "duration,time span,elapsed time,Format",
    related = "datetime,string,char,disp",
    examples = "t = duration(1, 30, 45);"
)]
async fn duration_builtin(args: Vec<Value>) -> crate::BuiltinResult<Value> {
    ensure_duration_class_registered();
    let args = gather_args(&args).await?;
    let (positional_end, format) = parse_trailing_format(&args)?;
    let positional = args[..positional_end].to_vec();

    match positional.len() {
        1..=3 => build_from_components(positional, format),
        _ => Err(duration_error(
            "duration: unsupported argument pattern; use H/M/S numeric component inputs",
        )),
    }
}

#[runmat_macros::runtime_builtin(
    name = "duration.subsref",
    builtin_path = "crate::builtins::duration"
)]
async fn duration_subsref(obj: Value, kind: String, payload: Value) -> crate::BuiltinResult<Value> {
    match kind.as_str() {
        "()" => duration_indexing(obj, payload).await,
        "." => {
            let Value::Object(object) = obj else {
                return Err(duration_error(
                    "duration.subsref: receiver must be a duration object",
                ));
            };
            let field = scalar_text(&payload, "field selector")?;
            match field.as_str() {
                FORMAT_FIELD => Ok(Value::String(format_for_object(&object))),
                _ => Err(duration_error(format!(
                    "duration.subsref: unsupported duration property '{field}'"
                ))),
            }
        }
        other => Err(duration_error(format!(
            "duration.subsref: unsupported indexing kind '{other}'"
        ))),
    }
}

#[runmat_macros::runtime_builtin(
    name = "duration.subsasgn",
    builtin_path = "crate::builtins::duration"
)]
async fn duration_subsasgn(
    obj: Value,
    kind: String,
    payload: Value,
    rhs: Value,
) -> crate::BuiltinResult<Value> {
    let Value::Object(mut object) = obj else {
        return Err(duration_error(
            "duration.subsasgn: receiver must be a duration object",
        ));
    };
    match kind.as_str() {
        "." => {
            let field = scalar_text(&payload, "field selector")?;
            match field.as_str() {
                FORMAT_FIELD => {
                    let text = scalar_text(&rhs, "Format value")?;
                    object
                        .properties
                        .insert(FORMAT_FIELD.to_string(), Value::String(text));
                    Ok(Value::Object(object))
                }
                _ => Err(duration_error(format!(
                    "duration.subsasgn: unsupported duration property '{field}'"
                ))),
            }
        }
        _ => Err(duration_error(format!(
            "duration.subsasgn: unsupported indexing kind '{kind}'"
        ))),
    }
}

#[runmat_macros::runtime_builtin(name = "duration.eq", builtin_path = "crate::builtins::duration")]
async fn duration_eq(lhs: Value, rhs: Value) -> crate::BuiltinResult<Value> {
    compare_duration(lhs, rhs, "eq", |a, b| (a - b).abs() <= 1e-12)
}

#[runmat_macros::runtime_builtin(name = "duration.ne", builtin_path = "crate::builtins::duration")]
async fn duration_ne(lhs: Value, rhs: Value) -> crate::BuiltinResult<Value> {
    compare_duration(lhs, rhs, "ne", |a, b| (a - b).abs() > 1e-12)
}

#[runmat_macros::runtime_builtin(name = "duration.lt", builtin_path = "crate::builtins::duration")]
async fn duration_lt(lhs: Value, rhs: Value) -> crate::BuiltinResult<Value> {
    compare_duration(lhs, rhs, "lt", |a, b| a < b)
}

#[runmat_macros::runtime_builtin(name = "duration.le", builtin_path = "crate::builtins::duration")]
async fn duration_le(lhs: Value, rhs: Value) -> crate::BuiltinResult<Value> {
    compare_duration(lhs, rhs, "le", |a, b| a <= b)
}

#[runmat_macros::runtime_builtin(name = "duration.gt", builtin_path = "crate::builtins::duration")]
async fn duration_gt(lhs: Value, rhs: Value) -> crate::BuiltinResult<Value> {
    compare_duration(lhs, rhs, "gt", |a, b| a > b)
}

#[runmat_macros::runtime_builtin(name = "duration.ge", builtin_path = "crate::builtins::duration")]
async fn duration_ge(lhs: Value, rhs: Value) -> crate::BuiltinResult<Value> {
    compare_duration(lhs, rhs, "ge", |a, b| a >= b)
}

#[runmat_macros::runtime_builtin(
    name = "duration.plus",
    builtin_path = "crate::builtins::duration"
)]
async fn duration_plus(lhs: Value, rhs: Value) -> crate::BuiltinResult<Value> {
    let lhs_days = duration_tensor_from_duration_value(&lhs)?;
    if crate::builtins::datetime::is_datetime_object(&rhs) {
        let rhs_serials = crate::builtins::datetime::serials_from_datetime_value(&rhs)?;
        let (left, right, shape) = binary_numeric_tensors(&lhs_days, &rhs_serials, "plus")?;
        let serials = left
            .iter()
            .zip(right.iter())
            .map(|(a, b)| a + b)
            .collect::<Vec<_>>();
        let tensor =
            Tensor::new(serials, shape).map_err(|err| duration_error(format!("plus: {err}")))?;
        return crate::builtins::datetime::datetime_object_from_serial_tensor(
            tensor,
            crate::builtins::datetime::datetime_format_from_value(&rhs),
        );
    }

    let rhs_days = duration_tensor_from_duration_value(&rhs)?;
    let (left, right, shape) = binary_numeric_tensors(&lhs_days, &rhs_days, "plus")?;
    let days = left
        .iter()
        .zip(right.iter())
        .map(|(a, b)| a + b)
        .collect::<Vec<_>>();
    duration_object_from_days(days, shape, duration_format_from_value(&lhs))
}

#[runmat_macros::runtime_builtin(
    name = "duration.minus",
    builtin_path = "crate::builtins::duration"
)]
async fn duration_minus(lhs: Value, rhs: Value) -> crate::BuiltinResult<Value> {
    let lhs_days = duration_tensor_from_duration_value(&lhs)?;
    let rhs_days = duration_tensor_from_duration_value(&rhs)?;
    let (left, right, shape) = binary_numeric_tensors(&lhs_days, &rhs_days, "minus")?;
    let days = left
        .iter()
        .zip(right.iter())
        .map(|(a, b)| a - b)
        .collect::<Vec<_>>();
    duration_object_from_days(days, shape, duration_format_from_value(&lhs))
}

#[cfg(test)]
mod tests {
    use super::*;

    fn run_duration(args: Vec<Value>) -> Value {
        futures::executor::block_on(duration_builtin(args)).expect("duration")
    }

    #[test]
    fn duration_builds_from_components() {
        let value = run_duration(vec![Value::Num(1.0), Value::Num(30.0), Value::Num(45.0)]);
        let rendered = duration_display_text(&value)
            .expect("display")
            .expect("duration text");
        assert_eq!(rendered, "01:30:45");
    }

    #[test]
    fn duration_formats_arrays() {
        let hours = Value::Tensor(Tensor::new(vec![1.0, 2.0], vec![1, 2]).unwrap());
        let minutes = Value::Tensor(Tensor::new(vec![15.0, 45.0], vec![1, 2]).unwrap());
        let value = run_duration(vec![hours, minutes]);
        let rendered = duration_display_text(&value)
            .expect("display")
            .expect("duration text");
        assert!(rendered.contains("01:15:00"));
        assert!(rendered.contains("02:45:00"));
    }

    #[test]
    fn duration_supports_format_assignment_and_indexing() {
        let value = run_duration(vec![Value::Num(1.0), Value::Num(5.0)]);
        let updated = futures::executor::block_on(duration_subsasgn(
            value.clone(),
            ".".to_string(),
            Value::String(FORMAT_FIELD.to_string()),
            Value::String("hh:mm".to_string()),
        ))
        .expect("subsasgn");
        let rendered = duration_display_text(&updated)
            .expect("display")
            .expect("duration text");
        assert_eq!(rendered, "01:05");

        let array = run_duration(vec![
            Value::Tensor(Tensor::new(vec![1.0, 2.0], vec![1, 2]).unwrap()),
            Value::Num(0.0),
            Value::Num(0.0),
        ]);
        let payload =
            Value::Cell(runmat_builtins::CellArray::new(vec![Value::Num(2.0)], 1, 1).unwrap());
        let indexed =
            futures::executor::block_on(duration_subsref(array, "()".to_string(), payload))
                .expect("subsref");
        let text = duration_display_text(&indexed)
            .expect("display")
            .expect("duration text");
        assert_eq!(text, "02:00:00");
    }
}
