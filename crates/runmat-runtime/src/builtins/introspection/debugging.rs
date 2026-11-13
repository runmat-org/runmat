use crate::call::{BuiltinArguments, FunctionArgumentView};
use runmat_builtins::BuiltinErrorDescriptor;
pub use runmat_builtins::{
    DBCLEAR_DESCRIPTOR, DBCLEAR_ERROR_TOO_MANY_OUTPUTS, DBSTACK_DESCRIPTOR, DBSTATUS_DESCRIPTOR,
    DBSTATUS_ERROR_TOO_MANY_OUTPUTS, DBTYPE_DESCRIPTOR, DBTYPE_ERROR_TOO_MANY_OUTPUTS,
    DEBUG_ERROR_FILE_NOT_FOUND, DEBUG_ERROR_FILE_READ, DEBUG_ERROR_INVALID_INPUT,
    DEBUG_ERROR_NO_CURRENT_FILE, DEBUG_ERROR_TOO_MANY_INPUTS, DEBUG_ERROR_TOO_MANY_OUTPUTS,
    GETCALLINFO_DESCRIPTOR, GETCALLINFO_ERROR_TOO_MANY_OUTPUTS, KEYBOARD_DESCRIPTOR,
    KEYBOARD_ERROR_TOO_MANY_OUTPUTS, MISLOCKED_DESCRIPTOR, MISLOCKED_ERROR_TOO_MANY_OUTPUTS,
    MLOCK_DESCRIPTOR, MLOCK_ERROR_TOO_MANY_OUTPUTS, MUNLOCK_DESCRIPTOR,
    MUNLOCK_ERROR_TOO_MANY_OUTPUTS,
};
use runmat_macros::runtime_builtin;
use runmat_thread_local::runmat_thread_local;
use runmat_value::{CellArray, NumericDType, StructArray, StructValue, Tensor, Value};
use std::cell::RefCell;
use std::collections::HashSet;
use std::path::{Path, PathBuf};

use crate::builtins::common::path_search::{file_candidates, path_is_file};
use crate::builtins::common::tensor;
use crate::console::{record_console_line, ConsoleStream};
use crate::{BuiltinResult, RuntimeError};

runmat_thread_local! {
    static LOCKED_FUNCTIONS: RefCell<HashSet<String>> = RefCell::new(HashSet::new());
}

pub fn reset_lock_registry_for_tests() {
    LOCKED_FUNCTIONS.with(|locks| locks.borrow_mut().clear());
}

fn debug_error(
    builtin: &'static str,
    descriptor: &'static BuiltinErrorDescriptor,
    detail: impl AsRef<str>,
) -> RuntimeError {
    let detail = detail.as_ref();
    let message = if detail.is_empty() {
        descriptor.message.to_string()
    } else {
        format!("{}: {detail}", descriptor.message)
    };
    let mut builder = crate::build_runtime_error(message).with_builtin(builtin);
    if let Some(identifier) = descriptor.identifier {
        builder = builder.with_identifier(identifier);
    }
    builder.build()
}

fn empty_value() -> Value {
    Value::Tensor(Tensor::new(Vec::new(), vec![0, 0]).expect("empty tensor"))
}

fn text_value(value: &Value, builtin: &'static str) -> BuiltinResult<Option<String>> {
    match value {
        Value::String(runmat_value::StringElement::Text(text)) => {
            Ok(Some(text.as_str().to_owned()))
        }
        Value::StringArray(array) if array.data().len() == 1 && !array.data()[0].is_missing() => {
            Ok(Some(
                array.data()[0]
                    .text_ref()
                    .expect("checked nonmissing scalar string")
                    .as_str()
                    .to_owned(),
            ))
        }
        Value::CharArray(array)
            if array.rows == 1 && array.shape().iter().skip(2).all(|n| *n == 1) =>
        {
            super::text::decode_utf16_with_error(array.data(), || {
                debug_error(builtin, &DEBUG_ERROR_INVALID_INPUT, "invalid UTF-16 text")
            })
            .map(Some)
        }
        _ => Ok(None),
    }
}

fn integer_value(value: &Value) -> Option<usize> {
    match value {
        Value::Num(n) => nonnegative_platform_usize(*n),
        Value::Tensor(tensor)
            if tensor.numeric_dtype() == NumericDType::F64
                && tensor::tensor_element_len(tensor) == 1 =>
        {
            nonnegative_platform_usize(tensor::tensor_value_f64(tensor, 0))
        }
        _ => tensor::scalar_integer_value(value).and_then(|int| int.try_to_usize()),
    }
}

fn nonnegative_platform_usize(value: f64) -> Option<usize> {
    if !value.is_finite() || value < 0.0 || value.fract() != 0.0 {
        return None;
    }
    if value > usize::MAX as f64 || (usize::BITS == 64 && value == usize::MAX as f64) {
        return None;
    }
    Some(value as usize)
}

fn stack_struct(frame: &crate::debug_context::DebugFrameInfo, complete_names: bool) -> StructValue {
    let mut st = runmat_value::UnboundedStructValueBuilder::new();
    let file = if complete_names {
        frame.file.clone()
    } else {
        Path::new(&frame.file)
            .file_name()
            .and_then(|name| name.to_str())
            .unwrap_or(&frame.file)
            .to_string()
    };
    st.insert("file", Value::string(file));
    st.insert("name", Value::string(frame.function.clone()));
    st.insert("line", Value::Num(frame.line as f64));
    st.finish()
}

fn cell_column(values: Vec<Value>) -> BuiltinResult<Value> {
    if values.is_empty() {
        return Ok(Value::Cell(CellArray::new(Vec::new(), 0, 0).map_err(
            |err| debug_error("debugger", &DEBUG_ERROR_INVALID_INPUT, &err),
        )?));
    }
    let rows = values.len();
    Ok(Value::Cell(CellArray::new(values, rows, 1).map_err(
        |err| debug_error("debugger", &DEBUG_ERROR_INVALID_INPUT, &err),
    )?))
}

fn stack_value(skip: usize, complete_names: bool) -> BuiltinResult<Value> {
    let mut frames = crate::debug_context::current_frames();
    if frames.is_empty() {
        if let Some(source) = crate::source_context::current_source_info() {
            frames.push(crate::debug_context::DebugFrameInfo {
                function: source.name.to_string(),
                file: source
                    .fullpath_name
                    .as_ref()
                    .map(ToString::to_string)
                    .unwrap_or_else(|| source.name.to_string()),
                line: 0,
            });
        }
    }
    let entries = frames
        .iter()
        .skip(skip)
        .map(|frame| stack_struct(frame, complete_names))
        .collect::<Vec<_>>();
    StructArray::normalize(
        vec!["file".into(), "name".into(), "line".into()],
        entries,
        vec![frames.len().saturating_sub(skip), 1],
    )
    .map_err(|error| debug_error("dbstack", &DEBUG_ERROR_INVALID_INPUT, &error))
}

fn parse_dbstack_args(args: FunctionArgumentView<'_>) -> BuiltinResult<(usize, bool)> {
    if args.len() > 2 {
        return Err(debug_error("dbstack", &DEBUG_ERROR_TOO_MANY_INPUTS, ""));
    }
    let first_complete = args
        .get(0)
        .map(|value| is_complete_names(value, "dbstack"))
        .transpose()?
        .unwrap_or(false);
    let second_complete = args
        .get(1)
        .map(|value| is_complete_names(value, "dbstack"))
        .transpose()?
        .unwrap_or(false);
    match (args.get(0), args.get(1)) {
        (None, None) => Ok((0, false)),
        (Some(_), None) if first_complete => Ok((0, true)),
        (Some(arg), None) => integer_value(arg)
            .map(|skip| (skip, false))
            .ok_or_else(|| debug_error("dbstack", &DEBUG_ERROR_INVALID_INPUT, "")),
        (Some(_), Some(n)) if first_complete => integer_value(n)
            .map(|skip| (skip, true))
            .ok_or_else(|| debug_error("dbstack", &DEBUG_ERROR_INVALID_INPUT, "")),
        (Some(n), Some(_)) if second_complete => integer_value(n)
            .map(|skip| (skip, true))
            .ok_or_else(|| debug_error("dbstack", &DEBUG_ERROR_INVALID_INPUT, "")),
        _ => Err(debug_error("dbstack", &DEBUG_ERROR_INVALID_INPUT, "")),
    }
}

fn is_complete_names(value: &Value, builtin: &'static str) -> BuiltinResult<bool> {
    Ok(text_value(value, builtin)?
        .is_some_and(|text| text.trim().eq_ignore_ascii_case("-completenames")))
}

pub(crate) fn dispatch_dbstack(args: FunctionArgumentView<'_>) -> BuiltinResult<Value> {
    crate::output_count::validate_bounded_outputs(
        &DBSTACK_DESCRIPTOR,
        &DEBUG_ERROR_TOO_MANY_OUTPUTS,
        "dbstack",
    )?;
    let (skip, complete_names) = parse_dbstack_args(args)?;
    let stack = stack_value(skip, complete_names)?;
    match crate::output_count::current_output_count() {
        Some(0) => {
            record_console_line(ConsoleStream::Stdout, format_stack_for_display(&stack)?)?;
            Ok(empty_value())
        }
        Some(n) if n >= 2 => Ok(crate::output_count::output_list_with_padding(
            n,
            vec![stack, Value::Num(1.0)],
        )),
        _ => Ok(stack),
    }
}

fn format_stack_for_display(stack: &Value) -> BuiltinResult<String> {
    let structures: Vec<StructValue> = match stack {
        Value::Struct(structure) => vec![structure.clone()],
        Value::StructArray(array) => array.elements().map(|value| value.to_owned()).collect(),
        _ => return Ok(String::new()),
    };
    let lines = structures
        .into_iter()
        .map(|st| {
            let name = st
                .fields
                .get("name")
                .map(|value| text_value(value, "dbstack"))
                .transpose()?
                .flatten()
                .unwrap_or_default();
            let file = st
                .fields
                .get("file")
                .map(|value| text_value(value, "dbstack"))
                .transpose()?
                .flatten()
                .unwrap_or_default();
            let line = st
                .fields
                .get("line")
                .and_then(|value| match value {
                    Value::Num(n) => nonnegative_platform_usize(*n),
                    Value::Int(int) => int.try_to_usize(),
                    _ => None,
                })
                .unwrap_or(0);
            Ok(if file.is_empty() {
                format!("In {name} at line {line}")
            } else {
                format!("In {name} ({file}) at line {line}")
            })
        })
        .collect::<BuiltinResult<Vec<_>>>()?;
    Ok(lines.join("\n"))
}

fn normalize_lock_key(raw: &str) -> String {
    let path = Path::new(raw.trim());
    let name = path
        .file_stem()
        .and_then(|stem| stem.to_str())
        .filter(|stem| !stem.is_empty())
        .unwrap_or_else(|| raw.trim());
    name.to_ascii_lowercase()
}

fn current_lock_key() -> BuiltinResult<String> {
    if let Some(name) = crate::debug_context::current_function_name() {
        if !name.is_empty() && name != "<main>" && name != "<anonymous>" {
            return Ok(normalize_lock_key(&name));
        }
    }
    crate::source_context::current_source_info()
        .map(|source| normalize_lock_key(&source.name))
        .filter(|key| !key.is_empty())
        .ok_or_else(|| debug_error("mlock", &DEBUG_ERROR_NO_CURRENT_FILE, ""))
}

fn optional_lock_key(
    args: FunctionArgumentView<'_>,
    builtin: &'static str,
) -> BuiltinResult<String> {
    match (args.len(), args.get(0)) {
        (0, _) => current_lock_key(),
        (1, Some(value)) => text_value(value, builtin)?
            .map(|text| normalize_lock_key(&text))
            .filter(|key| !key.is_empty())
            .ok_or_else(|| debug_error(builtin, &DEBUG_ERROR_INVALID_INPUT, "")),
        _ => Err(debug_error(builtin, &DEBUG_ERROR_TOO_MANY_INPUTS, "")),
    }
}

pub(crate) fn dispatch_mlock(args: FunctionArgumentView<'_>) -> BuiltinResult<Value> {
    if !args.is_empty() {
        return Err(debug_error("mlock", &DEBUG_ERROR_TOO_MANY_INPUTS, ""));
    }
    let key = current_lock_key()?;
    LOCKED_FUNCTIONS.with(|locks| {
        locks.borrow_mut().insert(key);
    });
    Ok(empty_value())
}

pub(crate) fn dispatch_munlock(args: FunctionArgumentView<'_>) -> BuiltinResult<Value> {
    let key = optional_lock_key(args, "munlock")?;
    LOCKED_FUNCTIONS.with(|locks| {
        locks.borrow_mut().remove(&key);
    });
    Ok(empty_value())
}

pub(crate) fn dispatch_mislocked(args: FunctionArgumentView<'_>) -> BuiltinResult<Value> {
    let key = optional_lock_key(args, "mislocked")?;
    let locked = LOCKED_FUNCTIONS.with(|locks| locks.borrow().contains(&key));
    Ok(Value::Bool(locked))
}

fn empty_breakpoint_list() -> BuiltinResult<Value> {
    cell_column(Vec::new())
}

pub(crate) fn dispatch_dbstatus(args: FunctionArgumentView<'_>) -> BuiltinResult<Value> {
    for arg in args.iter() {
        if text_value(arg, "dbstatus")?.is_none() {
            return Err(debug_error("dbstatus", &DEBUG_ERROR_INVALID_INPUT, ""));
        }
    }
    let status = empty_breakpoint_list()?;
    match crate::output_count::current_output_count() {
        Some(0) => Ok(empty_value()),
        _ => Ok(status),
    }
}

pub(crate) fn dispatch_dbclear(args: FunctionArgumentView<'_>) -> BuiltinResult<Value> {
    for arg in args.iter() {
        if text_value(arg, "dbclear")?.is_none() {
            return Err(debug_error("dbclear", &DEBUG_ERROR_INVALID_INPUT, ""));
        }
    }
    Ok(empty_value())
}

fn parse_line_range(value: Option<&Value>) -> BuiltinResult<Option<(usize, usize)>> {
    let Some(value) = value else {
        return Ok(None);
    };
    let text = text_value(value, "dbtype")?
        .ok_or_else(|| debug_error("dbtype", &DEBUG_ERROR_INVALID_INPUT, ""))?;
    let Some((start, end)) = text.split_once(':') else {
        let line = text
            .trim()
            .parse::<usize>()
            .map_err(|_| debug_error("dbtype", &DEBUG_ERROR_INVALID_INPUT, text.clone()))?;
        if line == 0 {
            return Err(debug_error("dbtype", &DEBUG_ERROR_INVALID_INPUT, text));
        }
        return Ok(Some((line, line)));
    };
    let start = start
        .trim()
        .parse::<usize>()
        .map_err(|_| debug_error("dbtype", &DEBUG_ERROR_INVALID_INPUT, text.clone()))?;
    let end = end
        .trim()
        .parse::<usize>()
        .map_err(|_| debug_error("dbtype", &DEBUG_ERROR_INVALID_INPUT, text.clone()))?;
    if start == 0 || end == 0 || start > end {
        return Err(debug_error("dbtype", &DEBUG_ERROR_INVALID_INPUT, text));
    }
    Ok(Some((start, end)))
}

async fn resolve_dbtype_path(
    name: &str,
) -> Result<Option<PathBuf>, crate::builtins::common::path_search::PathSearchError> {
    for candidate in file_candidates(name, &[".m", ""], "dbtype")? {
        if path_is_file(&candidate).await? {
            return Ok(Some(candidate));
        }
    }
    Ok(None)
}

pub(crate) async fn dispatch_dbtype(args: FunctionArgumentView<'_>) -> BuiltinResult<Value> {
    if args.is_empty() {
        return Err(debug_error("dbtype", &DEBUG_ERROR_INVALID_INPUT, ""));
    }
    if args.len() > 2 {
        return Err(debug_error("dbtype", &DEBUG_ERROR_TOO_MANY_INPUTS, ""));
    }
    let file = text_value(&args[0], "dbtype")?
        .ok_or_else(|| debug_error("dbtype", &DEBUG_ERROR_INVALID_INPUT, "file must be text"))?;
    let range = parse_line_range(args.get(1))?;
    let path = resolve_dbtype_path(&file)
        .await
        .map_err(|error| error.into_runtime_error("dbtype"))?
        .ok_or_else(|| debug_error("dbtype", &DEBUG_ERROR_FILE_NOT_FOUND, file.clone()))?;
    let text = runmat_filesystem::read_to_string_async(&path)
        .await
        .map_err(|source| {
            let mut error =
                crate::build_runtime_error(format!("{}: {source}", DEBUG_ERROR_FILE_READ.message))
                    .with_builtin("dbtype")
                    .with_source(source);
            if let Some(identifier) = DEBUG_ERROR_FILE_READ.identifier {
                error = error.with_identifier(identifier);
            }
            error
                .with_gpu_gather_retry(crate::GpuGatherRetry::Never)
                .build()
        })?;
    let lines = text.lines().collect::<Vec<_>>();
    let (start, end) = range.unwrap_or((1, lines.len()));
    let rendered = lines
        .iter()
        .enumerate()
        .filter_map(|(idx, line)| {
            let line_number = idx + 1;
            if line_number < start || line_number > end {
                None
            } else {
                Some(format!("{line_number:>5} {line}"))
            }
        })
        .collect::<Vec<_>>()
        .join("\n");
    record_console_line(ConsoleStream::Stdout, rendered)?;
    Ok(empty_value())
}

pub(crate) fn dispatch_keyboard(args: FunctionArgumentView<'_>) -> BuiltinResult<Value> {
    if !args.is_empty() {
        return Err(debug_error("keyboard", &DEBUG_ERROR_TOO_MANY_INPUTS, ""));
    }
    Ok(empty_value())
}

pub(crate) fn dispatch_getcallinfo(args: FunctionArgumentView<'_>) -> BuiltinResult<Value> {
    if !args.is_empty() {
        return Err(debug_error("getcallinfo", &DEBUG_ERROR_TOO_MANY_INPUTS, ""));
    }
    let mut info = runmat_value::UnboundedStructValueBuilder::new();
    let current = crate::debug_context::current_frames()
        .into_iter()
        .next()
        .unwrap_or_else(|| crate::debug_context::DebugFrameInfo {
            function: String::new(),
            file: String::new(),
            line: 0,
        });
    info.insert("name", Value::string(current.function));
    info.insert("file", Value::string(current.file));
    info.insert("line", Value::Num(current.line as f64));
    info.insert("stack", stack_value(0, false)?);
    Ok(Value::Struct(info.finish()))
}

#[runtime_builtin(
    name = "dbstack",
    binding_variant = "default",
    builtin_path = "crate::builtins::introspection::debugging"
)]
fn dbstack_builtin(args: BuiltinArguments) -> BuiltinResult<Value> {
    dispatch_dbstack(args.view())
}

#[runtime_builtin(
    name = "dbtype",
    binding_variant = "default",
    builtin_path = "crate::builtins::introspection::debugging"
)]
async fn dbtype_builtin(args: BuiltinArguments) -> BuiltinResult<Value> {
    crate::output_count::validate_bounded_outputs(
        &DBTYPE_DESCRIPTOR,
        &DBTYPE_ERROR_TOO_MANY_OUTPUTS,
        "dbtype",
    )?;
    dispatch_dbtype(args.view()).await
}

#[runtime_builtin(
    name = "keyboard",
    binding_variant = "default",
    builtin_path = "crate::builtins::introspection::debugging"
)]
fn keyboard_builtin(args: BuiltinArguments) -> BuiltinResult<Value> {
    crate::output_count::validate_bounded_outputs(
        &KEYBOARD_DESCRIPTOR,
        &KEYBOARD_ERROR_TOO_MANY_OUTPUTS,
        "keyboard",
    )?;
    dispatch_keyboard(args.view())
}

#[runtime_builtin(
    name = "mlock",
    binding_variant = "default",
    builtin_path = "crate::builtins::introspection::debugging"
)]
fn mlock_builtin(args: BuiltinArguments) -> BuiltinResult<Value> {
    crate::output_count::validate_bounded_outputs(
        &MLOCK_DESCRIPTOR,
        &MLOCK_ERROR_TOO_MANY_OUTPUTS,
        "mlock",
    )?;
    dispatch_mlock(args.view())
}

#[runtime_builtin(
    name = "munlock",
    binding_variant = "default",
    builtin_path = "crate::builtins::introspection::debugging"
)]
fn munlock_builtin(args: BuiltinArguments) -> BuiltinResult<Value> {
    crate::output_count::validate_bounded_outputs(
        &MUNLOCK_DESCRIPTOR,
        &MUNLOCK_ERROR_TOO_MANY_OUTPUTS,
        "munlock",
    )?;
    dispatch_munlock(args.view())
}

#[runtime_builtin(
    name = "mislocked",
    binding_variant = "default",
    builtin_path = "crate::builtins::introspection::debugging"
)]
fn mislocked_builtin(args: BuiltinArguments) -> BuiltinResult<Value> {
    crate::output_count::validate_bounded_outputs(
        &MISLOCKED_DESCRIPTOR,
        &MISLOCKED_ERROR_TOO_MANY_OUTPUTS,
        "mislocked",
    )?;
    dispatch_mislocked(args.view())
}

#[runtime_builtin(
    name = "dbstatus",
    binding_variant = "default",
    builtin_path = "crate::builtins::introspection::debugging"
)]
fn dbstatus_builtin(args: BuiltinArguments) -> BuiltinResult<Value> {
    crate::output_count::validate_bounded_outputs(
        &DBSTATUS_DESCRIPTOR,
        &DBSTATUS_ERROR_TOO_MANY_OUTPUTS,
        "dbstatus",
    )?;
    dispatch_dbstatus(args.view())
}

#[runtime_builtin(
    name = "dbclear",
    binding_variant = "default",
    builtin_path = "crate::builtins::introspection::debugging"
)]
fn dbclear_builtin(args: BuiltinArguments) -> BuiltinResult<Value> {
    crate::output_count::validate_bounded_outputs(
        &DBCLEAR_DESCRIPTOR,
        &DBCLEAR_ERROR_TOO_MANY_OUTPUTS,
        "dbclear",
    )?;
    dispatch_dbclear(args.view())
}

#[runtime_builtin(
    name = "getcallinfo",
    binding_variant = "default",
    builtin_path = "crate::builtins::introspection::debugging"
)]
fn getcallinfo_builtin(args: BuiltinArguments) -> BuiltinResult<Value> {
    crate::output_count::validate_bounded_outputs(
        &GETCALLINFO_DESCRIPTOR,
        &GETCALLINFO_ERROR_TOO_MANY_OUTPUTS,
        "getcallinfo",
    )?;
    dispatch_getcallinfo(args.view())
}

#[cfg(test)]
mod tests {
    use super::*;
    use runmat_types::SourceId;
    use runmat_value::{CharArray, IntValue, IntegerStorage};

    #[test]
    fn debug_controls_reject_character_pages_zero_lines_and_excess_outputs() {
        let pages =
            CharArray::from_unicode_scalars_with_shape(vec!['a', 'b'], vec![1, 1, 2]).unwrap();
        assert!(text_value(&Value::CharArray(pages), "dbtype")
            .unwrap()
            .is_none());
        assert!(parse_line_range(Some(&Value::string("0"))).is_err());
        let _outputs = crate::output_count::push_output_count(Some(1)).unwrap();
        let arguments = || crate::builtins::common::test_support::builtin_arguments(vec![]);
        for error in [
            keyboard_builtin(arguments()).unwrap_err(),
            mlock_builtin(arguments()).unwrap_err(),
            munlock_builtin(arguments()).unwrap_err(),
            dbclear_builtin(arguments()).unwrap_err(),
        ] {
            assert!(error
                .identifier()
                .is_some_and(|id| id.ends_with(":TooManyOutputs")));
        }
    }

    fn cell_len(value: &Value) -> usize {
        let Value::Cell(cell) = value else {
            panic!("expected cell row, got {value:?}");
        };
        cell.data().len()
    }

    fn stack_len(value: &Value) -> usize {
        match value {
            Value::Struct(_) => 1,
            Value::StructArray(array) => array.len(),
            other => panic!("expected stack structure array, got {other:?}"),
        }
    }

    fn stack_element(value: &Value, index: usize) -> StructValue {
        match value {
            Value::Struct(structure) if index == 0 => structure.clone(),
            Value::StructArray(array) => array.get_linear(index).unwrap().to_owned(),
            other => panic!("expected stack structure array, got {other:?}"),
        }
    }

    #[test]
    fn typed_debug_offsets_preserve_platform_representable_uint64() {
        assert_eq!(integer_value(&Value::Int(IntValue::U64(3))), Some(3));
        assert_eq!(
            integer_value(&Value::Int(IntValue::U64(u64::MAX))),
            usize::try_from(u64::MAX).ok()
        );
        assert_eq!(integer_value(&Value::Int(IntValue::I64(-1))), None);
        assert_eq!(integer_value(&Value::Num(1.5)), None);
        assert_eq!(integer_value(&Value::Num(usize::MAX as f64 + 1.0)), None);
        if usize::BITS == 64 {
            assert_eq!(integer_value(&Value::Num(usize::MAX as f64)), None);
        }
    }

    #[test]
    fn dbstack_accepts_every_exact_integer_scalar_class() {
        for storage in [
            IntegerStorage::I8(vec![0]),
            IntegerStorage::I16(vec![0]),
            IntegerStorage::I32(vec![0]),
            IntegerStorage::I64(vec![0]),
            IntegerStorage::U8(vec![0]),
            IntegerStorage::U16(vec![0]),
            IntegerStorage::U32(vec![0]),
            IntegerStorage::U64(vec![0]),
        ] {
            let scalar = Tensor::new_integer(storage, vec![1, 1]).expect("integer scalar");
            assert_eq!(
                parse_dbstack_args(FunctionArgumentView::from_slice(&[Value::Tensor(scalar)]))
                    .unwrap(),
                (0, false)
            );
        }
        let scalar_double = Tensor::new(vec![0.0], vec![1, 1]).expect("double scalar");
        assert_eq!(
            parse_dbstack_args(FunctionArgumentView::from_slice(&[Value::Tensor(
                scalar_double
            )]))
            .unwrap(),
            (0, false)
        );
    }

    #[test]
    fn dbstack_accepts_documented_option_orders_and_rejects_duplicates() {
        assert!(parse_dbstack_args(FunctionArgumentView::from_slice(&[
            Value::Num(1.0),
            Value::Num(2.0)
        ]))
        .is_err());
        assert_eq!(
            parse_dbstack_args(FunctionArgumentView::from_slice(&[
                Value::Num(1.0),
                Value::String("-completenames".into())
            ]))
            .unwrap(),
            (1, true)
        );
        assert_eq!(
            parse_dbstack_args(FunctionArgumentView::from_slice(&[
                Value::String("-completenames".into()),
                Value::Num(1.0)
            ]))
            .unwrap(),
            (1, true)
        );
        assert!(parse_dbstack_args(FunctionArgumentView::from_slice(&[
            Value::String("-completenames".into()),
            Value::String("-completenames".into())
        ]))
        .is_err());
        assert!(parse_dbstack_args(FunctionArgumentView::from_slice(&[
            Value::Num(1.0),
            Value::String("-completenames".into()),
            Value::String("-completenames".into())
        ]))
        .is_err());
    }

    #[test]
    fn dbstack_rejects_more_than_two_outputs() {
        let _outputs = crate::output_count::push_output_count(Some(3)).expect("output-count scope");
        let err = dispatch_dbstack(FunctionArgumentView::from_slice(&[])).unwrap_err();
        assert_eq!(err.identifier(), Some("RunMat:dbstack:TooManyOutputs"));
    }

    #[test]
    fn dbstack_returns_source_backed_structure_array() {
        let source_id = SourceId(31);
        let _catalog = crate::source_context::replace_source_catalog_with_fullpaths(vec![(
            source_id,
            "worker.m".to_string(),
            Some("/tmp/worker.m".to_string()),
            "function worker\nx = dbstack;\nend\n".to_string(),
        )])
        .expect("debugging source catalog admission");
        let _guard = crate::debug_context::push_frame("worker", Some(source_id), Some((16, 25)));
        let value = dispatch_dbstack(FunctionArgumentView::from_slice(&[])).expect("dbstack");
        assert_eq!(stack_len(&value), 1);
        let st = stack_element(&value, 0);
        assert_eq!(st.fields.get("name"), Some(&Value::String("worker".into())));
        assert_eq!(
            st.fields.get("file"),
            Some(&Value::String("worker.m".into()))
        );
        assert_eq!(st.fields.len(), 3);
        let complete = dispatch_dbstack(FunctionArgumentView::from_slice(&[Value::String(
            "-completenames".into(),
        )]))
        .expect("complete names");
        let complete_frame = stack_element(&complete, 0);
        assert_eq!(
            complete_frame.fields.get("file"),
            Some(&Value::String("/tmp/worker.m".into()))
        );
    }

    #[test]
    fn dbstack_skips_leading_entries() {
        let _outer = crate::debug_context::push_frame("outer", None, None);
        let _inner = crate::debug_context::push_frame("inner", None, None);
        let value = dispatch_dbstack(FunctionArgumentView::from_slice(&[Value::Num(1.0)]))
            .expect("dbstack");
        assert_eq!(stack_len(&value), 1);
        let st = stack_element(&value, 0);
        assert_eq!(st.fields.get("name"), Some(&Value::String("outer".into())));
    }

    #[test]
    fn mlock_mislocked_munlock_current_function() {
        let _guard = crate::debug_context::push_frame("locked_fn", None, None);
        dispatch_mlock(FunctionArgumentView::from_slice(&[])).expect("mlock");
        assert_eq!(
            dispatch_mislocked(FunctionArgumentView::from_slice(&[])).expect("mislocked"),
            Value::Bool(true)
        );
        dispatch_munlock(FunctionArgumentView::from_slice(&[Value::String(
            "locked_fn".into(),
        )]))
        .expect("munlock");
        assert_eq!(
            dispatch_mislocked(FunctionArgumentView::from_slice(&[Value::String(
                "locked_fn".into()
            )]))
            .expect("mislocked"),
            Value::Bool(false)
        );
    }

    #[test]
    fn dbstatus_returns_empty_breakpoint_cell() {
        let value = dispatch_dbstatus(FunctionArgumentView::from_slice(&[])).expect("dbstatus");
        assert_eq!(cell_len(&value), 0);
    }

    #[test]
    fn dbtype_reads_source_file_with_line_range() {
        let dir = tempfile::tempdir().expect("tempdir");
        let path = dir.path().join("demo_dbtype.m");
        std::fs::write(&path, "a = 1;\nb = 2;\nc = 3;\n").expect("write source");
        let result =
            futures::executor::block_on(dispatch_dbtype(FunctionArgumentView::from_slice(&[
                Value::String(path.to_string_lossy().to_string().into()),
                Value::String("2:3".to_string().into()),
            ])))
            .expect("dbtype");
        assert!(matches!(result, Value::Tensor(t) if t.materialize_f64().is_empty()));
    }

    #[test]
    fn getcallinfo_reports_current_frame() {
        let _guard = crate::debug_context::push_frame("callsite", None, None);
        let value =
            dispatch_getcallinfo(FunctionArgumentView::from_slice(&[])).expect("getcallinfo");
        let Value::Struct(st) = value else {
            panic!("expected struct");
        };
        assert_eq!(
            st.fields.get("name"),
            Some(&Value::String("callsite".into()))
        );
        let stack = st.fields.get("stack").expect("stack field");
        assert_eq!(stack_len(stack), 1);
        let frame = stack_element(stack, 0);
        assert_eq!(frame.fields.get("name"), st.fields.get("name"));
        assert_eq!(frame.fields.get("file"), st.fields.get("file"));
        assert_eq!(frame.fields.get("line"), st.fields.get("line"));
        assert_eq!(frame.fields.len(), 3);
    }

    #[test]
    fn getcallinfo_preserves_ordered_structure_stack() {
        let _outer = crate::debug_context::push_frame("outer", None, None);
        let _inner = crate::debug_context::push_frame("inner", None, None);
        let Value::Struct(info) =
            dispatch_getcallinfo(FunctionArgumentView::from_slice(&[])).expect("getcallinfo")
        else {
            panic!("expected call information structure");
        };
        let stack = info.fields.get("stack").expect("stack field");
        let Value::StructArray(array) = stack else {
            panic!("expected structure array for two frames");
        };
        assert_eq!(array.shape(), [2, 1]);
        assert_eq!(stack_len(stack), 2);
        for (index, name) in ["inner", "outer"].into_iter().enumerate() {
            assert_eq!(
                stack_element(stack, index).fields.get("name"),
                Some(&Value::String(name.into()))
            );
        }
        assert_eq!(
            info.fields.get("name"),
            Some(&Value::String("inner".into()))
        );
    }
}
