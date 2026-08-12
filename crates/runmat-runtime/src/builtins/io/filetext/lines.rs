//! MATLAB-compatible `readlines` and `writelines` helpers.

use std::path::PathBuf;

use runmat_builtins::{
    BuiltinCompletionPolicy, BuiltinDescriptor, BuiltinOutputMode, BuiltinParamArity,
    BuiltinParamDescriptor, BuiltinParamType, BuiltinSignatureDescriptor,
};
use runmat_filesystem as vfs;
use runmat_macros::runtime_builtin;
use runmat_value::{CellArray, StringArray, Value};

use crate::builtins::common::fs::expand_user_path;
use crate::{build_runtime_error, gather_if_needed_async, BuiltinResult, RuntimeError};

fn line_error(name: &str, message: impl Into<String>) -> RuntimeError {
    build_runtime_error(message).with_builtin(name).build()
}

const READLINES_INPUTS: [BuiltinParamDescriptor; 1] = [BuiltinParamDescriptor {
    name: "filename",
    ty: BuiltinParamType::StringScalar,
    arity: BuiltinParamArity::Required,
    default: None,
    description: "Text file to read.",
}];
const WRITELINES_INPUTS: [BuiltinParamDescriptor; 2] = [
    BuiltinParamDescriptor {
        name: "lines",
        ty: BuiltinParamType::Any,
        arity: BuiltinParamArity::Required,
        default: None,
        description: "Text lines to write.",
    },
    BuiltinParamDescriptor {
        name: "filename",
        ty: BuiltinParamType::StringScalar,
        arity: BuiltinParamArity::Required,
        default: None,
        description: "Destination text file.",
    },
];
const OUTPUT_VALUE: [BuiltinParamDescriptor; 1] = [BuiltinParamDescriptor {
    name: "value",
    ty: BuiltinParamType::Any,
    arity: BuiltinParamArity::Required,
    default: None,
    description: "Result value.",
}];
const READLINES_SIGNATURES: [BuiltinSignatureDescriptor; 1] = [BuiltinSignatureDescriptor {
    label: "lines = readlines(filename)",
    inputs: &READLINES_INPUTS,
    outputs: &OUTPUT_VALUE,
}];
pub const READLINES_DESCRIPTOR: BuiltinDescriptor = BuiltinDescriptor {
    signatures: &READLINES_SIGNATURES,
    output_mode: BuiltinOutputMode::Fixed,
    completion_policy: BuiltinCompletionPolicy::Public,
    errors: &[],
};
const WRITELINES_SIGNATURES: [BuiltinSignatureDescriptor; 1] = [BuiltinSignatureDescriptor {
    label: "status = writelines(lines, filename)",
    inputs: &WRITELINES_INPUTS,
    outputs: &OUTPUT_VALUE,
}];
pub const WRITELINES_DESCRIPTOR: BuiltinDescriptor = BuiltinDescriptor {
    signatures: &WRITELINES_SIGNATURES,
    output_mode: BuiltinOutputMode::Fixed,
    completion_policy: BuiltinCompletionPolicy::Public,
    errors: &[],
};

fn map_control_flow(name: &str, err: RuntimeError) -> RuntimeError {
    build_runtime_error(format!("{name}: {}", err.message()))
        .with_builtin(name)
        .with_source(err)
        .build()
}

async fn gather_args(name: &str, args: &[Value]) -> BuiltinResult<Vec<Value>> {
    let mut out = Vec::with_capacity(args.len());
    for value in args {
        out.push(
            gather_if_needed_async(value)
                .await
                .map_err(|err| map_control_flow(name, err))?,
        );
    }
    Ok(out)
}

fn scalar_text(value: &Value, name: &str, arg: &str) -> BuiltinResult<String> {
    match value {
        Value::String(text) => Ok(text.clone()),
        Value::CharArray(array) if array.rows == 1 => Ok(array.data.iter().collect()),
        Value::StringArray(array) if array.data.len() == 1 => Ok(array.data[0].clone()),
        _ => Err(line_error(
            name,
            format!("{name}: {arg} must be a string scalar or character vector"),
        )),
    }
}

fn path_from_value(value: &Value, name: &str) -> BuiltinResult<PathBuf> {
    let text = scalar_text(value, name, "filename")?;
    Ok(PathBuf::from(
        expand_user_path(text.trim(), name).map_err(|err| line_error(name, err))?,
    ))
}

#[runtime_builtin(
    name = "readlines",
    category = "io/filetext",
    summary = "Read text file lines into a string array.",
    keywords = "readlines,text,file,string array,lines",
    accel = "cpu",
    type_resolver(crate::builtins::io::type_resolvers::readlines_type),
    descriptor(crate::builtins::io::filetext::lines::READLINES_DESCRIPTOR),
    builtin_path = "crate::builtins::io::filetext::lines"
)]
async fn readlines_builtin(args: Vec<Value>) -> BuiltinResult<Value> {
    let args = gather_args("readlines", &args).await?;
    if args.is_empty() || args.len() > 3 {
        return Err(line_error("readlines", "readlines: filename is required"));
    }
    let path = path_from_value(&args[0], "readlines")?;
    let text = vfs::read_to_string_async(&path)
        .await
        .map_err(|err| line_error("readlines", format!("readlines: {err}")))?;
    let mut lines = split_lines_preserving_content(&text);
    if parse_empty_line_rule(&args[1..])? {
        lines.retain(|line| !line.is_empty());
    }
    let rows = lines.len();
    Ok(Value::StringArray(
        StringArray::new(lines, vec![rows, 1]).map_err(|err| line_error("readlines", err))?,
    ))
}

fn split_lines_preserving_content(text: &str) -> Vec<String> {
    if text.is_empty() {
        return Vec::new();
    }
    let normalized = text.replace("\r\n", "\n").replace('\r', "\n");
    let mut lines: Vec<String> = normalized.split('\n').map(str::to_string).collect();
    if normalized.ends_with('\n') {
        lines.pop();
    }
    lines
}

fn parse_empty_line_rule(args: &[Value]) -> BuiltinResult<bool> {
    let mut omit_empty = false;
    let mut idx = 0usize;
    while idx < args.len() {
        if idx + 1 >= args.len() {
            return Err(line_error(
                "readlines",
                "readlines: name-value options must be paired",
            ));
        }
        let name = scalar_text(&args[idx], "readlines", "option")?;
        if name.eq_ignore_ascii_case("EmptyLineRule") {
            let value = scalar_text(&args[idx + 1], "readlines", "EmptyLineRule")?;
            omit_empty = value.eq_ignore_ascii_case("skip")
                || value.eq_ignore_ascii_case("omit")
                || value.eq_ignore_ascii_case("omitempty");
        }
        idx += 2;
    }
    Ok(omit_empty)
}

#[runtime_builtin(
    name = "writelines",
    category = "io/filetext",
    summary = "Write strings or text lines to a file.",
    keywords = "writelines,text,file,string array,lines",
    accel = "cpu",
    type_resolver(crate::builtins::io::type_resolvers::num_type),
    descriptor(crate::builtins::io::filetext::lines::WRITELINES_DESCRIPTOR),
    builtin_path = "crate::builtins::io::filetext::lines"
)]
async fn writelines_builtin(args: Vec<Value>) -> BuiltinResult<Value> {
    let args = gather_args("writelines", &args).await?;
    if args.len() < 2 {
        return Err(line_error(
            "writelines",
            "writelines: text and filename are required",
        ));
    }
    let lines = lines_from_value(&args[0])?;
    let path = path_from_value(&args[1], "writelines")?;
    let newline = parse_newline(&args[2..])?.unwrap_or("\n".to_string());
    let mut output = lines.join(&newline);
    output.push_str(&newline);
    vfs::write_async(&path, output.as_bytes())
        .await
        .map_err(|err| line_error("writelines", format!("writelines: {err}")))?;
    Ok(Value::Num(output.len() as f64))
}

fn lines_from_value(value: &Value) -> BuiltinResult<Vec<String>> {
    match value {
        Value::String(text) => Ok(vec![text.clone()]),
        Value::CharArray(array) if array.rows == 1 => Ok(vec![array.data.iter().collect()]),
        Value::CharArray(array) => {
            let mut lines = Vec::with_capacity(array.rows);
            for row in 0..array.rows {
                let start = row * array.cols;
                lines.push(array.data[start..start + array.cols].iter().collect());
            }
            Ok(lines)
        }
        Value::StringArray(array) => Ok(array.data.clone()),
        Value::Cell(array) => cell_lines(array),
        _ => Err(line_error(
            "writelines",
            "writelines: text must be string, char, string array, or cell text",
        )),
    }
}

fn cell_lines(array: &CellArray) -> BuiltinResult<Vec<String>> {
    let mut out = Vec::with_capacity(array.data.len());
    for value in &array.data {
        out.push(scalar_text(value, "writelines", "cell element")?);
    }
    Ok(out)
}

fn parse_newline(args: &[Value]) -> BuiltinResult<Option<String>> {
    let mut newline = None;
    let mut idx = 0usize;
    while idx < args.len() {
        if idx + 1 >= args.len() {
            return Err(line_error(
                "writelines",
                "writelines: name-value options must be paired",
            ));
        }
        let name = scalar_text(&args[idx], "writelines", "option")?;
        if name.eq_ignore_ascii_case("LineEnding") {
            let value = scalar_text(&args[idx + 1], "writelines", "LineEnding")?;
            newline = Some(match value.to_ascii_lowercase().as_str() {
                "windows" | "crlf" => "\r\n".to_string(),
                "mac" | "cr" => "\r".to_string(),
                _ => "\n".to_string(),
            });
        }
        idx += 2;
    }
    Ok(newline)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::builtins::io::repl_fs::REPL_FS_TEST_LOCK;

    fn run(value: impl std::future::Future<Output = BuiltinResult<Value>>) -> BuiltinResult<Value> {
        futures::executor::block_on(value)
    }

    #[test]
    fn readlines_returns_column_string_array() {
        let _guard = REPL_FS_TEST_LOCK.lock().unwrap();
        let path = std::env::temp_dir().join("runmat_readlines_test.txt");
        std::fs::write(&path, "a\nb\n").unwrap();
        let value = run(readlines_builtin(vec![Value::String(
            path.to_string_lossy().into_owned(),
        )]))
        .unwrap();
        match value {
            Value::StringArray(array) => {
                assert_eq!(array.data, vec!["a", "b"]);
                assert_eq!(array.shape, vec![2, 1]);
            }
            other => panic!("unexpected value {other:?}"),
        }
        let _ = std::fs::remove_file(path);
    }

    #[test]
    fn writelines_writes_each_string_array_element() {
        let _guard = REPL_FS_TEST_LOCK.lock().unwrap();
        let path = std::env::temp_dir().join("runmat_writelines_test.txt");
        let lines = StringArray::new(vec!["a".to_string(), "b".to_string()], vec![2, 1]).unwrap();
        run(writelines_builtin(vec![
            Value::StringArray(lines),
            Value::String(path.to_string_lossy().into_owned()),
        ]))
        .unwrap();
        assert_eq!(std::fs::read_to_string(&path).unwrap(), "a\nb\n");
        let _ = std::fs::remove_file(path);
    }
}
