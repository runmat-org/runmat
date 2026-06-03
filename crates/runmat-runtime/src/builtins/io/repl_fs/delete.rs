//! MATLAB-compatible `delete` builtin for RunMat.

use runmat_filesystem as vfs;
use std::io;
use std::path::{Path, PathBuf};

use glob::{Pattern, PatternError};
use runmat_builtins::{
    BuiltinCompletionPolicy, BuiltinDescriptor, BuiltinErrorDescriptor, BuiltinOutputMode,
    BuiltinParamArity, BuiltinParamDescriptor, BuiltinParamType, BuiltinSignatureDescriptor,
    CellArray, CharArray, StringArray, Value,
};
use runmat_macros::runtime_builtin;

use crate::builtins::common::fs::{contains_wildcards, expand_user_path, path_to_string};
use crate::builtins::common::spec::{
    BroadcastSemantics, BuiltinFusionSpec, BuiltinGpuSpec, ConstantStrategy, GpuOpKind,
    ReductionNaN, ResidencyPolicy, ShapeRequirements,
};
use crate::{build_runtime_error, gather_if_needed_async, BuiltinResult, RuntimeError};

#[runmat_macros::register_gpu_spec(builtin_path = "crate::builtins::io::repl_fs::delete")]
pub const GPU_SPEC: BuiltinGpuSpec = BuiltinGpuSpec {
    name: "delete",
    op_kind: GpuOpKind::Custom("io"),
    supported_precisions: &[],
    broadcast: BroadcastSemantics::None,
    provider_hooks: &[],
    constant_strategy: ConstantStrategy::InlineLiteral,
    residency: ResidencyPolicy::GatherImmediately,
    nan_mode: ReductionNaN::Include,
    two_pass_threshold: None,
    workgroup_size: None,
    accepts_nan_mode: false,
    notes:
        "Host-only filesystem operation. GPU-resident path values are gathered automatically before deletion.",
};

#[runmat_macros::register_fusion_spec(builtin_path = "crate::builtins::io::repl_fs::delete")]
pub const FUSION_SPEC: BuiltinFusionSpec = BuiltinFusionSpec {
    name: "delete",
    shape: ShapeRequirements::Any,
    constant_strategy: ConstantStrategy::InlineLiteral,
    elementwise: None,
    reduction: None,
    emits_nan: false,
    notes:
        "Filesystem side-effects are executed immediately; metadata registered for completeness.",
};

const BUILTIN_NAME: &str = "delete";

const DELETE_OUTPUT: [BuiltinParamDescriptor; 1] = [BuiltinParamDescriptor {
    name: "status",
    ty: BuiltinParamType::NumericScalar,
    arity: BuiltinParamArity::Required,
    default: None,
    description: "Always 0 on success; function primarily acts as a sink.",
}];
const DELETE_INPUTS_ONE: [BuiltinParamDescriptor; 1] = [BuiltinParamDescriptor {
    name: "filename",
    ty: BuiltinParamType::Any,
    arity: BuiltinParamArity::Required,
    default: None,
    description: "Filename/pattern string, string array, char matrix, cell array of path strings, or handle input.",
}];
const DELETE_INPUTS_VARIADIC: [BuiltinParamDescriptor; 2] = [
    BuiltinParamDescriptor {
        name: "filename1",
        ty: BuiltinParamType::Any,
        arity: BuiltinParamArity::Required,
        default: None,
        description: "First filename/pattern/handle input.",
    },
    BuiltinParamDescriptor {
        name: "filenameN",
        ty: BuiltinParamType::Any,
        arity: BuiltinParamArity::Variadic,
        default: None,
        description: "Additional filename/pattern/handle inputs.",
    },
];
const DELETE_SIGNATURES: [BuiltinSignatureDescriptor; 2] = [
    BuiltinSignatureDescriptor {
        label: "status = delete(filename)",
        inputs: &DELETE_INPUTS_ONE,
        outputs: &DELETE_OUTPUT,
    },
    BuiltinSignatureDescriptor {
        label: "status = delete(filename1, filename2, ...)",
        inputs: &DELETE_INPUTS_VARIADIC,
        outputs: &DELETE_OUTPUT,
    },
];
const DELETE_ERROR_INVALID_INPUT: BuiltinErrorDescriptor = BuiltinErrorDescriptor {
    code: "RM.DELETE.INVALID_INPUT",
    identifier: Some("RunMat:delete:InvalidInput"),
    when: "Input arguments are missing or contain unsupported filename value types.",
    message: "delete: invalid input",
};
const DELETE_ERROR_INVALID_HANDLE: BuiltinErrorDescriptor = BuiltinErrorDescriptor {
    code: "RM.DELETE.INVALID_HANDLE",
    identifier: Some("RunMat:delete:InvalidHandle"),
    when: "Handle deletion inputs are mixed with filename inputs or contain unsupported handle values.",
    message: "delete: invalid handle input",
};
const DELETE_ERROR_EMPTY_FILENAME: BuiltinErrorDescriptor = BuiltinErrorDescriptor {
    code: "RM.DELETE.EMPTY_FILENAME",
    identifier: Some("RunMat:delete:EmptyFilename"),
    when: "A filename input is empty after trimming.",
    message: "delete: filename cannot be empty",
};
const DELETE_ERROR_INVALID_PATTERN: BuiltinErrorDescriptor = BuiltinErrorDescriptor {
    code: "RM.DELETE.INVALID_PATTERN",
    identifier: Some("RunMat:delete:InvalidPattern"),
    when: "Wildcard pattern parsing or structure validation fails.",
    message: "delete: invalid wildcard pattern",
};
const DELETE_ERROR_FILE_NOT_FOUND: BuiltinErrorDescriptor = BuiltinErrorDescriptor {
    code: "RM.DELETE.FILE_NOT_FOUND",
    identifier: Some("RunMat:DELETE:FileNotFound"),
    when: "Target path does not exist or pattern matches no files.",
    message: "delete: file not found",
};
const DELETE_ERROR_IS_DIRECTORY: BuiltinErrorDescriptor = BuiltinErrorDescriptor {
    code: "RM.DELETE.IS_DIRECTORY",
    identifier: Some("RunMat:delete:Directories"),
    when: "Target path is a directory instead of a file.",
    message: "delete: cannot delete directories",
};
const DELETE_ERROR_OS_ERROR: BuiltinErrorDescriptor = BuiltinErrorDescriptor {
    code: "RM.DELETE.OS_ERROR",
    identifier: Some("RunMat:DELETE:PermissionDenied"),
    when: "Underlying filesystem operation fails while deleting files.",
    message: "delete: filesystem deletion failed",
};
const DELETE_ERRORS: [BuiltinErrorDescriptor; 7] = [
    DELETE_ERROR_INVALID_INPUT,
    DELETE_ERROR_INVALID_HANDLE,
    DELETE_ERROR_EMPTY_FILENAME,
    DELETE_ERROR_INVALID_PATTERN,
    DELETE_ERROR_FILE_NOT_FOUND,
    DELETE_ERROR_IS_DIRECTORY,
    DELETE_ERROR_OS_ERROR,
];
pub const DELETE_DESCRIPTOR: BuiltinDescriptor = BuiltinDescriptor {
    signatures: &DELETE_SIGNATURES,
    output_mode: BuiltinOutputMode::Fixed,
    completion_policy: BuiltinCompletionPolicy::Public,
    errors: &DELETE_ERRORS,
};

fn delete_error_with(
    error: &'static BuiltinErrorDescriptor,
    message: impl Into<String>,
) -> RuntimeError {
    let mut builder = build_runtime_error(message).with_builtin(BUILTIN_NAME);
    if let Some(identifier) = error.identifier {
        builder = builder.with_identifier(identifier);
    }
    builder.build()
}

fn map_control_flow(err: RuntimeError) -> RuntimeError {
    let identifier = err.identifier().map(str::to_string);
    let mut builder = build_runtime_error(format!("{BUILTIN_NAME}: {}", err.message()))
        .with_builtin(BUILTIN_NAME)
        .with_source(err);
    if let Some(identifier) = identifier {
        builder = builder.with_identifier(identifier);
    }
    builder.build()
}

#[runtime_builtin(
    name = "delete",
    category = "io/repl_fs",
    summary = "Remove files.",
    keywords = "delete,remove file,wildcard delete,cleanup,temporary files,MATLAB delete",
    accel = "cpu",
    sink = true,
    suppress_auto_output = true,
    type_resolver(crate::builtins::io::type_resolvers::delete_type),
    descriptor(crate::builtins::io::repl_fs::delete::DELETE_DESCRIPTOR),
    builtin_path = "crate::builtins::io::repl_fs::delete"
)]
async fn delete_builtin(args: Vec<Value>) -> crate::BuiltinResult<Value> {
    if args.is_empty() {
        return Err(delete_error_with(
            &DELETE_ERROR_INVALID_INPUT,
            "delete: missing filename input",
        ));
    }
    let gathered = gather_arguments(&args).await?;

    if gathered.iter().all(is_handle_input) {
        return delete_handles(&gathered).await;
    }

    if gathered.iter().any(contains_handle_input) {
        return Err(delete_error_with(
            &DELETE_ERROR_INVALID_HANDLE,
            "delete: cannot mix handle and filename inputs",
        ));
    }

    let mut raw_targets = Vec::new();
    for value in &gathered {
        collect_targets(value, &mut raw_targets)?;
    }

    if raw_targets.is_empty() {
        return Ok(Value::Num(0.0));
    }

    for raw in raw_targets {
        delete_target(&raw).await?;
    }

    Ok(Value::Num(0.0))
}

async fn delete_target(raw: &str) -> BuiltinResult<()> {
    let expanded = expand_user_path(raw, "delete")
        .map_err(|msg| delete_error_with(&DELETE_ERROR_INVALID_INPUT, msg))?;
    if expanded.is_empty() {
        return Err(delete_error_with(
            &DELETE_ERROR_EMPTY_FILENAME,
            DELETE_ERROR_EMPTY_FILENAME.message,
        ));
    }

    if contains_wildcards(&expanded) {
        delete_with_pattern(&expanded, raw).await
    } else {
        delete_single_path_async(&PathBuf::from(&expanded), raw).await
    }
}

async fn delete_with_pattern(pattern: &str, display: &str) -> BuiltinResult<()> {
    validate_wildcard_pattern(pattern, display)?;

    if let Err(PatternError { msg, .. }) = Pattern::new(pattern) {
        return Err(delete_error_with(
            &DELETE_ERROR_INVALID_PATTERN,
            format!("delete: invalid wildcard pattern '{display}' ({msg})"),
        ));
    }

    let paths = match glob::glob(pattern) {
        Ok(iter) => iter,
        Err(PatternError { msg, .. }) => {
            return Err(delete_error_with(
                &DELETE_ERROR_INVALID_PATTERN,
                format!("delete: invalid wildcard pattern '{display}' ({msg})"),
            ))
        }
    };

    let mut matches = Vec::new();
    for entry in paths {
        match entry {
            Ok(path) => matches.push(path),
            Err(err) => {
                let problem_path = path_to_string(err.path());
                return Err(delete_error_with(
                    &DELETE_ERROR_OS_ERROR,
                    format!(
                        "delete: unable to delete '{}' ({})",
                        problem_path,
                        err.error()
                    ),
                ));
            }
        }
    }

    if matches.is_empty() {
        return Err(delete_error_with(
            &DELETE_ERROR_FILE_NOT_FOUND,
            format!(
                "delete: cannot delete '{}' because it does not exist",
                display
            ),
        ));
    }

    for path in matches {
        let display_path = path_to_string(&path);
        delete_single_path_async(&path, &display_path).await?;
    }
    Ok(())
}

async fn delete_single_path_async(path: &Path, display: &str) -> BuiltinResult<()> {
    match vfs::metadata_async(path).await {
        Ok(meta) => {
            if meta.is_dir() {
                return Err(delete_error_with(
                    &DELETE_ERROR_IS_DIRECTORY,
                    format!(
                        "delete: cannot delete '{}' because it is a directory (use rmdir instead)",
                        display
                    ),
                ));
            }
            vfs::remove_file_async(path).await.map_err(|err| {
                delete_error_with(
                    &DELETE_ERROR_OS_ERROR,
                    format!("delete: unable to delete '{}' ({})", display, err),
                )
            })
        }
        Err(err) => {
            if err.kind() == io::ErrorKind::NotFound {
                Err(delete_error_with(
                    &DELETE_ERROR_FILE_NOT_FOUND,
                    format!(
                        "delete: cannot delete '{}' because it does not exist",
                        display
                    ),
                ))
            } else {
                Err(delete_error_with(
                    &DELETE_ERROR_OS_ERROR,
                    format!("delete: unable to delete '{}' ({})", display, err),
                ))
            }
        }
    }
}

#[cfg(test)]
fn delete_single_path(path: &Path, display: &str) -> BuiltinResult<()> {
    futures::executor::block_on(delete_single_path_async(path, display))
}

fn validate_wildcard_pattern(pattern: &str, display: &str) -> BuiltinResult<()> {
    if has_unbalanced(pattern, '[', ']') || has_unbalanced(pattern, '{', '}') {
        return Err(delete_error_with(
            &DELETE_ERROR_INVALID_PATTERN,
            format!("delete: invalid wildcard pattern '{display}'"),
        ));
    }
    Ok(())
}

fn has_unbalanced(pattern: &str, open: char, close: char) -> bool {
    let mut depth = 0usize;
    let mut chars = pattern.chars();
    while let Some(ch) = chars.next() {
        if ch == '\\' {
            // Skip escaped characters to avoid false positives
            let _ = chars.next();
            continue;
        }
        if ch == open {
            depth += 1;
        } else if ch == close {
            if depth == 0 {
                return true;
            }
            depth -= 1;
        }
    }
    depth != 0
}

async fn gather_arguments(args: &[Value]) -> BuiltinResult<Vec<Value>> {
    let mut out = Vec::with_capacity(args.len());
    for value in args {
        out.push(
            gather_if_needed_async(value)
                .await
                .map_err(map_control_flow)?,
        );
    }
    Ok(out)
}

fn collect_targets(value: &Value, targets: &mut Vec<String>) -> BuiltinResult<()> {
    match value {
        Value::String(text) => push_nonempty_target(text, targets),
        Value::CharArray(array) => collect_char_array_targets(array, targets),
        Value::StringArray(array) => collect_string_array_targets(array, targets),
        Value::Cell(cell) => collect_cell_targets(cell, targets),
        _ => Err(delete_error_with(
            &DELETE_ERROR_INVALID_INPUT,
            "delete: filename must be a character vector, string scalar, string array, or cell array of character vectors",
        )),
    }
}

fn collect_char_array_targets(array: &CharArray, targets: &mut Vec<String>) -> BuiltinResult<()> {
    if array.rows == 0 || array.cols == 0 {
        return Ok(());
    }
    for row in 0..array.rows {
        let mut text = String::with_capacity(array.cols);
        for col in 0..array.cols {
            text.push(array.data[row * array.cols + col]);
        }
        let trimmed = text.trim_end().to_string();
        if trimmed.is_empty() {
            return Err(delete_error_with(
                &DELETE_ERROR_EMPTY_FILENAME,
                DELETE_ERROR_EMPTY_FILENAME.message,
            ));
        }
        targets.push(trimmed);
    }
    Ok(())
}

fn collect_string_array_targets(
    array: &StringArray,
    targets: &mut Vec<String>,
) -> BuiltinResult<()> {
    for text in &array.data {
        if text.is_empty() {
            return Err(delete_error_with(
                &DELETE_ERROR_EMPTY_FILENAME,
                DELETE_ERROR_EMPTY_FILENAME.message,
            ));
        }
        targets.push(text.clone());
    }
    Ok(())
}

fn collect_cell_targets(cell: &CellArray, targets: &mut Vec<String>) -> BuiltinResult<()> {
    for handle in &cell.data {
        let value = unsafe { &*handle.as_raw() };
        collect_targets(value, targets)?;
    }
    Ok(())
}

async fn delete_handles(values: &[Value]) -> BuiltinResult<Value> {
    let mut mutated_last: Option<Value> = None;
    let mut total = 0usize;
    for value in values {
        total += process_handle_value(value, &mut mutated_last).await?;
    }
    if total == 1 {
        Ok(mutated_last.unwrap_or(Value::Num(0.0)))
    } else {
        Ok(Value::Num(0.0))
    }
}

async fn process_handle_value(
    value: &Value,
    mutated_last: &mut Option<Value>,
) -> BuiltinResult<usize> {
    match value {
        Value::HandleObject(handle) => {
            if let Some((delete_method, _owner)) =
                runmat_builtins::lookup_method(&handle.class_name, "delete")
            {
                if let Some(result) = crate::user_functions::try_call_semantic_function_by_name(
                    &delete_method.function_name,
                    &[Value::HandleObject(handle.clone())],
                    0,
                )
                .await
                {
                    result?;
                } else {
                    return Err(delete_error_with(
                        &DELETE_ERROR_INVALID_HANDLE,
                        format!(
                            "delete: unresolved handle delete method '{}'",
                            delete_method.function_name
                        ),
                    ));
                }
            }
            let _ = runmat_builtins::set_handle_valid(handle, false);
            let mut invalid = handle.clone();
            invalid.valid = false;
            *mutated_last = Some(Value::HandleObject(invalid));
            Ok(1)
        }
        Value::Listener(listener) => {
            crate::invalidate_listener_registration(listener.id);
            let mut invalid = listener.clone();
            invalid.valid = false;
            invalid.enabled = false;
            *mutated_last = Some(Value::Listener(invalid));
            Ok(1)
        }
        Value::Cell(cell) => {
            let mut total = 0usize;
            for handle in &cell.data {
                let inner = unsafe { &*handle.as_raw() };
                total += Box::pin(process_handle_value(inner, mutated_last)).await?;
            }
            Ok(total)
        }
        other => Err(delete_error_with(
            &DELETE_ERROR_INVALID_HANDLE,
            format!("delete: unsupported handle input {other:?}"),
        )),
    }
}

fn is_handle_input(value: &Value) -> bool {
    match value {
        Value::HandleObject(_) | Value::Listener(_) => true,
        Value::Cell(cell) => cell
            .data
            .iter()
            .all(|ptr| is_handle_input(unsafe { &*ptr.as_raw() })),
        _ => false,
    }
}

fn contains_handle_input(value: &Value) -> bool {
    match value {
        Value::HandleObject(_) | Value::Listener(_) => true,
        Value::Cell(cell) => cell
            .data
            .iter()
            .any(|ptr| contains_handle_input(unsafe { &*ptr.as_raw() })),
        _ => false,
    }
}

fn push_nonempty_target(text: &str, targets: &mut Vec<String>) -> BuiltinResult<()> {
    if text.is_empty() {
        Err(delete_error_with(
            &DELETE_ERROR_EMPTY_FILENAME,
            DELETE_ERROR_EMPTY_FILENAME.message,
        ))
    } else {
        targets.push(text.to_string());
        Ok(())
    }
}

#[cfg(test)]
pub(crate) mod tests {
    use super::super::REPL_FS_TEST_LOCK;
    use super::*;
    use runmat_builtins::{CharArray, StringArray, Value};
    use std::fs::File;
    use tempfile::tempdir;

    fn delete_builtin(args: Vec<Value>) -> BuiltinResult<Value> {
        futures::executor::block_on(super::delete_builtin(args))
    }

    fn ident(error: &'static BuiltinErrorDescriptor) -> Option<&'static str> {
        error.identifier
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn delete_descriptor_signatures_cover_core_forms() {
        let labels: Vec<&str> = DELETE_DESCRIPTOR
            .signatures
            .iter()
            .map(|sig| sig.label)
            .collect();
        assert!(labels.contains(&"status = delete(filename)"));
        assert!(labels.contains(&"status = delete(filename1, filename2, ...)"));
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn delete_removes_single_file() {
        let _lock = REPL_FS_TEST_LOCK
            .lock()
            .unwrap_or_else(|poison| poison.into_inner());

        let temp = tempdir().expect("temp dir");
        let target = temp.path().join("single.txt");
        File::create(&target).expect("create");

        let result = delete_builtin(vec![Value::from(target.to_string_lossy().to_string())])
            .expect("delete");
        assert_eq!(result, Value::Num(0.0));
        assert!(!target.exists());
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn delete_removes_files_with_wildcard() {
        let _lock = REPL_FS_TEST_LOCK
            .lock()
            .unwrap_or_else(|poison| poison.into_inner());

        let temp = tempdir().expect("temp dir");
        let file_a = temp.path().join("log-01.txt");
        let file_b = temp.path().join("log-02.txt");
        File::create(&file_a).expect("create a");
        File::create(&file_b).expect("create b");

        let pattern = temp.path().join("log-*.txt");
        delete_builtin(vec![Value::from(pattern.to_string_lossy().to_string())]).expect("delete");
        assert!(!file_a.exists());
        assert!(!file_b.exists());
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn delete_accepts_string_array() {
        let _lock = REPL_FS_TEST_LOCK
            .lock()
            .unwrap_or_else(|poison| poison.into_inner());

        let temp = tempdir().expect("temp dir");
        let file_a = temp.path().join("stageA.dat");
        let file_b = temp.path().join("stageB.dat");
        File::create(&file_a).expect("create a");
        File::create(&file_b).expect("create b");

        let array = StringArray::new(
            vec![
                file_a.to_string_lossy().to_string(),
                file_b.to_string_lossy().to_string(),
            ],
            vec![2],
        )
        .expect("string array");

        delete_builtin(vec![Value::StringArray(array)]).expect("delete");
        assert!(!file_a.exists());
        assert!(!file_b.exists());
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn delete_accepts_char_array() {
        let _lock = REPL_FS_TEST_LOCK
            .lock()
            .unwrap_or_else(|poison| poison.into_inner());

        let temp = tempdir().expect("temp dir");
        let paths: Vec<_> = ["stageA.tmp", "stageB.tmp"]
            .into_iter()
            .map(|name| temp.path().join(name))
            .collect();
        let path_strings: Vec<String> = paths
            .iter()
            .map(|p| p.to_string_lossy().to_string())
            .collect();
        let max_len = path_strings.iter().map(|s| s.len()).max().unwrap();
        let mut data: Vec<char> = Vec::with_capacity(path_strings.len() * max_len);

        for (path, path_string) in paths.iter().zip(path_strings.iter()) {
            File::create(path).expect("create file");
            let mut chars: Vec<char> = path_string.chars().collect();
            while chars.len() < max_len {
                chars.push(' ');
            }
            data.extend(&chars);
        }

        let char_array = CharArray::new(data, path_strings.len(), max_len).expect("char array");
        delete_builtin(vec![Value::CharArray(char_array)]).expect("delete");

        for path in paths {
            assert!(!path.exists(), "{path:?} should be removed");
        }
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn delete_accepts_cell_array_of_paths() {
        let _lock = REPL_FS_TEST_LOCK
            .lock()
            .unwrap_or_else(|poison| poison.into_inner());

        let temp = tempdir().expect("temp dir");
        let file_a = temp.path().join("cellA.dat");
        let file_b = temp.path().join("cellB.dat");
        File::create(&file_a).expect("create cellA");
        File::create(&file_b).expect("create cellB");

        let cell_value = crate::make_cell(
            vec![
                Value::from(file_a.to_string_lossy().to_string()),
                Value::from(file_b.to_string_lossy().to_string()),
            ],
            1,
            2,
        )
        .expect("cell");

        delete_builtin(vec![cell_value]).expect("delete");
        assert!(!file_a.exists());
        assert!(!file_b.exists());
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn delete_empty_string_array_is_noop() {
        let array = StringArray::new(Vec::<String>::new(), vec![0]).expect("empty array");
        let result = delete_builtin(vec![Value::StringArray(array)]).expect("delete");
        assert_eq!(result, Value::Num(0.0));
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn delete_errors_on_empty_string_argument() {
        let err = delete_builtin(vec![Value::from(String::new())]).expect_err("empty string");
        assert_eq!(err.identifier(), ident(&DELETE_ERROR_EMPTY_FILENAME));
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn delete_errors_on_string_array_empty_element() {
        let array =
            StringArray::new(vec![String::new()], vec![1]).expect("single empty string element");
        let err = delete_builtin(vec![Value::StringArray(array)]).expect_err("empty element");
        assert_eq!(err.identifier(), ident(&DELETE_ERROR_EMPTY_FILENAME));
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn delete_errors_on_char_array_blank_row() {
        let data = vec![' '; 4];
        let char_array = CharArray::new(data, 1, 4).expect("char matrix");
        let err = delete_builtin(vec![Value::CharArray(char_array)]).expect_err("blank row");
        assert_eq!(err.identifier(), ident(&DELETE_ERROR_EMPTY_FILENAME));
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn delete_errors_on_invalid_pattern() {
        let pattern = "{invalid*";
        let err = futures::executor::block_on(delete_target(pattern))
            .expect_err("invalid pattern should error");
        assert_eq!(err.identifier(), ident(&DELETE_ERROR_INVALID_PATTERN));
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn delete_errors_on_missing_file() {
        let _lock = REPL_FS_TEST_LOCK
            .lock()
            .unwrap_or_else(|poison| poison.into_inner());

        let temp = tempdir().expect("temp dir");
        let missing = temp.path().join("missing.txt");
        let missing_str = missing.to_string_lossy().to_string();
        let err = futures::executor::block_on(delete_target(&missing_str)).expect_err("error");
        assert_eq!(err.identifier(), ident(&DELETE_ERROR_FILE_NOT_FOUND));
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn delete_errors_on_directory() {
        let _lock = REPL_FS_TEST_LOCK
            .lock()
            .unwrap_or_else(|poison| poison.into_inner());

        let temp = tempdir().expect("temp dir");
        let dir = temp.path().join("dir");
        std::fs::create_dir(&dir).expect("create dir");
        let dir_display = dir.to_string_lossy().to_string();
        let err = delete_single_path(&dir, &dir_display).expect_err("error");
        assert_eq!(err.identifier(), ident(&DELETE_ERROR_IS_DIRECTORY));
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn delete_handle_returns_invalid_handle() {
        let handle = futures::executor::block_on(crate::new_handle_object_builtin(
            "ReplFsDeleteTestHandle".to_string(),
        ))
        .expect("handle");
        let original_handle = handle.clone();
        let result = delete_builtin(vec![handle]).expect("delete handle");
        match result {
            Value::HandleObject(h) => {
                assert!(!h.valid, "handle should be marked invalid");
                let valid_value = futures::executor::block_on(crate::isvalid_builtin(
                    Value::HandleObject(h.clone()),
                ))
                .expect("isvalid");
                match valid_value {
                    Value::Bool(flag) => assert!(!flag, "isvalid should report false after delete"),
                    other => panic!("expected bool from isvalid, got {other:?}"),
                }
                let original_valid_value =
                    futures::executor::block_on(crate::isvalid_builtin(original_handle))
                        .expect("isvalid");
                match original_valid_value {
                    Value::Bool(flag) => assert!(!flag, "original alias should also be invalid"),
                    other => panic!("expected bool from isvalid, got {other:?}"),
                }
            }
            other => panic!("expected handle result, got {other:?}"),
        }
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn delete_rejects_mixed_handle_and_filename() {
        let handle = futures::executor::block_on(crate::new_handle_object_builtin(
            "ReplFsDeleteTestHandle".to_string(),
        ))
        .expect("handle");
        let err = delete_builtin(vec![
            handle,
            Value::from("mixed-handle-path.txt".to_string()),
        ])
        .expect_err("expected mixed error");
        assert_eq!(err.identifier(), ident(&DELETE_ERROR_INVALID_HANDLE));
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn delete_accepts_cell_of_handles() {
        let handle_a = futures::executor::block_on(crate::new_handle_object_builtin(
            "ReplFsDeleteTestHandle".to_string(),
        ))
        .expect("handle");
        let handle_b = futures::executor::block_on(crate::new_handle_object_builtin(
            "ReplFsDeleteTestHandle".to_string(),
        ))
        .expect("handle");
        let cell = crate::make_cell(vec![handle_a, handle_b], 1, 2).expect("cell of handles");
        let result = delete_builtin(vec![cell]).expect("delete handles");
        assert_eq!(result, Value::Num(0.0));
    }

    #[cfg(feature = "wgpu")]
    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn delete_runs_with_wgpu_provider_registered() {
        let _lock = REPL_FS_TEST_LOCK
            .lock()
            .unwrap_or_else(|poison| poison.into_inner());

        let _ = runmat_accelerate::backend::wgpu::provider::register_wgpu_provider(
            runmat_accelerate::backend::wgpu::provider::WgpuProviderOptions::default(),
        );

        let temp = tempdir().expect("temp dir");
        let path = temp.path().join("wgpu-file.txt");
        File::create(&path).expect("create file");

        delete_builtin(vec![Value::from(path.to_string_lossy().to_string())]).expect("delete");
        assert!(!path.exists());
    }
}
