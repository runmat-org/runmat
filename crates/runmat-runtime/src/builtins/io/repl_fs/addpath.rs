//! MATLAB-compatible `addpath` builtin for manipulating the RunMat search path.

#[cfg(test)]
use runmat_builtins::CellArray;
use runmat_builtins::{
    BuiltinCompletionPolicy, BuiltinDescriptor, BuiltinErrorDescriptor, BuiltinOutputMode,
    BuiltinParamArity, BuiltinParamDescriptor, BuiltinParamType, BuiltinSignatureDescriptor,
    CharArray, StringArray, Tensor, Value,
};
use runmat_macros::runtime_builtin;

use crate::builtins::common::fs::{expand_user_path, path_to_string};
use crate::builtins::common::path_state::{
    current_path_segments, current_path_string, set_path_string, PATH_LIST_SEPARATOR,
};
use crate::builtins::common::spec::{
    BroadcastSemantics, BuiltinFusionSpec, BuiltinGpuSpec, ConstantStrategy, GpuOpKind,
    ReductionNaN, ResidencyPolicy, ShapeRequirements,
};
use crate::{build_runtime_error, gather_if_needed_async, BuiltinResult, RuntimeError};

use runmat_filesystem as vfs;
use std::collections::HashSet;
use std::path::{Component, Path, PathBuf};

#[runmat_macros::register_gpu_spec(builtin_path = "crate::builtins::io::repl_fs::addpath")]
pub const GPU_SPEC: BuiltinGpuSpec = BuiltinGpuSpec {
    name: "addpath",
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
    notes: "Search-path manipulation is a host-only operation; GPU inputs are gathered before processing.",
};

#[runmat_macros::register_fusion_spec(builtin_path = "crate::builtins::io::repl_fs::addpath")]
pub const FUSION_SPEC: BuiltinFusionSpec = BuiltinFusionSpec {
    name: "addpath",
    shape: ShapeRequirements::Any,
    constant_strategy: ConstantStrategy::InlineLiteral,
    elementwise: None,
    reduction: None,
    emits_nan: false,
    notes: "IO builtins are not eligible for fusion; metadata registered for completeness.",
};

const BUILTIN_NAME: &str = "addpath";

const ADDPATH_OUTPUT: [BuiltinParamDescriptor; 1] = [BuiltinParamDescriptor {
    name: "oldpath",
    ty: BuiltinParamType::StringScalar,
    arity: BuiltinParamArity::Required,
    default: None,
    description: "Previous search path string.",
}];
const ADDPATH_INPUTS_FOLDER: [BuiltinParamDescriptor; 1] = [BuiltinParamDescriptor {
    name: "folder1",
    ty: BuiltinParamType::Any,
    arity: BuiltinParamArity::Required,
    default: None,
    description: "Folder, path-list string, or container of folders to add.",
}];
const ADDPATH_INPUTS_FOLDER_VARIADIC: [BuiltinParamDescriptor; 2] = [
    BuiltinParamDescriptor {
        name: "folder1",
        ty: BuiltinParamType::Any,
        arity: BuiltinParamArity::Required,
        default: None,
        description: "First folder argument.",
    },
    BuiltinParamDescriptor {
        name: "folderN",
        ty: BuiltinParamType::Any,
        arity: BuiltinParamArity::Variadic,
        default: None,
        description: "Additional folder arguments.",
    },
];
const ADDPATH_INPUTS_WITH_POSITION: [BuiltinParamDescriptor; 3] = [
    BuiltinParamDescriptor {
        name: "folder1",
        ty: BuiltinParamType::Any,
        arity: BuiltinParamArity::Required,
        default: None,
        description: "First folder argument.",
    },
    BuiltinParamDescriptor {
        name: "folders",
        ty: BuiltinParamType::Any,
        arity: BuiltinParamArity::Variadic,
        default: None,
        description: "Additional folder arguments.",
    },
    BuiltinParamDescriptor {
        name: "position",
        ty: BuiltinParamType::StringScalar,
        arity: BuiltinParamArity::Required,
        default: Some("\"-begin\""),
        description: "Insertion position flag: \"-begin\" or \"-end\".",
    },
];
const ADDPATH_INPUTS_WITH_FROZEN: [BuiltinParamDescriptor; 3] = [
    BuiltinParamDescriptor {
        name: "folder1",
        ty: BuiltinParamType::Any,
        arity: BuiltinParamArity::Required,
        default: None,
        description: "First folder argument.",
    },
    BuiltinParamDescriptor {
        name: "folders",
        ty: BuiltinParamType::Any,
        arity: BuiltinParamArity::Variadic,
        default: None,
        description: "Additional folder arguments.",
    },
    BuiltinParamDescriptor {
        name: "frozen",
        ty: BuiltinParamType::StringScalar,
        arity: BuiltinParamArity::Required,
        default: Some("\"-frozen\""),
        description: "Compatibility flag accepted but currently ignored.",
    },
];
const ADDPATH_INPUTS_WITH_POSITION_AND_FROZEN: [BuiltinParamDescriptor; 4] = [
    BuiltinParamDescriptor {
        name: "folder1",
        ty: BuiltinParamType::Any,
        arity: BuiltinParamArity::Required,
        default: None,
        description: "First folder argument.",
    },
    BuiltinParamDescriptor {
        name: "folders",
        ty: BuiltinParamType::Any,
        arity: BuiltinParamArity::Variadic,
        default: None,
        description: "Additional folder arguments.",
    },
    BuiltinParamDescriptor {
        name: "position",
        ty: BuiltinParamType::StringScalar,
        arity: BuiltinParamArity::Required,
        default: Some("\"-begin\""),
        description: "Insertion position flag: \"-begin\" or \"-end\".",
    },
    BuiltinParamDescriptor {
        name: "frozen",
        ty: BuiltinParamType::StringScalar,
        arity: BuiltinParamArity::Required,
        default: Some("\"-frozen\""),
        description: "Compatibility flag accepted but currently ignored.",
    },
];
const ADDPATH_SIGNATURES: [BuiltinSignatureDescriptor; 5] = [
    BuiltinSignatureDescriptor {
        label: "oldpath = addpath(folder1)",
        inputs: &ADDPATH_INPUTS_FOLDER,
        outputs: &ADDPATH_OUTPUT,
    },
    BuiltinSignatureDescriptor {
        label: "oldpath = addpath(folder1, folder2, ...)",
        inputs: &ADDPATH_INPUTS_FOLDER_VARIADIC,
        outputs: &ADDPATH_OUTPUT,
    },
    BuiltinSignatureDescriptor {
        label: "oldpath = addpath(folder1, ..., position)",
        inputs: &ADDPATH_INPUTS_WITH_POSITION,
        outputs: &ADDPATH_OUTPUT,
    },
    BuiltinSignatureDescriptor {
        label: "oldpath = addpath(folder1, ..., \"-frozen\")",
        inputs: &ADDPATH_INPUTS_WITH_FROZEN,
        outputs: &ADDPATH_OUTPUT,
    },
    BuiltinSignatureDescriptor {
        label: "oldpath = addpath(folder1, ..., position, \"-frozen\")",
        inputs: &ADDPATH_INPUTS_WITH_POSITION_AND_FROZEN,
        outputs: &ADDPATH_OUTPUT,
    },
];

const ADDPATH_ERROR_ARG_TYPE: BuiltinErrorDescriptor = BuiltinErrorDescriptor {
    code: "RM.ADDPATH.ARG_TYPE",
    identifier: None,
    when: "Folder arguments are not character vectors, string scalars/arrays, tensors of character codes, or cell arrays containing those forms.",
    message:
        "addpath: folder names must be character vectors, string scalars, string arrays, or cell arrays of character vectors",
};
const ADDPATH_ERROR_TOO_FEW_ARGS: BuiltinErrorDescriptor = BuiltinErrorDescriptor {
    code: "RM.ADDPATH.TOO_FEW_ARGS",
    identifier: None,
    when: "No folder arguments are provided, or all provided folder tokens are empty/options only.",
    message: "addpath: at least one folder must be specified",
};
const ADDPATH_ERROR_POSITION_REPEATED: BuiltinErrorDescriptor = BuiltinErrorDescriptor {
    code: "RM.ADDPATH.POSITION_REPEATED",
    identifier: None,
    when: "A position option is repeated or multiple position options are provided.",
    message: "addpath: position option must be '-begin' or '-end' and may only appear once",
};
const ADDPATH_ERROR_PATHDEF_UNSUPPORTED: BuiltinErrorDescriptor = BuiltinErrorDescriptor {
    code: "RM.ADDPATH.PATHDEF_UNSUPPORTED",
    identifier: None,
    when: "A pathdef token is provided; loading pathdef.m is not implemented.",
    message: "addpath: loading pathdef.m is not implemented yet",
};
const ADDPATH_ERROR_CWD_RESOLVE: BuiltinErrorDescriptor = BuiltinErrorDescriptor {
    code: "RM.ADDPATH.CWD_RESOLVE",
    identifier: None,
    when: "Current directory cannot be resolved while normalizing a relative folder.",
    message: "addpath: unable to resolve current directory",
};
const ADDPATH_ERROR_FOLDER_NOT_FOUND: BuiltinErrorDescriptor = BuiltinErrorDescriptor {
    code: "RM.ADDPATH.FOLDER_NOT_FOUND",
    identifier: None,
    when: "A requested folder path does not exist.",
    message: "addpath: folder not found",
};
const ADDPATH_ERROR_NOT_FOLDER: BuiltinErrorDescriptor = BuiltinErrorDescriptor {
    code: "RM.ADDPATH.NOT_FOLDER",
    identifier: None,
    when: "A requested path exists but is not a directory.",
    message: "addpath: path is not a folder",
};
const ADDPATH_ERRORS: [BuiltinErrorDescriptor; 7] = [
    ADDPATH_ERROR_ARG_TYPE,
    ADDPATH_ERROR_TOO_FEW_ARGS,
    ADDPATH_ERROR_POSITION_REPEATED,
    ADDPATH_ERROR_PATHDEF_UNSUPPORTED,
    ADDPATH_ERROR_CWD_RESOLVE,
    ADDPATH_ERROR_FOLDER_NOT_FOUND,
    ADDPATH_ERROR_NOT_FOLDER,
];
pub const ADDPATH_DESCRIPTOR: BuiltinDescriptor = BuiltinDescriptor {
    signatures: &ADDPATH_SIGNATURES,
    output_mode: BuiltinOutputMode::Fixed,
    completion_policy: BuiltinCompletionPolicy::Public,
    errors: &ADDPATH_ERRORS,
};

fn addpath_error(error: &'static BuiltinErrorDescriptor) -> RuntimeError {
    addpath_error_with_message(error.message, error)
}

fn addpath_error_with_message(
    message: impl Into<String>,
    error: &'static BuiltinErrorDescriptor,
) -> RuntimeError {
    let mut builder = build_runtime_error(message).with_builtin(BUILTIN_NAME);
    if let Some(identifier) = error.identifier {
        builder = builder.with_identifier(identifier);
    }
    builder.build()
}

fn addpath_error_with_detail(
    error: &'static BuiltinErrorDescriptor,
    detail: impl AsRef<str>,
) -> RuntimeError {
    addpath_error_with_message(format!("{}: {}", error.message, detail.as_ref()), error)
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

#[derive(Clone, Copy, PartialEq, Eq)]
enum InsertPosition {
    Begin,
    End,
}

struct AddPathSpec {
    directories: Vec<String>,
    position: InsertPosition,
    _frozen: bool,
}

#[runtime_builtin(
    name = "addpath",
    category = "io/repl_fs",
    summary = "Add folders to the MATLAB/RunMat search path for function and script resolution.",
    keywords = "addpath,search path,matlab path,-begin,-end,-frozen",
    accel = "cpu",
    suppress_auto_output = true,
    type_resolver(crate::builtins::io::type_resolvers::addpath_type),
    descriptor(crate::builtins::io::repl_fs::addpath::ADDPATH_DESCRIPTOR),
    builtin_path = "crate::builtins::io::repl_fs::addpath"
)]
async fn addpath_builtin(args: Vec<Value>) -> crate::BuiltinResult<Value> {
    if args.is_empty() {
        return Err(addpath_error(&ADDPATH_ERROR_TOO_FEW_ARGS));
    }

    let gathered = gather_arguments(&args).await?;
    let previous = current_path_string();
    let spec = parse_arguments(&gathered).await?;
    apply_addpath(spec).await?;
    Ok(char_array_value(&previous))
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

async fn parse_arguments(args: &[Value]) -> BuiltinResult<AddPathSpec> {
    let mut position = InsertPosition::Begin;
    let mut position_set = false;
    let mut frozen = false;
    let mut directories = Vec::new();

    for value in args {
        collect_strings(value, &mut directories).await?;
    }

    if directories.is_empty() {
        return Err(addpath_error(&ADDPATH_ERROR_TOO_FEW_ARGS));
    }

    let mut resolved = Vec::new();
    for token in directories {
        let trimmed = token.trim();
        if trimmed.is_empty() {
            continue;
        }
        match parse_option(trimmed) {
            Some(AddPathOption::Begin) => {
                if position_set {
                    return Err(addpath_error(&ADDPATH_ERROR_POSITION_REPEATED));
                }
                position = InsertPosition::Begin;
                position_set = true;
            }
            Some(AddPathOption::End) => {
                if position_set {
                    return Err(addpath_error(&ADDPATH_ERROR_POSITION_REPEATED));
                }
                position = InsertPosition::End;
                position_set = true;
            }
            Some(AddPathOption::Frozen) => {
                frozen = true;
            }
            None => {
                for segment in split_path_list(trimmed) {
                    resolved.push(segment);
                }
            }
        }
    }

    if resolved.is_empty() {
        return Err(addpath_error(&ADDPATH_ERROR_TOO_FEW_ARGS));
    }

    Ok(AddPathSpec {
        directories: resolved,
        position,
        _frozen: frozen,
    })
}

enum AddPathOption {
    Begin,
    End,
    Frozen,
}

fn parse_option(text: &str) -> Option<AddPathOption> {
    let lowered = text.trim().to_ascii_lowercase();
    match lowered.as_str() {
        "-begin" => Some(AddPathOption::Begin),
        "-end" => Some(AddPathOption::End),
        "-frozen" => Some(AddPathOption::Frozen),
        _ => None,
    }
}

async fn apply_addpath(spec: AddPathSpec) -> BuiltinResult<()> {
    let mut existing = current_path_segments();
    let mut seen = HashSet::new();
    let mut additions = Vec::new();

    for raw in spec.directories {
        let normalized = normalize_directory(&raw).await?;
        let key = path_identity(&normalized);
        if seen.insert(key.clone()) {
            existing.retain(|entry| path_identity(entry) != key);
            additions.push(normalized);
        }
    }

    if additions.is_empty() {
        return Ok(());
    }

    let final_segments = match spec.position {
        InsertPosition::Begin => {
            let mut combined = additions;
            combined.extend(existing);
            combined
        }
        InsertPosition::End => {
            let mut combined = existing;
            combined.extend(additions);
            combined
        }
    };

    let final_segments = final_segments
        .into_iter()
        .filter(|segment| !segment.is_empty())
        .collect::<Vec<_>>();

    // Preserve empty path (clears search path) if all entries were removed.
    let new_path = if final_segments.is_empty() {
        String::new()
    } else {
        join_segments(&final_segments)
    };

    set_path_string(&new_path);
    Ok(())
}

#[async_recursion::async_recursion(?Send)]
async fn collect_strings(value: &Value, output: &mut Vec<String>) -> BuiltinResult<()> {
    match value {
        Value::String(text) => {
            output.push(text.clone());
            Ok(())
        }
        Value::StringArray(StringArray { data, .. }) => {
            for entry in data {
                output.push(entry.clone());
            }
            Ok(())
        }
        Value::CharArray(chars) => {
            if chars.rows == 1 {
                output.push(chars.data.iter().collect());
                return Ok(());
            }
            for row in 0..chars.rows {
                let mut line = String::with_capacity(chars.cols);
                for col in 0..chars.cols {
                    line.push(chars.data[row * chars.cols + col]);
                }
                output.push(line.trim_end().to_string());
            }
            Ok(())
        }
        Value::Tensor(tensor) => {
            output.push(tensor_to_string(tensor)?);
            Ok(())
        }
        Value::Cell(cell) => {
            for ptr in &cell.data {
                let inner = (ptr).clone();
                let gathered = gather_if_needed_async(&inner)
                    .await
                    .map_err(map_control_flow)?;
                collect_strings(&gathered, output).await?;
            }
            Ok(())
        }
        Value::GpuTensor(_) => Err(addpath_error(&ADDPATH_ERROR_ARG_TYPE)),
        _ => Err(addpath_error(&ADDPATH_ERROR_ARG_TYPE)),
    }
}

fn split_path_list(text: &str) -> Vec<String> {
    text.split(PATH_LIST_SEPARATOR)
        .map(|segment| segment.trim())
        .filter(|segment| !segment.is_empty())
        .map(|segment| segment.to_string())
        .collect()
}

async fn normalize_directory(raw: &str) -> BuiltinResult<String> {
    let trimmed = raw.trim();
    if trimmed.is_empty() {
        return Err(addpath_error(&ADDPATH_ERROR_ARG_TYPE));
    }

    if trimmed.eq_ignore_ascii_case("pathdef") || trimmed.eq_ignore_ascii_case("pathdef.m") {
        return Err(addpath_error(&ADDPATH_ERROR_PATHDEF_UNSUPPORTED));
    }

    let expanded = expand_user_path(trimmed, "addpath")
        .map_err(|err| addpath_error_with_detail(&ADDPATH_ERROR_FOLDER_NOT_FOUND, err))?;
    let path = Path::new(&expanded);
    let joined = if super::is_rooted_path(path) {
        path.to_path_buf()
    } else {
        vfs::current_dir()
            .map_err(|_| addpath_error(&ADDPATH_ERROR_CWD_RESOLVE))?
            .join(path)
    };
    let normalized = normalize_pathbuf(&joined);

    let metadata = vfs::metadata_async(&normalized)
        .await
        .map_err(|_| addpath_error_with_detail(&ADDPATH_ERROR_FOLDER_NOT_FOUND, trimmed))?;
    if !metadata.is_dir() {
        return Err(addpath_error_with_detail(
            &ADDPATH_ERROR_NOT_FOLDER,
            trimmed,
        ));
    }

    Ok(path_to_string(&normalized))
}

fn normalize_pathbuf(path: &Path) -> PathBuf {
    let mut normalized = PathBuf::new();
    for component in path.components() {
        match component {
            Component::Prefix(prefix) => {
                normalized.push(prefix.as_os_str());
            }
            Component::RootDir => {
                normalized.push(component.as_os_str());
            }
            Component::CurDir => {}
            Component::ParentDir => {
                normalized.pop();
            }
            Component::Normal(part) => {
                normalized.push(part);
            }
        }
    }
    if normalized.as_os_str().is_empty() {
        path.to_path_buf()
    } else {
        normalized
    }
}

fn tensor_to_string(tensor: &Tensor) -> BuiltinResult<String> {
    if tensor.shape.len() > 2 {
        return Err(addpath_error(&ADDPATH_ERROR_ARG_TYPE));
    }
    if tensor.rows() > 1 {
        return Err(addpath_error(&ADDPATH_ERROR_ARG_TYPE));
    }
    let mut text = String::with_capacity(tensor.data.len());
    for &code in &tensor.data {
        if !code.is_finite() {
            return Err(addpath_error(&ADDPATH_ERROR_ARG_TYPE));
        }
        let rounded = code.round();
        if (code - rounded).abs() > 1e-6 {
            return Err(addpath_error(&ADDPATH_ERROR_ARG_TYPE));
        }
        let int_code = rounded as i64;
        if !(0..=0x10FFFF).contains(&int_code) {
            return Err(addpath_error(&ADDPATH_ERROR_ARG_TYPE));
        }
        let ch = char::from_u32(int_code as u32)
            .ok_or_else(|| addpath_error(&ADDPATH_ERROR_ARG_TYPE))?;
        text.push(ch);
    }
    Ok(text)
}

fn path_identity(path: &str) -> String {
    #[cfg(windows)]
    {
        path.replace('/', "\\").to_ascii_lowercase()
    }
    #[cfg(not(windows))]
    {
        path.to_string()
    }
}

fn join_segments(segments: &[String]) -> String {
    let mut joined = String::new();
    for (idx, segment) in segments.iter().enumerate() {
        if idx > 0 {
            joined.push(PATH_LIST_SEPARATOR);
        }
        joined.push_str(segment);
    }
    joined
}

fn char_array_value(text: &str) -> Value {
    Value::CharArray(CharArray::new_row(text))
}

#[cfg(test)]
pub(crate) mod tests {
    use super::super::REPL_FS_TEST_LOCK;
    use super::*;
    use crate::builtins::common::path_state::set_path_string;
    use crate::builtins::common::path_state::{current_path_segments, PATH_LIST_SEPARATOR};
    use std::convert::TryFrom;
    use std::fs;
    use tempfile::tempdir;

    fn addpath_builtin(args: Vec<Value>) -> BuiltinResult<Value> {
        futures::executor::block_on(super::addpath_builtin(args))
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn addpath_descriptor_signatures_cover_core_forms() {
        let labels: Vec<&str> = ADDPATH_DESCRIPTOR
            .signatures
            .iter()
            .map(|sig| sig.label)
            .collect();
        assert!(labels.contains(&"oldpath = addpath(folder1)"));
        assert!(labels.contains(&"oldpath = addpath(folder1, folder2, ...)"));
        assert!(labels.contains(&"oldpath = addpath(folder1, ..., position)"));
        assert!(labels.contains(&"oldpath = addpath(folder1, ..., \"-frozen\")"));
        assert!(labels.contains(&"oldpath = addpath(folder1, ..., position, \"-frozen\")"));
    }

    struct PathGuard {
        previous: String,
    }

    impl PathGuard {
        fn new() -> Self {
            Self {
                previous: current_path_string(),
            }
        }
    }

    impl Drop for PathGuard {
        fn drop(&mut self) {
            set_path_string(&self.previous);
        }
    }

    fn canonical(dir: &Path) -> String {
        let normalized = normalize_pathbuf(dir);
        path_to_string(&normalized)
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn addpath_prepends_by_default() {
        let _lock = REPL_FS_TEST_LOCK
            .lock()
            .unwrap_or_else(|poison| poison.into_inner());
        let _guard = PathGuard::new();

        let base_dir = tempdir().expect("tempdir");
        let extra_dir = tempdir().expect("extra dir");

        let base_string = path_to_string(base_dir.path());
        set_path_string(&base_string);

        let input = Value::CharArray(CharArray::new_row(
            extra_dir.path().to_string_lossy().as_ref(),
        ));
        let returned = addpath_builtin(vec![input]).expect("addpath");
        let returned_str = String::try_from(&returned).expect("convert");
        assert_eq!(returned_str, base_string);

        let segments = current_path_segments();
        let expected_front = canonical(extra_dir.path());
        assert_eq!(segments.first().unwrap(), &expected_front);
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn addpath_removes_duplicates() {
        let _lock = REPL_FS_TEST_LOCK
            .lock()
            .unwrap_or_else(|poison| poison.into_inner());
        let _guard = PathGuard::new();

        let first = tempdir().expect("first");
        let second = tempdir().expect("second");
        let first_str = canonical(first.path());
        let second_str = canonical(second.path());
        let combined = format!(
            "{first}{sep}{second}",
            first = first_str,
            second = second_str,
            sep = PATH_LIST_SEPARATOR
        );
        set_path_string(&combined);

        let arg = Value::String(first_str.clone());
        addpath_builtin(vec![arg]).expect("addpath");

        let segments = current_path_segments();
        assert_eq!(segments[0], first_str);
        assert_eq!(segments[1], second_str);
        assert_eq!(segments.iter().filter(|p| *p == &first_str).count(), 1);
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn addpath_respects_end_option() {
        let _lock = REPL_FS_TEST_LOCK
            .lock()
            .unwrap_or_else(|poison| poison.into_inner());
        let _guard = PathGuard::new();

        let first = tempdir().expect("first");
        let second = tempdir().expect("second");
        set_path_string(&canonical(first.path()));

        let args = vec![
            Value::String(second.path().to_string_lossy().into_owned()),
            Value::String("-end".to_string()),
        ];
        addpath_builtin(args).expect("addpath");

        let segments = current_path_segments();
        assert_eq!(segments.last().unwrap(), &canonical(second.path()));
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn addpath_handles_string_array_and_cell_input() {
        let _lock = REPL_FS_TEST_LOCK
            .lock()
            .unwrap_or_else(|poison| poison.into_inner());
        let _guard = PathGuard::new();

        let dir1 = tempdir().expect("dir1");
        let dir2 = tempdir().expect("dir2");

        set_path_string("");

        let strings =
            StringArray::new(vec![dir1.path().to_string_lossy().into_owned()], vec![1, 1])
                .expect("string array");
        let cell = CellArray::new(
            vec![Value::String(dir2.path().to_string_lossy().into_owned())],
            1,
            1,
        )
        .expect("cell");

        addpath_builtin(vec![Value::StringArray(strings), Value::Cell(cell)]).expect("addpath");

        let segments = current_path_segments();
        assert_eq!(segments.len(), 2);
        assert_eq!(segments[0], canonical(dir1.path()));
        assert_eq!(segments[1], canonical(dir2.path()));
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn addpath_supports_multi_row_char_arrays() {
        let _lock = REPL_FS_TEST_LOCK
            .lock()
            .unwrap_or_else(|poison| poison.into_inner());
        let _guard = PathGuard::new();

        let dir1 = tempdir().expect("dir1");
        let dir2 = tempdir().expect("dir2");

        let one = dir1.path().to_string_lossy();
        let two = dir2.path().to_string_lossy();
        let len_one = one.chars().count();
        let len_two = two.chars().count();
        let max_len = len_one.max(len_two);
        let mut data = Vec::with_capacity(2 * max_len);
        let mut push_row = |text: &str, length: usize| {
            data.extend(text.chars());
            data.extend(std::iter::repeat_n(' ', max_len - length));
        };
        push_row(&one, len_one);
        push_row(&two, len_two);
        let char_array = CharArray::new(data, 2, max_len).expect("char array");
        addpath_builtin(vec![Value::CharArray(char_array)]).expect("addpath");

        let segments = current_path_segments();
        assert_eq!(segments[0], canonical(dir1.path()));
        assert_eq!(segments[1], canonical(dir2.path()));
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn addpath_errors_on_missing_folder() {
        let _lock = REPL_FS_TEST_LOCK
            .lock()
            .unwrap_or_else(|poison| poison.into_inner());
        let _guard = PathGuard::new();

        let missing = Value::String("this/folder/does/not/exist".into());
        let err = addpath_builtin(vec![missing]).expect_err("expected error");
        assert!(
            err.message().contains("folder") && err.message().contains("not found"),
            "unexpected error message: {}",
            err.message()
        );
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn addpath_genpath_string_is_expanded() {
        let _lock = REPL_FS_TEST_LOCK
            .lock()
            .unwrap_or_else(|poison| poison.into_inner());
        let _guard = PathGuard::new();

        let base = tempdir().expect("base");
        let sub = base.path().join("sub");
        fs::create_dir(&sub).expect("create sub");

        set_path_string("");
        let combined = format!(
            "{}{sep}{}",
            base.path().to_string_lossy(),
            sub.to_string_lossy(),
            sep = PATH_LIST_SEPARATOR
        );
        addpath_builtin(vec![Value::String(combined)]).expect("addpath");

        let segments = current_path_segments();
        assert_eq!(segments.len(), 2);
        assert_eq!(segments[0], canonical(base.path()));
        assert_eq!(segments[1], canonical(&sub));
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn addpath_returns_previous_path() {
        let _lock = REPL_FS_TEST_LOCK
            .lock()
            .unwrap_or_else(|poison| poison.into_inner());
        let guard = PathGuard::new();

        let dir = tempdir().expect("dir");
        let returned = addpath_builtin(vec![Value::String(
            dir.path().to_string_lossy().into_owned(),
        )])
        .expect("addpath");
        let returned_str = String::try_from(&returned).expect("string");
        assert_eq!(returned_str, guard.previous);
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn addpath_rejects_conflicting_position_flags() {
        let _lock = REPL_FS_TEST_LOCK
            .lock()
            .unwrap_or_else(|poison| poison.into_inner());
        let _guard = PathGuard::new();

        let dir = tempdir().expect("dir");
        let args = vec![
            Value::String(dir.path().to_string_lossy().into_owned()),
            Value::String("-begin".into()),
            Value::String("-end".into()),
        ];
        let err = addpath_builtin(args).expect_err("expected error");
        assert!(err.message().contains("position option"));
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn addpath_handles_dash_begin() {
        let _lock = REPL_FS_TEST_LOCK
            .lock()
            .unwrap_or_else(|poison| poison.into_inner());
        let _guard = PathGuard::new();

        let dir1 = tempdir().expect("dir1");
        let dir2 = tempdir().expect("dir2");
        set_path_string(&canonical(dir2.path()));

        let args = vec![
            Value::String(dir1.path().to_string_lossy().into_owned()),
            Value::String("-begin".into()),
        ];
        addpath_builtin(args).expect("addpath");

        let segments = current_path_segments();
        assert_eq!(segments[0], canonical(dir1.path()));
        assert_eq!(segments[1], canonical(dir2.path()));
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn addpath_accepts_string_containers() {
        let _lock = REPL_FS_TEST_LOCK
            .lock()
            .unwrap_or_else(|poison| poison.into_inner());
        let _guard = PathGuard::new();

        set_path_string("");

        let cwd = vfs::current_dir().expect("cwd");
        let string_array = StringArray::new(vec![cwd.to_string_lossy().into_owned()], vec![1, 1])
            .expect("string array");
        addpath_builtin(vec![Value::StringArray(string_array)]).expect("addpath");
        let current = current_path_string();
        assert_eq!(current, canonical(&cwd));
    }
}
