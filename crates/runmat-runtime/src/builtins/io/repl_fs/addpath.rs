//! MATLAB-compatible `addpath` builtin for manipulating the RunMat search path.

#[cfg(test)]
use runmat_builtins::CellArray;
use runmat_builtins::{CharArray, StringArray, Tensor, Value};
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
use std::env;
use std::path::{Component, Path, PathBuf};

const ERROR_ARG_TYPE: &str =
    "addpath: folder names must be character vectors, string scalars, string arrays, or cell arrays of character vectors";
const ERROR_TOO_FEW_ARGS: &str = "addpath: at least one folder must be specified";
const ERROR_POSITION_REPEATED: &str =
    "addpath: position option must be '-begin' or '-end' and may only appear once";

#[cfg_attr(
    feature = "doc_export",
    runmat_macros::register_doc_text(
        name = "addpath",
        builtin_path = "crate::builtins::io::repl_fs::addpath"
    )
)]
#[cfg_attr(not(feature = "doc_export"), allow(dead_code))]
pub const DOC_MD: &str = r#"---
title: "addpath"
category: "io/repl_fs"
keywords: ["addpath", "search path", "matlab path", "-begin", "-end", "-frozen"]
summary: "Add folders to the MATLAB search path used by RunMat."
references:
  - https://www.mathworks.com/help/matlab/ref/addpath.html
gpu_support:
  elementwise: false
  reduction: false
  precisions: []
  broadcasting: "none"
  notes: "Runs entirely on the CPU. gpuArray text inputs are gathered automatically before processing."
fusion:
  elementwise: false
  reduction: false
  max_inputs: 8
  constants: "inline"
requires_feature: null
tested:
  unit: "builtins::io::repl_fs::addpath::tests"
  integration: "builtins::io::repl_fs::addpath::tests::addpath_accepts_string_containers"
---

# What does the `addpath` function do in MATLAB / RunMat?
`addpath` prepends or appends folders to the MATLAB search path that RunMat consults when resolving functions, scripts, classes, and data files. The change affects the current session immediately, and the resulting ordering matches MATLAB semantics.

## How does the `addpath` function behave in MATLAB / RunMat?
- Folder arguments may be character vectors, string scalars, string arrays, or cell arrays of character vectors or strings. Multi-row char arrays contribute one folder per row (trailing padding is stripped).
- Multiple folders can be passed in a single argument using the platform path separator (`:` on Linux/macOS, `;` on Windows); this is compatible with `genpath`.
- By default, folders are added to the top of the search path. Use `'-end'` to append or `'-begin'` to force prepending explicitly. Only one position flag is permitted.
- The `'-frozen'` flag is accepted for MATLAB compatibility. RunMat does not currently track frozen entries separately; the flag simply suppresses incompatibility warnings.
- Duplicate entries are removed automatically so each folder appears at most once. On Windows, comparisons are case-insensitive.
- Inputs are resolved to absolute, canonicalised paths. Relative inputs are interpreted relative to the current working directory, and `~` expands to the user’s home directory.
- Folders must exist. RunMat raises `addpath: folder '<name>' not found` when a directory is missing or `addpath: '<name>' is not a folder` when the target is not a directory.

## `addpath` Function GPU Execution Behaviour
`addpath` manipulates host-side configuration. If any input value resides on the GPU, RunMat gathers it back to the host before parsing. No acceleration provider hooks or kernels are involved.

## GPU residency in RunMat (Do I need `gpuArray`?)
No. `addpath` operates entirely on CPU-side strings. Supplying `gpuArray` text inputs offers no benefit—RunMat gathers them automatically.

## Examples of using the `addpath` function in MATLAB / RunMat

### Add a single folder to the top of the search path
```matlab
old = addpath("util/toolbox");
path()
```
Expected behaviour:
```matlab
old          % Previous search path as a character vector
% The folder util/toolbox now appears before the old entries.
```

### Append folders to the end of the search path
```matlab
dirs = ["shared/filters", "shared/signal"];
addpath(dirs, "-end");
```
Expected behaviour:
```matlab
% The specified folders are appended to the end of the search path.
```

### Use `genpath` output to add a folder tree
```matlab
toolchain = genpath("third_party/toolchain");
addpath(toolchain);
```
Expected behaviour:
```matlab
% Every directory produced by genpath is inserted at the top of the search path.
```

### Add folders from a cell array
```matlab
folders = {'src/algorithms', 'src/visualization'};
addpath(folders{:}, "-begin");
```
Expected output:
```matlab
% Both folders appear at the beginning of the MATLAB search path.
```

### Move an existing folder to the top of the search path
```matlab
addpath("src/algorithms");
```
Expected behaviour:
```matlab
% The folder is removed from its previous position and reinserted at the front.
```

### Accept the MATLAB `-frozen` flag
```matlab
addpath("vendor/hardware", "-frozen");
```
Expected behaviour:
```matlab
% The folder is added; RunMat currently treats '-frozen' as an informational hint.
```

### Retrieve the previous path and restore it later
```matlab
old = addpath("analysis/utilities");
% ... run experiments ...
path(old);
```
Expected behaviour:
```matlab
% The original search path is restored when you call path(old).
```

### Combine multiple options
```matlab
addpath("contrib", "docs/examples", "-end", "-frozen");
```
Expected behaviour:
```matlab
% Both folders are appended to the end of the search path.
```

## FAQ
- **Does `addpath` insist on absolute paths?** No. Relative inputs are resolved against the current working directory and stored as absolute paths.
- **What happens with duplicate folders?** Existing occurrences are removed before the new ordering is applied, so each folder appears only once.
- **How do I append instead of prepend?** Supply `'-end'` as the final argument. Use `'-begin'` to force prepending explicitly.
- **Is `-frozen` supported?** The flag is accepted for MATLAB compatibility. RunMat currently treats it as a no-op but plans to integrate tighter tooling once savepath support lands.
- **Can I load `pathdef.m` directly?** Not yet. RunMat will add parity support in a future release. For now, evaluate the file manually and pass the resulting character vector to `path`.
- **Do folders need to exist?** Yes. RunMat validates that every entry exists and is a directory before updating the search path.
- **Will `addpath` accept GPU strings?** Yes. Inputs are gathered automatically, then processed on the CPU.
- **Does `addpath` return the new path?** Like MATLAB, `addpath` returns the previous path so it can be restored later with `path(old)`.

## See Also
[path](./path), [rmpath](https://www.mathworks.com/help/matlab/ref/rmpath.html), [genpath](https://www.mathworks.com/help/matlab/ref/genpath.html), [which](./which), [exist](./exist)

## Source & Feedback
- Source: [`crates/runmat-runtime/src/builtins/io/repl_fs/addpath.rs`](https://github.com/runmat-org/runmat/blob/main/crates/runmat-runtime/src/builtins/io/repl_fs/addpath.rs)
- Found an issue? [Open a GitHub ticket](https://github.com/runmat-org/runmat/issues/new/choose) with a repro.
"#;

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

fn addpath_error(message: impl Into<String>) -> RuntimeError {
    build_runtime_error(message)
        .with_builtin(BUILTIN_NAME)
        .build()
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
    summary = "Add folders to the MATLAB search path used by RunMat.",
    keywords = "addpath,search path,matlab path,-begin,-end,-frozen",
    accel = "cpu",
    suppress_auto_output = true,
    builtin_path = "crate::builtins::io::repl_fs::addpath"
)]
async fn addpath_builtin(args: Vec<Value>) -> crate::BuiltinResult<Value> {
    if args.is_empty() {
        return Err(addpath_error(ERROR_TOO_FEW_ARGS));
    }

    let gathered = gather_arguments(&args).await?;
    let previous = current_path_string();
    let spec = parse_arguments(&gathered).await?;
    apply_addpath(spec)?;
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
        return Err(addpath_error(ERROR_TOO_FEW_ARGS));
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
                    return Err(addpath_error(ERROR_POSITION_REPEATED));
                }
                position = InsertPosition::Begin;
                position_set = true;
            }
            Some(AddPathOption::End) => {
                if position_set {
                    return Err(addpath_error(ERROR_POSITION_REPEATED));
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
        return Err(addpath_error(ERROR_TOO_FEW_ARGS));
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

fn apply_addpath(spec: AddPathSpec) -> BuiltinResult<()> {
    let mut existing = current_path_segments();
    let mut seen = HashSet::new();
    let mut additions = Vec::new();

    for raw in spec.directories {
        let normalized = normalize_directory(&raw)?;
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
                let inner = (*ptr).clone();
                let gathered = gather_if_needed_async(&inner)
                    .await
                    .map_err(map_control_flow)?;
                collect_strings(&gathered, output).await?;
            }
            Ok(())
        }
        Value::GpuTensor(_) => Err(addpath_error(ERROR_ARG_TYPE)),
        _ => Err(addpath_error(ERROR_ARG_TYPE)),
    }
}

fn split_path_list(text: &str) -> Vec<String> {
    text.split(PATH_LIST_SEPARATOR)
        .map(|segment| segment.trim())
        .filter(|segment| !segment.is_empty())
        .map(|segment| segment.to_string())
        .collect()
}

fn normalize_directory(raw: &str) -> BuiltinResult<String> {
    let trimmed = raw.trim();
    if trimmed.is_empty() {
        return Err(addpath_error(ERROR_ARG_TYPE));
    }

    if trimmed.eq_ignore_ascii_case("pathdef") || trimmed.eq_ignore_ascii_case("pathdef.m") {
        return Err(addpath_error(
            "addpath: loading pathdef.m is not implemented yet",
        ));
    }

    let expanded = expand_user_path(trimmed, "addpath").map_err(addpath_error)?;
    let path = Path::new(&expanded);
    let joined = if path.is_absolute() {
        path.to_path_buf()
    } else {
        env::current_dir()
            .map_err(|_| addpath_error("addpath: unable to resolve current directory"))?
            .join(path)
    };
    let normalized = normalize_pathbuf(&joined);

    let metadata = vfs::metadata(&normalized)
        .map_err(|_| addpath_error(format!("addpath: folder '{trimmed}' not found")))?;
    if !metadata.is_dir() {
        return Err(addpath_error(format!(
            "addpath: '{trimmed}' is not a folder"
        )));
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
        return Err(addpath_error(ERROR_ARG_TYPE));
    }
    if tensor.rows() > 1 {
        return Err(addpath_error(ERROR_ARG_TYPE));
    }
    let mut text = String::with_capacity(tensor.data.len());
    for &code in &tensor.data {
        if !code.is_finite() {
            return Err(addpath_error(ERROR_ARG_TYPE));
        }
        let rounded = code.round();
        if (code - rounded).abs() > 1e-6 {
            return Err(addpath_error(ERROR_ARG_TYPE));
        }
        let int_code = rounded as i64;
        if !(0..=0x10FFFF).contains(&int_code) {
            return Err(addpath_error(ERROR_ARG_TYPE));
        }
        let ch = char::from_u32(int_code as u32).ok_or_else(|| addpath_error(ERROR_ARG_TYPE))?;
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
    use crate::builtins::common::test_support;
    use std::convert::TryFrom;
    use std::env;
    use std::fs;
    use tempfile::tempdir;

    fn addpath_builtin(args: Vec<Value>) -> BuiltinResult<Value> {
        futures::executor::block_on(super::addpath_builtin(args))
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

        let cwd = env::current_dir().expect("cwd");
        let string_array = StringArray::new(vec![cwd.to_string_lossy().into_owned()], vec![1, 1])
            .expect("string array");
        addpath_builtin(vec![Value::StringArray(string_array)]).expect("addpath");
        let current = current_path_string();
        assert_eq!(current, canonical(&cwd));
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn addpath_doc_examples_present() {
        let blocks = test_support::doc_examples(DOC_MD);
        assert!(!blocks.is_empty());
    }
}
