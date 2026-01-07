//! MATLAB-compatible `genpath` builtin for generating recursive search paths.

use runmat_builtins::{CharArray, StringArray, Tensor, Value};
use runmat_macros::runtime_builtin;

use crate::builtins::common::fs::{compare_names, expand_user_path, path_to_string};
use crate::builtins::common::spec::{
    BroadcastSemantics, BuiltinFusionSpec, BuiltinGpuSpec, ConstantStrategy, GpuOpKind,
    ReductionNaN, ResidencyPolicy, ShapeRequirements,
};
use crate::gather_if_needed;

use runmat_filesystem as vfs;
use std::collections::HashSet;
use std::env;
use std::path::{Path, PathBuf};

const ERROR_FOLDER_TYPE: &str = "genpath: folder must be a character vector or string scalar";
const ERROR_EXCLUDES_TYPE: &str = "genpath: excludes must be a character vector or string scalar";

#[cfg_attr(
    feature = "doc_export",
    runmat_macros::register_doc_text(
        name = "genpath",
        builtin_path = "crate::builtins::io::repl_fs::genpath"
    )
)]
#[cfg_attr(not(feature = "doc_export"), allow(dead_code))]
pub const DOC_MD: &str = r#"---
title: "genpath"
category: "io/repl_fs"
keywords: ["genpath", "recursive path", "search path", "addpath"]
summary: "Generate a MATLAB-style search path string for a folder tree."
references:
  - https://www.mathworks.com/help/matlab/ref/genpath.html
gpu_support:
  elementwise: false
  reduction: false
  precisions: []
  broadcasting: "none"
  notes: "Runs entirely on the CPU. gpuArray text inputs are gathered automatically before processing."
fusion:
  elementwise: false
  reduction: false
  max_inputs: 2
  constants: "inline"
requires_feature: null
tested:
  unit: "builtins::io::repl_fs::genpath::tests"
  integration: "builtins::io::repl_fs::genpath::tests::genpath_excludes_specified_directories"
---

# What does the `genpath` function do in MATLAB / RunMat?
`genpath` walks a folder tree and returns a character vector that lists the folder
and every subfolder in depth-first order. Each entry is separated by the platform
`pathsep` character (`:` on Linux/macOS, `;` on Windows), making the result a
drop-in argument for `addpath`, `rmpath`, and other path-editing utilities.

## How does the `genpath` function behave in MATLAB / RunMat?
- `genpath()` with no inputs uses the current working directory (`pwd`) as the
  root folder. `genpath(folder)` accepts either an absolute or relative path.
- MATLAB-reserved directories named `private` or `resources`, as well as class
  folders (`@MyClass`) and namespace folders (`+pkg`), are skipped automatically
  along with all of their descendants. The root folder is still included even if
  it matches one of these reserved names.
- All returned entries are absolute, canonicalised paths beginning with the root
  directory, followed by each descendant directory in lexical order. The walking
  order matches MATLAB's depth-first behaviour.
- Duplicate directories are removed automatically. Symlinked folders that resolve
  to the same physical location only appear once in the output.
- `genpath(folder, excludes)` omits any folder listed in `excludes` together with
  its descendants. The `excludes` argument is a character vector of absolute or
  relative paths separated by `pathsep`. Relative entries are resolved against the
  supplied root folder before canonicalisation.
- If a folder in the tree cannot be read (for example due to permissions), RunMat
  silently skips that branch while keeping the rest of the result intact.
- If the requested root folder does not exist, RunMat raises
  `genpath: folder '<name>' not found`, matching MATLAB error semantics.

## `genpath` Function GPU Execution Behaviour
`genpath` operates entirely on the host filesystem. Any string inputs that reside
on the GPU are gathered back to the CPU before processing. No acceleration
provider hooks are invoked and no device kernels are launched.

## GPU residency in RunMat (Do I need `gpuArray`?)
No. `genpath` manipulates host-side strings and filesystem metadata. Supplying
`gpuArray` values provides no benefit—RunMat gathers the data automatically.

## Examples of using the `genpath` function in MATLAB / RunMat

### Generate a recursive path for the current working folder
```matlab
p = genpath();
addpath(p);   % Add the current folder and all subfolders to the MATLAB path
```

### Build a search path for a project toolbox
```matlab
toolboxRoot = "toolbox/signal";
toolboxPath = genpath(toolboxRoot);
addpath(toolboxPath, "-end");
```

### Exclude build output directories from the generated path
```matlab
root = "projects/solver";
excludes = strjoin([
    fullfile(root, "build"),
    fullfile(root, "dist")
], pathsep);
p = genpath(root, excludes);
```

### Skip a private utilities folder while including the rest of the tree
```matlab
root = "analysis";
p = genpath(root, fullfile(root, "private"));
```

### Automatically skip MATLAB-reserved folders
```matlab
root = "toolbox";
mkdir(root, "private");
mkdir(root, "+package");
p = genpath(root);
```

Expected output:
```matlab
% `p` only includes `toolbox` and subfolders that are not named `private` or
% `resources`, and that do not begin with `@` or `+`.
```

### Combine `genpath` with `savepath` for persistent tooling setup
```matlab
p = genpath("third_party/toolchain");
addpath(p);
savepath();
```

## FAQ
- **Does `genpath` include package (`+pkg`) and class (`@Class`) folders?** No.
  These directories are excluded automatically, along with folders named
  `private` or `resources`. Use the `excludes` argument for additional rules.
- **How do I exclude multiple folders?** Build a character vector or string using
  `pathsep` (for example, `strjoin({path1, path2}, pathsep)`) and pass it as the
  second argument. Relative entries are interpreted relative to the root folder.
- **Why are some folders missing from the result?** RunMat only omits folders that
  you explicitly exclude or that cannot be read due to filesystem permissions.
- **What happens if the root folder does not exist?** `genpath` raises
  `genpath: folder '<name>' not found`, matching MATLAB. Create the folder first
  or adjust the argument.
- **Can I call `genpath` on a GPU array?** Yes—the input is gathered to the CPU
  before processing, and the result is returned as a standard character vector.
- **Is the result always absolute paths?** Yes. RunMat resolves the root and all
  discovered folders to their absolute, canonical locations to avoid duplicates.
- **Will symbolic links be traversed?** Yes, but the target directory only appears
  once in the output. Directory cycles are prevented by tracking canonical paths.
- **Can I feed `genpath` output directly into `addpath` or `rmpath`?** Absolutely.
  The output string is `pathsep`-delimited specifically for that purpose.

## See Also
[addpath](./addpath), [rmpath](./rmpath), [path](./path), [which](../../introspection/which)

## Source & Feedback
- Source: [`crates/runmat-runtime/src/builtins/io/repl_fs/genpath.rs`](https://github.com/runmat-org/runmat/blob/main/crates/runmat-runtime/src/builtins/io/repl_fs/genpath.rs)
- Found an issue? [Open a GitHub ticket](https://github.com/runmat-org/runmat/issues/new/choose) with steps to reproduce.
"#;

#[runmat_macros::register_gpu_spec(builtin_path = "crate::builtins::io::repl_fs::genpath")]
pub const GPU_SPEC: BuiltinGpuSpec = BuiltinGpuSpec {
    name: "genpath",
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
    notes: "Filesystem traversal is a host-only operation; inputs are gathered before processing.",
};

#[runmat_macros::register_fusion_spec(builtin_path = "crate::builtins::io::repl_fs::genpath")]
pub const FUSION_SPEC: BuiltinFusionSpec = BuiltinFusionSpec {
    name: "genpath",
    shape: ShapeRequirements::Any,
    constant_strategy: ConstantStrategy::InlineLiteral,
    elementwise: None,
    reduction: None,
    emits_nan: false,
    notes:
        "I/O-oriented builtins are not eligible for fusion; metadata registered for completeness.",
};

#[runtime_builtin(
    name = "genpath",
    category = "io/repl_fs",
    summary = "Generate a MATLAB-style search path string for a folder tree.",
    keywords = "genpath,recursive path,search path,addpath",
    accel = "cpu",
    suppress_auto_output = true,
    builtin_path = "crate::builtins::io::repl_fs::genpath"
)]
fn genpath_builtin(args: Vec<Value>) -> Result<Value, String> {
    let gathered = gather_arguments(args)?;
    match gathered.len() {
        0 => generate_from_current_directory(),
        1 => generate_from_root(&gathered[0], None),
        2 => generate_from_root(&gathered[0], Some(&gathered[1])),
        _ => Err("genpath: too many input arguments".to_string()),
    }
}

fn generate_from_current_directory() -> Result<Value, String> {
    let cwd = env::current_dir()
        .map_err(|err| format!("genpath: unable to resolve current directory: {err}"))?;
    let (canonical_path, canonical_str) = canonicalize_existing(&cwd, "current directory")?;
    let excludes = ExcludeSet::default();
    let mut seen = HashSet::new();
    let mut segments = Vec::new();
    traverse(
        &canonical_path,
        canonical_str,
        &excludes,
        &mut seen,
        &mut segments,
    )?;
    Ok(char_array_value(&join_segments(&segments)))
}

fn generate_from_root(root: &Value, excludes: Option<&Value>) -> Result<Value, String> {
    let root_text = extract_text(root, ERROR_FOLDER_TYPE)?;
    let root_info = normalize_root(&root_text)?;
    let exclude_text = excludes
        .map(|value| extract_text(value, ERROR_EXCLUDES_TYPE))
        .transpose()?;
    let exclude_set = build_exclude_set(exclude_text.as_deref(), &root_info)?;
    let mut seen = HashSet::new();
    let mut segments = Vec::new();
    traverse(
        &root_info.path,
        root_info.canonical.clone(),
        &exclude_set,
        &mut seen,
        &mut segments,
    )?;
    Ok(char_array_value(&join_segments(&segments)))
}

fn gather_arguments(args: Vec<Value>) -> Result<Vec<Value>, String> {
    let mut gathered = Vec::with_capacity(args.len());
    for value in args {
        let host_value = gather_if_needed(&value).map_err(|err| format!("genpath: {err}"))?;
        gathered.push(host_value);
    }
    Ok(gathered)
}

struct RootInfo {
    path: PathBuf,
    canonical: String,
}

fn normalize_root(text: &str) -> Result<RootInfo, String> {
    if text.trim().is_empty() {
        return Err(format!("genpath: folder '{text}' not found"));
    }

    let expanded = expand_user_path(text, "genpath")?;
    let raw_path = PathBuf::from(&expanded);
    let absolute = if raw_path.is_absolute() {
        raw_path
    } else {
        let cwd = env::current_dir()
            .map_err(|err| format!("genpath: unable to resolve current directory: {err}"))?;
        cwd.join(raw_path)
    };

    let (canonical_path, canonical_str) = canonicalize_existing(&absolute, text)?;

    Ok(RootInfo {
        path: canonical_path,
        canonical: canonical_str,
    })
}

fn canonicalize_existing(path: &Path, display: &str) -> Result<(PathBuf, String), String> {
    let canonical =
        vfs::canonicalize(path).map_err(|_| format!("genpath: folder '{display}' not found"))?;
    let canonical_str = canonical_string_from_path(&canonical);
    Ok((canonical, canonical_str))
}

#[cfg(windows)]
fn canonical_string_from_path(path: &Path) -> String {
    let mut text = path_to_string(path);
    if let Some(stripped) = text.strip_prefix(r"\\?\") {
        text = stripped.to_string();
    }
    text
}

#[cfg(not(windows))]
fn canonical_string_from_path(path: &Path) -> String {
    path_to_string(path)
}

fn join_segments(segments: &[String]) -> String {
    if segments.is_empty() {
        return String::new();
    }
    let mut output = String::new();
    for (index, segment) in segments.iter().enumerate() {
        if index > 0 {
            output.push(crate::builtins::common::path_state::PATH_LIST_SEPARATOR);
        }
        output.push_str(segment);
    }
    output
}

fn traverse(
    path: &Path,
    canonical: String,
    excludes: &ExcludeSet,
    seen: &mut HashSet<String>,
    segments: &mut Vec<String>,
) -> Result<(), String> {
    let normalized = normalize_case(&canonical);
    if !seen.insert(normalized) {
        return Ok(());
    }

    if excludes.contains(&canonical) {
        return Ok(());
    }

    segments.push(canonical.clone());

    let mut children = Vec::new();
    let entries = match vfs::read_dir(path) {
        Ok(listing) => listing,
        Err(_) => return Ok(()),
    };
    for entry in entries {
        let source_path = entry.path().to_path_buf();
        let metadata = match vfs::metadata(&source_path) {
            Ok(meta) => meta,
            Err(_) => continue,
        };
        if !metadata.is_dir() {
            continue;
        }
        let name = entry.file_name().to_string_lossy().into_owned();
        if is_matlab_reserved_folder(&name) {
            continue;
        }
        let child_path = match vfs::canonicalize(&source_path) {
            Ok(path) => path,
            Err(_) => continue,
        };

        let child_str = canonical_string_from_path(&child_path);
        children.push(ChildEntry {
            path: child_path,
            canonical: child_str,
            name,
        });
    }

    children.sort_by(|a, b| compare_names(&a.name, &b.name));

    for child in children {
        traverse(
            &child.path,
            child.canonical.clone(),
            excludes,
            seen,
            segments,
        )?;
    }

    Ok(())
}

struct ChildEntry {
    path: PathBuf,
    canonical: String,
    name: String,
}

fn is_matlab_reserved_folder(name: &str) -> bool {
    if name.starts_with('@') || name.starts_with('+') {
        return true;
    }

    #[cfg(windows)]
    {
        let lower = name.to_ascii_lowercase();
        matches!(lower.as_str(), "private" | "resources")
    }
    #[cfg(not(windows))]
    {
        matches!(name, "private" | "resources")
    }
}

#[derive(Default)]
struct ExcludeSet {
    entries: Vec<ExcludeEntry>,
}

impl ExcludeSet {
    fn from_entries(entries: Vec<String>) -> Self {
        let normalized_entries = entries
            .into_iter()
            .map(|canonical| {
                let normalized = normalize_case(&canonical);
                let mut prefix = normalized.clone();
                if !prefix.ends_with(std::path::MAIN_SEPARATOR) {
                    prefix.push(std::path::MAIN_SEPARATOR);
                }
                ExcludeEntry {
                    normalized,
                    normalized_with_sep: prefix,
                }
            })
            .collect();

        ExcludeSet {
            entries: normalized_entries,
        }
    }

    fn contains(&self, canonical: &str) -> bool {
        if self.entries.is_empty() {
            return false;
        }
        let key = normalize_case(canonical);
        self.entries
            .iter()
            .any(|entry| key == entry.normalized || key.starts_with(&entry.normalized_with_sep))
    }
}

struct ExcludeEntry {
    normalized: String,
    normalized_with_sep: String,
}

fn build_exclude_set(excludes: Option<&str>, root: &RootInfo) -> Result<ExcludeSet, String> {
    let mut entries = Vec::new();
    if let Some(text) = excludes {
        for raw in text.split(crate::builtins::common::path_state::PATH_LIST_SEPARATOR) {
            let trimmed = raw.trim();
            if trimmed.is_empty() {
                continue;
            }

            let expanded = match expand_user_path(trimmed, "genpath") {
                Ok(val) => val,
                Err(_) => continue,
            };

            let mut candidate = PathBuf::from(&expanded);
            if !candidate.is_absolute() {
                candidate = root.path.join(candidate);
            }

            if let Ok((_, canonical_str)) = canonicalize_existing(&candidate, trimmed) {
                entries.push(canonical_str);
                continue;
            }

            // Fallback: try relative to the current working directory
            if let Ok(cwd) = env::current_dir() {
                let alt = if Path::new(trimmed).is_absolute() {
                    PathBuf::from(trimmed)
                } else {
                    cwd.join(trimmed)
                };
                if let Ok((_, canonical_alt)) = canonicalize_existing(&alt, trimmed) {
                    entries.push(canonical_alt);
                }
            }
        }
    }

    Ok(ExcludeSet::from_entries(entries))
}

fn normalize_case(text: &str) -> String {
    #[cfg(windows)]
    {
        text.replace('/', "\\").to_ascii_lowercase()
    }
    #[cfg(not(windows))]
    {
        text.to_string()
    }
}

fn char_array_value(text: &str) -> Value {
    Value::CharArray(CharArray::new_row(text))
}

fn extract_text(value: &Value, type_error: &str) -> Result<String, String> {
    match value {
        Value::String(text) => Ok(text.clone()),
        Value::StringArray(StringArray { data, .. }) => {
            if data.len() != 1 {
                Err(type_error.to_string())
            } else {
                Ok(data[0].clone())
            }
        }
        Value::CharArray(chars) => {
            if chars.rows != 1 {
                return Err(type_error.to_string());
            }
            Ok(chars.data.iter().collect())
        }
        Value::Tensor(tensor) => tensor_to_string(tensor, type_error),
        _ => Err(type_error.to_string()),
    }
}

fn tensor_to_string(tensor: &Tensor, type_error: &str) -> Result<String, String> {
    if tensor.shape.len() > 2 {
        return Err(type_error.to_string());
    }

    if tensor.rows() != 1 {
        return Err(type_error.to_string());
    }

    let mut text = String::with_capacity(tensor.data.len());
    for &code in &tensor.data {
        if !code.is_finite() {
            return Err(type_error.to_string());
        }
        let rounded = code.round();
        if (code - rounded).abs() > 1e-6 {
            return Err(type_error.to_string());
        }
        let int_code = rounded as i64;
        if !(0..=0x10FFFF).contains(&int_code) {
            return Err(type_error.to_string());
        }
        let ch = char::from_u32(int_code as u32).ok_or_else(|| type_error.to_string())?;
        text.push(ch);
    }

    Ok(text)
}

#[cfg(test)]
pub(crate) mod tests {
    use super::super::REPL_FS_TEST_LOCK;
    use super::*;
    use crate::builtins::common::path_state::PATH_LIST_SEPARATOR;
    use crate::builtins::common::test_support;
    use runmat_builtins::{CharArray, StringArray, Tensor};
    use std::convert::TryFrom;
    use std::fs;
    use tempfile::tempdir;

    struct DirGuard {
        previous: PathBuf,
    }

    impl DirGuard {
        fn change(to: &Path) -> Result<Self, String> {
            let previous = env::current_dir()
                .map_err(|err| format!("genpath: unable to capture current directory: {err}"))?;
            env::set_current_dir(to)
                .map_err(|err| format!("genpath: unable to change directory: {err}"))?;
            Ok(Self { previous })
        }
    }

    impl Drop for DirGuard {
        fn drop(&mut self) {
            let _ = env::set_current_dir(&self.previous);
        }
    }

    fn canonical(path: &Path) -> String {
        let (_, canonical_str) =
            canonicalize_existing(path, &path_to_string(path)).expect("canonical path");
        canonical_str
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn genpath_returns_char_array() {
        let _lock = REPL_FS_TEST_LOCK
            .lock()
            .unwrap_or_else(|poison| poison.into_inner());

        let base = tempdir().expect("tempdir");
        let result = genpath_builtin(vec![Value::String(
            base.path().to_string_lossy().into_owned(),
        )])
        .expect("genpath");

        match result {
            Value::CharArray(CharArray { rows, .. }) => assert_eq!(rows, 1),
            other => panic!("expected CharArray, got {other:?}"),
        }
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn genpath_without_arguments_uses_current_directory() {
        let _lock = REPL_FS_TEST_LOCK
            .lock()
            .unwrap_or_else(|poison| poison.into_inner());

        let base = tempdir().expect("base");
        let alpha = base.path().join("alpha");
        let beta = base.path().join("beta");
        let gamma = alpha.join("gamma");
        fs::create_dir(&alpha).expect("alpha");
        fs::create_dir(&beta).expect("beta");
        fs::create_dir(&gamma).expect("gamma");

        let _guard = DirGuard::change(base.path()).expect("dir guard");

        let value = genpath_builtin(Vec::new()).expect("genpath");
        let text = String::try_from(&value).expect("string");
        let segments: Vec<&str> = if text.is_empty() {
            Vec::new()
        } else {
            text.split(PATH_LIST_SEPARATOR).collect()
        };

        let expected = [
            canonical(base.path()),
            canonical(&alpha),
            canonical(&gamma),
            canonical(&beta),
        ];

        assert_eq!(segments.len(), expected.len());
        for (seg, exp) in segments.iter().zip(expected.iter()) {
            assert_eq!(*seg, exp);
        }
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn genpath_accepts_char_array_root_argument() {
        let _lock = REPL_FS_TEST_LOCK
            .lock()
            .unwrap_or_else(|poison| poison.into_inner());

        let base = tempdir().expect("base");
        let path_text = base.path().to_string_lossy().into_owned();
        let char_arg = Value::CharArray(CharArray::new_row(&path_text));
        let value = genpath_builtin(vec![char_arg]).expect("genpath");
        let text = String::try_from(&value).expect("string");
        assert!(
            text.starts_with(&canonical(base.path())),
            "expected output to begin with canonical root path"
        );
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn genpath_accepts_string_array_root_argument() {
        let _lock = REPL_FS_TEST_LOCK
            .lock()
            .unwrap_or_else(|poison| poison.into_inner());

        let base = tempdir().expect("base");
        let array = StringArray::new(vec![base.path().to_string_lossy().into_owned()], vec![1])
            .expect("string array");
        let value = genpath_builtin(vec![Value::StringArray(array)]).expect("genpath");
        let text = String::try_from(&value).expect("string");
        assert!(
            text.starts_with(&canonical(base.path())),
            "expected output to begin with canonical root path"
        );
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn genpath_accepts_tensor_char_codes_root_argument() {
        let _lock = REPL_FS_TEST_LOCK
            .lock()
            .unwrap_or_else(|poison| poison.into_inner());

        let base = tempdir().expect("base");
        let path_text = base.path().to_string_lossy().into_owned();
        let codes: Vec<f64> = path_text.bytes().map(|b| b as f64).collect();
        let tensor =
            Tensor::new_2d(codes, 1, path_text.len()).expect("tensor from path characters");
        let value = genpath_builtin(vec![Value::Tensor(tensor)]).expect("genpath");
        let text = String::try_from(&value).expect("string");
        assert!(
            text.starts_with(&canonical(base.path())),
            "expected output to begin with canonical root path"
        );
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn genpath_excludes_relative_entries() {
        let _lock = REPL_FS_TEST_LOCK
            .lock()
            .unwrap_or_else(|poison| poison.into_inner());

        let base = tempdir().expect("base");
        let keep = base.path().join("keep");
        let skip = base.path().join("skip");
        fs::create_dir(&keep).expect("keep");
        fs::create_dir(&skip).expect("skip");

        let result = genpath_builtin(vec![
            Value::String(base.path().to_string_lossy().into_owned()),
            Value::String("skip".into()),
        ])
        .expect("genpath");

        let text = String::try_from(&result).expect("string");
        let segments: Vec<String> = if text.is_empty() {
            Vec::new()
        } else {
            text.split(PATH_LIST_SEPARATOR)
                .map(|segment| segment.to_string())
                .collect()
        };

        assert!(
            !segments.contains(&canonical(&skip)),
            "expected skip directory to be excluded"
        );
        assert!(
            segments.contains(&canonical(&keep)),
            "expected keep directory to be present"
        );
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn genpath_errors_on_invalid_argument_type() {
        let _lock = REPL_FS_TEST_LOCK
            .lock()
            .unwrap_or_else(|poison| poison.into_inner());

        let err = genpath_builtin(vec![Value::Num(1.0)]).expect_err("expected error");
        assert_eq!(err, ERROR_FOLDER_TYPE);
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn genpath_excludes_specified_directories() {
        let _lock = REPL_FS_TEST_LOCK
            .lock()
            .unwrap_or_else(|poison| poison.into_inner());

        let base = tempdir().expect("base");
        let alpha = base.path().join("alpha");
        let beta = base.path().join("beta");
        let skip = alpha.join("skip");
        fs::create_dir(&alpha).expect("alpha");
        fs::create_dir(&beta).expect("beta");
        fs::create_dir(&skip).expect("skip");

        let exclude_string = format!(
            "{}{}{}",
            canonical(&alpha),
            PATH_LIST_SEPARATOR,
            canonical(&skip)
        );

        let result = genpath_builtin(vec![
            Value::String(base.path().to_string_lossy().into_owned()),
            Value::String(exclude_string),
        ])
        .expect("genpath");

        let text = String::try_from(&result).expect("string");
        let segments: Vec<&str> = if text.is_empty() {
            Vec::new()
        } else {
            text.split(PATH_LIST_SEPARATOR).collect()
        };

        let expected = [canonical(base.path()), canonical(&beta)];

        assert_eq!(segments.len(), expected.len());
        for (seg, exp) in segments.iter().zip(expected.iter()) {
            assert_eq!(*seg, exp);
        }
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn genpath_skips_matlab_reserved_directories() {
        let _lock = REPL_FS_TEST_LOCK
            .lock()
            .unwrap_or_else(|poison| poison.into_inner());

        let base = tempdir().expect("base");
        let private_dir = base.path().join("private");
        let class_dir = base.path().join("@MyClass");
        let package_dir = base.path().join("+pkg");
        let resources_dir = base.path().join("resources");
        let keep_dir = base.path().join("keep");
        let keep_child = keep_dir.join("child");

        for dir in [
            &private_dir,
            &class_dir,
            &package_dir,
            &resources_dir,
            &keep_dir,
            &keep_child,
        ] {
            fs::create_dir_all(dir).expect("mkdir");
        }

        let result = genpath_builtin(vec![Value::String(
            base.path().to_string_lossy().into_owned(),
        )])
        .expect("genpath");

        let text = String::try_from(&result).expect("string");
        let segments: Vec<String> = if text.is_empty() {
            Vec::new()
        } else {
            text.split(PATH_LIST_SEPARATOR)
                .map(|segment| segment.to_string())
                .collect()
        };

        let expected = vec![
            canonical(base.path()),
            canonical(&keep_dir),
            canonical(&keep_child),
        ];

        assert_eq!(segments, expected);

        for skipped in [
            canonical(&private_dir),
            canonical(&class_dir),
            canonical(&package_dir),
            canonical(&resources_dir),
        ] {
            assert!(
                !segments.contains(&skipped),
                "expected {skipped} to be absent from the generated path"
            );
        }
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    #[cfg(unix)]
    fn genpath_deduplicates_symlink_targets() {
        use std::os::unix::fs::symlink;

        let _lock = REPL_FS_TEST_LOCK
            .lock()
            .unwrap_or_else(|poison| poison.into_inner());

        let base = tempdir().expect("base");
        let alpha = base.path().join("alpha");
        let alias = base.path().join("alias_alpha");
        fs::create_dir(&alpha).expect("alpha");
        symlink(&alpha, &alias).expect("symlink");

        let value = genpath_builtin(vec![Value::String(
            base.path().to_string_lossy().into_owned(),
        )])
        .expect("genpath");

        let text = String::try_from(&value).expect("string");
        let segments: Vec<String> = if text.is_empty() {
            Vec::new()
        } else {
            text.split(PATH_LIST_SEPARATOR)
                .map(|segment| segment.to_string())
                .collect()
        };

        let root = canonical(base.path());
        let alpha_canonical = canonical(&alpha);

        assert!(
            segments.contains(&root),
            "expected root directory to be present"
        );
        let count = segments
            .iter()
            .filter(|segment| **segment == alpha_canonical)
            .count();
        assert_eq!(count, 1, "expected canonical alpha path to appear once");
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn genpath_errors_on_missing_root() {
        let _lock = REPL_FS_TEST_LOCK
            .lock()
            .unwrap_or_else(|poison| poison.into_inner());

        let missing = Value::String("this/does/not/exist".into());
        let err = genpath_builtin(vec![missing]).expect_err("expected error");
        assert!(err.contains("not found"));
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn doc_examples_present() {
        let blocks = test_support::doc_examples(DOC_MD);
        assert!(!blocks.is_empty());
    }
}
