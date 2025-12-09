//! MATLAB-compatible `cd` builtin for RunMat.

use std::env;
use std::path::{Path, PathBuf};

use runmat_builtins::{CharArray, Value};
use runmat_macros::runtime_builtin;

use crate::builtins::common::spec::{
    BroadcastSemantics, BuiltinFusionSpec, BuiltinGpuSpec, ConstantStrategy, GpuOpKind,
    ReductionNaN, ResidencyPolicy, ShapeRequirements,
};
use crate::gather_if_needed;

#[cfg(feature = "doc_export")]
#[runmat_macros::register_doc_text(name = "cd")]
pub const DOC_MD: &str = r#"---
title: "cd"
category: "io/repl_fs"
keywords: ["cd", "change directory", "current folder", "working directory", "pwd"]
summary: "Change the current working folder or query the folder that RunMat is executing in."
references:
  - https://www.mathworks.com/help/matlab/ref/cd.html
gpu_support:
  elementwise: false
  reduction: false
  precisions: []
  broadcasting: "none"
  notes: "Runs entirely on the host CPU. When the folder argument resides on the GPU, RunMat gathers it to the host before resolving the new path."
fusion:
  elementwise: false
  reduction: false
  max_inputs: 1
  constants: "inline"
requires_feature: null
tested:
  unit: "builtins::io::repl_fs::cd::tests"
  integration: "builtins::io::repl_fs::cd::tests::cd_changes_directory_and_returns_previous"
---

# What does the `cd` function do in MATLAB / RunMat?
`cd` displays the current working folder or switches RunMat to a different folder. It mirrors MATLAB so scripts and interactive sessions can rely on the same workspace layout when loading files, saving artifacts, or invoking other builtins that reference relative paths.

## How does the `cd` function behave in MATLAB / RunMat?
- `cd` with no input returns the absolute path of the current folder as a character row vector (`1×N`).
- `cd(newFolder)` changes the working folder and returns the previous folder, enabling the classic MATLAB pattern `old = cd(new); ...; cd(old);`.
- Accepts character vectors and string scalars. String arrays must contain exactly one element. Other types raise `cd: folder name must be a character vector or string scalar`.
- Supports relative paths (for example `cd('..')`), absolute paths, and the shell-style `~` expansion to jump to the user home folder.
- Emits descriptive errors when the folder does not exist or cannot be accessed.
- Does not modify GPU residency; path values always live on the host.

## `cd` Function GPU Execution Behavior
`cd` performs host-side path manipulation and process-level directory changes. When callers supply the folder argument from GPU memory, RunMat gathers the value before resolving the new path. Providers are not expected to implement hooks for this builtin, and no GPU kernels run as part of `cd`.

## GPU residency in RunMat (Do I need `gpuArray`?)
`cd` operates entirely on the CPU, so there is no performance benefit to moving folder arguments onto the GPU. If a path value is already wrapped in `gpuArray`, RunMat transparently gathers it before resolving the change directory request. Scripts can keep using plain character vectors or string scalars without worrying about residency.

## Examples of using the `cd` function in MATLAB / RunMat

### Display The Current Working Folder In RunMat
```matlab
current = cd;
```
Expected output:
```matlab
% current is the absolute path to the folder where RunMat is executing
```

### Change Directory To A Project Subfolder
```matlab
projectRoot = pwd;
mkdir("data/logs");
old = cd("data/logs");
% ... work inside the logs folder ...
cd(old);
```
Expected output:
```matlab
% old contains the previous folder so you can restore it later, and the final cd(old)
% call brings you back to projectRoot
```

### Navigate Up One Level From The Current Folder
```matlab
old = cd("..");
```
Expected output:
```matlab
% RunMat moves to the parent folder and old captures the prior location
```

### Switch To Your Home Folder With cd ~
```matlab
old = cd("~");
```
Expected output:
```matlab
% Changes to the user home directory; old stores the previous folder
```

### Capture The Previous Folder And Restore Later
```matlab
old = cd("results");
% ... run code that writes outputs into results ...
cd(old);
```
Expected output:
```matlab
% The working folder is restored to its original location at the end
```

### Handle Missing Folders With A try/catch Block
```matlab
try
    cd("missing-folder");
catch err
    disp(err.message);
end
```
Expected output:
```matlab
% Displays a descriptive error such as:
% "cd: unable to change directory to 'missing-folder' (No such file or directory)"
```

## FAQ
- **Does `cd` return the new folder or the old folder?** When you pass an input, `cd` returns the previous folder so that you can restore it later. Calling `cd` with no input returns the current folder.
- **Can I pass GPU-resident strings to `cd`?** Yes. RunMat gathers any GPU scalar arguments automatically before resolving the path.
- **Is tilde (`~`) expansion supported?** Yes. `cd('~')` and `cd('~/subdir')` jump to the user home folder. Other leading `~` patterns are treated literally so existing MATLAB scripts continue to work on platforms that support them.
- **How are relative paths resolved?** Relative folders are interpreted with respect to whatever `cd` reports as the current folder, exactly like MATLAB.
- **What happens if the target folder does not exist?** `cd` throws an error that includes both the requested folder and the underlying operating system message.
- **Can I change to folders whose names include spaces?** Yes. Provide them as strings or character vectors—RunMat preserves whitespace exactly.

## See Also
[pwd](../../../../builtins/io/repl_fs/pwd), [ls](../../../../builtins/io/repl_fs/ls), [dir](../../../../builtins/io/repl_fs/dir), [fileread](../../filetext/fileread)

## Source & Feedback
- Source: [`crates/runmat-runtime/src/builtins/io/repl_fs/cd.rs`](https://github.com/runmat-org/runmat/blob/main/crates/runmat-runtime/src/builtins/io/repl_fs/cd.rs)
- Found an issue? [Open a GitHub ticket](https://github.com/runmat-org/runmat/issues/new/choose) with a minimal reproduction.
"#;

#[runmat_macros::register_gpu_spec]
pub const GPU_SPEC: BuiltinGpuSpec = BuiltinGpuSpec {
    name: "cd",
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
        "Host-only operation that updates the process working folder; GPU inputs are gathered before path resolution.",
};

#[runmat_macros::register_fusion_spec]
pub const FUSION_SPEC: BuiltinFusionSpec = BuiltinFusionSpec {
    name: "cd",
    shape: ShapeRequirements::Any,
    constant_strategy: ConstantStrategy::InlineLiteral,
    elementwise: None,
    reduction: None,
    emits_nan: false,
    notes: "I/O builtins are not eligible for fusion; metadata is registered for completeness.",
};

#[runtime_builtin(
    name = "cd",
    category = "io/repl_fs",
    summary = "Change the current working folder or query the folder that RunMat is executing in.",
    keywords = "cd,change directory,current folder,working directory,pwd",
    accel = "cpu"
)]
fn cd_builtin(args: Vec<Value>) -> Result<Value, String> {
    let gathered = gather_arguments(&args)?;
    match gathered.len() {
        0 => current_directory_value(),
        1 => change_directory(&gathered[0]),
        _ => Err("cd: too many input arguments".to_string()),
    }
}

fn current_directory_value() -> Result<Value, String> {
    let current = env::current_dir()
        .map_err(|err| format!("cd: unable to determine current directory ({err})"))?;
    Ok(path_to_value(&current))
}

fn change_directory(value: &Value) -> Result<Value, String> {
    let target_raw = extract_path(value)?;
    let target = expand_path(&target_raw)?;
    let previous = env::current_dir()
        .map_err(|err| format!("cd: unable to determine current directory ({err})"))?;

    env::set_current_dir(&target)
        .map_err(|err| format!("cd: unable to change directory to '{target_raw}' ({err})"))?;

    Ok(path_to_value(&previous))
}

fn extract_path(value: &Value) -> Result<String, String> {
    match value {
        Value::String(text) => {
            if text.is_empty() {
                Err("cd: folder name must not be empty".to_string())
            } else {
                Ok(text.clone())
            }
        }
        Value::StringArray(array) => {
            if array.data.len() != 1 {
                return Err(
                    "cd: folder name must be a character vector or string scalar".to_string(),
                );
            }
            let text = array.data[0].clone();
            if text.is_empty() {
                Err("cd: folder name must not be empty".to_string())
            } else {
                Ok(text)
            }
        }
        Value::CharArray(chars) => {
            if chars.rows != 1 {
                return Err(
                    "cd: folder name must be a character vector or string scalar".to_string(),
                );
            }
            let text: String = chars.data.iter().collect();
            if text.is_empty() {
                Err("cd: folder name must not be empty".to_string())
            } else {
                Ok(text)
            }
        }
        _ => Err("cd: folder name must be a character vector or string scalar".to_string()),
    }
}

fn expand_path(raw: &str) -> Result<PathBuf, String> {
    if raw == "~" {
        return home_directory().ok_or_else(|| "cd: unable to resolve home directory".to_string());
    }

    if let Some(stripped) = raw.strip_prefix("~/").or_else(|| raw.strip_prefix("~\\")) {
        let home =
            home_directory().ok_or_else(|| "cd: unable to resolve home directory".to_string())?;
        if stripped.is_empty() {
            return Ok(home);
        }
        let mut buf = home;
        buf.push(stripped);
        return Ok(buf);
    }

    Ok(PathBuf::from(raw))
}

fn home_directory() -> Option<PathBuf> {
    #[cfg(windows)]
    {
        if let Ok(user_profile) = env::var("USERPROFILE") {
            return Some(PathBuf::from(user_profile));
        }
        if let (Ok(drive), Ok(path)) = (env::var("HOMEDRIVE"), env::var("HOMEPATH")) {
            return Some(PathBuf::from(format!("{drive}{path}")));
        }
        None
    }
    #[cfg(not(windows))]
    {
        env::var("HOME").map(PathBuf::from).ok()
    }
}

fn gather_arguments(args: &[Value]) -> Result<Vec<Value>, String> {
    let mut out = Vec::with_capacity(args.len());
    for value in args {
        out.push(gather_if_needed(value).map_err(|err| format!("cd: {err}"))?);
    }
    Ok(out)
}

fn path_to_value(path: &Path) -> Value {
    let text = path_to_string(path);
    char_array_value(&text)
}

fn path_to_string(path: &Path) -> String {
    path.to_string_lossy().into_owned()
}

fn char_array_value(text: &str) -> Value {
    Value::CharArray(CharArray::new_row(text))
}

#[cfg(test)]
mod tests {
    use super::super::REPL_FS_TEST_LOCK;
    use super::*;
    use runmat_builtins::StringArray;
    use std::convert::TryFrom;
    use std::path::{Path, PathBuf};
    use tempfile::tempdir;

    fn canonical_path(path: &Path) -> PathBuf {
        std::fs::canonicalize(path).unwrap_or_else(|_| path.to_path_buf())
    }

    struct DirGuard {
        original: PathBuf,
    }

    impl DirGuard {
        fn new() -> Self {
            let original = env::current_dir().expect("current dir");
            Self { original }
        }
    }

    impl Drop for DirGuard {
        fn drop(&mut self) {
            let _ = env::set_current_dir(&self.original);
        }
    }

    #[test]
    fn cd_returns_current_directory() {
        let _lock = REPL_FS_TEST_LOCK
            .lock()
            .unwrap_or_else(|poison| poison.into_inner());
        let _guard = DirGuard::new();

        let expected = env::current_dir().expect("current dir");
        let value = cd_builtin(Vec::new()).expect("cd");
        let actual = String::try_from(&value).expect("string conversion");
        assert_eq!(actual, expected.to_string_lossy());
    }

    #[test]
    fn cd_changes_directory_and_returns_previous() {
        let _lock = REPL_FS_TEST_LOCK
            .lock()
            .unwrap_or_else(|poison| poison.into_inner());
        let _guard = DirGuard::new();

        let original = env::current_dir().expect("current dir");
        let temp = tempdir().expect("tempdir");
        let path_str = temp.path().to_string_lossy().to_string();

        let previous = cd_builtin(vec![Value::from(path_str)]).expect("cd change");
        let previous_str = String::try_from(&previous).expect("string conversion");
        let previous_path = PathBuf::from(previous_str);
        assert_eq!(canonical_path(&previous_path), canonical_path(&original));

        let new_dir = env::current_dir().expect("current dir");
        assert_eq!(canonical_path(&new_dir), canonical_path(temp.path()));
    }

    #[test]
    fn cd_supports_relative_char_array_paths() {
        let _lock = REPL_FS_TEST_LOCK
            .lock()
            .unwrap_or_else(|poison| poison.into_inner());
        let _guard = DirGuard::new();

        let root = tempdir().expect("root tempdir");
        let child = root.path().join("child");
        std::fs::create_dir(&child).expect("create child");

        let _ = cd_builtin(vec![Value::from(root.path().to_string_lossy().to_string())])
            .expect("cd root");

        let relative = Value::CharArray(CharArray::new_row("child"));
        let previous = cd_builtin(vec![relative]).expect("cd child");
        let previous_str = String::try_from(&previous).expect("string conversion");
        let previous_path = PathBuf::from(previous_str);
        assert_eq!(canonical_path(&previous_path), canonical_path(root.path()));
        let current = env::current_dir().expect("current dir");
        assert_eq!(canonical_path(&current), canonical_path(&child));
    }

    #[test]
    fn cd_errors_when_folder_missing() {
        let _lock = REPL_FS_TEST_LOCK
            .lock()
            .unwrap_or_else(|poison| poison.into_inner());
        let _guard = DirGuard::new();

        let missing = Value::from("this-directory-does-not-exist".to_string());
        let err = cd_builtin(vec![missing]).expect_err("error");
        assert!(err.contains("cd: unable to change directory"));
    }

    #[test]
    fn cd_tilde_expands_to_home_directory() {
        let _lock = REPL_FS_TEST_LOCK
            .lock()
            .unwrap_or_else(|poison| poison.into_inner());
        let guard = DirGuard::new();
        let original = guard.original.clone();

        let home = home_directory().expect("home directory");
        let previous = cd_builtin(vec![Value::from("~")]).expect("cd ~");
        let previous_str = String::try_from(&previous).expect("string conversion");
        let previous_path = PathBuf::from(previous_str);

        assert_eq!(canonical_path(&previous_path), canonical_path(&original));
        let current = env::current_dir().expect("current dir");
        assert_eq!(canonical_path(&current), canonical_path(&home));
        drop(guard);
    }

    #[test]
    fn cd_errors_on_empty_string() {
        let _lock = REPL_FS_TEST_LOCK
            .lock()
            .unwrap_or_else(|poison| poison.into_inner());
        let _guard = DirGuard::new();

        let err = cd_builtin(vec![Value::from("".to_string())]).expect_err("empty string error");
        assert_eq!(err, "cd: folder name must not be empty");
    }

    #[test]
    fn cd_errors_on_multi_element_string_array() {
        let _lock = REPL_FS_TEST_LOCK
            .lock()
            .unwrap_or_else(|poison| poison.into_inner());
        let _guard = DirGuard::new();

        let strings =
            StringArray::new(vec!["foo".to_string(), "bar".to_string()], vec![2]).expect("array");
        let err = cd_builtin(vec![Value::StringArray(strings)]).expect_err("string array error");
        assert_eq!(
            err,
            "cd: folder name must be a character vector or string scalar"
        );
    }

    #[test]
    fn cd_errors_on_multiline_char_array() {
        let _lock = REPL_FS_TEST_LOCK
            .lock()
            .unwrap_or_else(|poison| poison.into_inner());
        let _guard = DirGuard::new();

        let chars = CharArray::new(vec!['a', 'b', 'c', 'd'], 2, 2).expect("char array");
        let err = cd_builtin(vec![Value::CharArray(chars)]).expect_err("char array error");
        assert_eq!(
            err,
            "cd: folder name must be a character vector or string scalar"
        );
    }

    #[test]
    fn cd_accepts_string_array_scalar() {
        let _lock = REPL_FS_TEST_LOCK
            .lock()
            .unwrap_or_else(|poison| poison.into_inner());
        let guard = DirGuard::new();

        let current = env::current_dir().expect("current dir");
        let scalar = StringArray::new(vec![current.to_string_lossy().to_string()], vec![1])
            .expect("scalar string array");
        let previous = cd_builtin(vec![Value::StringArray(scalar)]).expect("cd");
        let previous_str = String::try_from(&previous).expect("string conversion");
        assert_eq!(previous_str, current.to_string_lossy());

        drop(guard);
    }

    #[test]
    fn doc_examples_present() {
        let blocks = crate::builtins::common::test_support::doc_examples(DOC_MD);
        assert!(!blocks.is_empty());
    }
}
