//! MATLAB-compatible `pwd` builtin for RunMat.

use std::env;
use std::path::Path;

use runmat_builtins::{CharArray, Value};
use runmat_macros::runtime_builtin;

use crate::builtins::common::spec::{
    BroadcastSemantics, BuiltinFusionSpec, BuiltinGpuSpec, ConstantStrategy, GpuOpKind,
    ReductionNaN, ResidencyPolicy, ShapeRequirements,
};
use crate::console::{record_console_output, ConsoleStream};
use crate::{build_runtime_error, RuntimeControlFlow};
#[cfg_attr(
    feature = "doc_export",
    runmat_macros::register_doc_text(
        name = "pwd",
        builtin_path = "crate::builtins::io::repl_fs::pwd"
    )
)]
#[cfg_attr(not(feature = "doc_export"), allow(dead_code))]
pub const DOC_MD: &str = r#"---
title: "pwd"
category: "io/repl_fs"
keywords: ["pwd", "current directory", "working folder", "present working directory"]
summary: "Return the absolute path to the folder where RunMat is currently executing."
references:
  - https://www.mathworks.com/help/matlab/ref/pwd.html
gpu_support:
  elementwise: false
  reduction: false
  precisions: []
  broadcasting: "none"
  notes: "Runs entirely on the host CPU. No GPU kernels are launched and no provider hooks are required."
fusion:
  elementwise: false
  reduction: false
  max_inputs: 0
  constants: "inline"
requires_feature: null
tested:
  unit: "builtins::io::repl_fs::pwd::tests"
  integration: "builtins::io::repl_fs::pwd::tests::pwd_reflects_directory_changes"
---

# What does the `pwd` function do in MATLAB / RunMat?
`pwd` returns the absolute path of the folder where RunMat is executing. The result is a character row vector (`1×N`) so existing MATLAB code that expects traditional character output keeps working.

## How does the `pwd` function behave in MATLAB / RunMat?
- Always returns the current working folder of the RunMat process.
- Output is a character vector using the platform-native path separators.
- Does not accept any input arguments; if arguments are provided, MATLAB raises an error. RunMat follows the same rule by reporting `pwd: too many input arguments`.
- Errors if RunMat is unable to query the current folder from the operating system.
- Designed to cooperate with `cd` so workflows like `start = pwd; cd("subfolder"); ...; cd(start);` behave exactly as they do in MATLAB.

## `pwd` Function GPU Execution Behavior
`pwd` never runs on the GPU. It performs a host-side query of the process working directory and returns the result as a character vector. The builtin registers a CPU-only GPU spec with `ResidencyPolicy::GatherImmediately`, ensuring fusion plans always surface the path on the host. Because there are no inputs, nothing is gathered from device memory and acceleration providers do not need to implement any hooks for this builtin.

## GPU residency in RunMat (Do I need `gpuArray`?)
No. `pwd` always operates on the CPU. Even in scripts that perform extensive GPU computation, you can call `pwd` at any time to confirm the working folder without affecting GPU residency or triggering device transfers.

## Examples of using the `pwd` function in MATLAB / RunMat

### Show The Current Working Folder
```matlab
current = pwd;
disp(current);
```
Expected output (example on macOS/Linux):
```matlab
/Users/alex/runmat-project
```
Expected output (example on Windows):
```matlab
C:\Users\alex\runmat-project
```

### Capture The Folder Before Changing Directories
```matlab
startDir = pwd;
if ~exist("results", "dir")
    mkdir("results");
end
cd("results");
% ... produce artifacts in results/ ...
cd(startDir);
```
Expected output:
```matlab
% Restores the original folder after work completes
```

### Combine `pwd` With `cd` To Print Relative Paths
```matlab
old = cd("..");
fprintf("Now working in: %s\n", pwd);
cd(old);
```
Expected output:
```matlab
% Prints the absolute path to the parent directory, then returns to the old folder
% For example:
%   Now working in: /Users/alex
```

### Confirm The Folder Inside Scripts
```matlab
fprintf("Script started in %s\n", pwd);
```
Expected output:
```matlab
% Example output:
%   Script started in /Users/alex/runmat-project
```

### Log The Working Folder Together With Results
```matlab
logFile = "run.log";
fid = fopen(logFile, "w");
fprintf(fid, "Working folder: %s\n", pwd);
fclose(fid);
```
Expected output:
```matlab
% The log file contains a line such as:
%   Working folder: /Users/alex/runmat-project
```

### Handle Errors When The Working Folder Is Unavailable
```matlab
try
    location = pwd;
catch err
    disp(err.message);
end
```
Expected output:
```matlab
% Displays a descriptive error message if RunMat cannot query the folder,
% for example:
%   pwd: unable to determine current directory (Permission denied)
```

## FAQ
- **Why does `pwd` return a character vector instead of a string?** MATLAB historically returns character vectors for `pwd`. RunMat mirrors that behavior so existing code keeps working. Use `string(pwd)` if you prefer string scalars.
- **Does `pwd` reflect changes made with `cd`?** Yes. Any successful `cd` call immediately affects what `pwd` returns, matching MATLAB's semantics.
- **Can `pwd` fail?** It is rare, but the operating system can prevent the process from querying the current folder. In that case RunMat raises an error that includes the OS reason.
- **Does `pwd` normalize the path?** RunMat returns the operating-system path exactly as reported, just like MATLAB.
- **Is `pwd` safe to call from GPU-heavy scripts?** Absolutely. The builtin does not allocate GPU memory or trigger device operations.

## See Also
[cd](./cd)

## Source & Feedback
- Source: [`crates/runmat-runtime/src/builtins/io/repl_fs/pwd.rs`](https://github.com/runmat-org/runmat/blob/main/crates/runmat-runtime/src/builtins/io/repl_fs/pwd.rs)
- Found an issue? [Open a GitHub ticket](https://github.com/runmat-org/runmat/issues/new/choose) with a minimal reproduction.
"#;

#[runmat_macros::register_gpu_spec(builtin_path = "crate::builtins::io::repl_fs::pwd")]
pub const GPU_SPEC: BuiltinGpuSpec = BuiltinGpuSpec {
    name: "pwd",
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
    notes: "Host-only operation that queries the process working folder; no GPU provider hooks are required.",
};

#[runmat_macros::register_fusion_spec(builtin_path = "crate::builtins::io::repl_fs::pwd")]
pub const FUSION_SPEC: BuiltinFusionSpec = BuiltinFusionSpec {
    name: "pwd",
    shape: ShapeRequirements::Any,
    constant_strategy: ConstantStrategy::InlineLiteral,
    elementwise: None,
    reduction: None,
    emits_nan: false,
    notes: "I/O builtins are not eligible for fusion; metadata registered for introspection completeness.",
};

const BUILTIN_NAME: &str = "pwd";

fn pwd_error(message: impl Into<String>) -> RuntimeControlFlow {
    RuntimeControlFlow::Error(
        build_runtime_error(message)
            .with_builtin(BUILTIN_NAME)
            .build(),
    )
}

#[runtime_builtin(
    name = "pwd",
    category = "io/repl_fs",
    summary = "Return the absolute path to the folder where RunMat is currently executing.",
    keywords = "pwd,current directory,working folder,present working directory",
    accel = "cpu",
    suppress_auto_output = true,
    builtin_path = "crate::builtins::io::repl_fs::pwd"
)]
fn pwd_builtin(args: Vec<Value>) -> crate::BuiltinResult<Value> {
    if !args.is_empty() {
        return Err(pwd_error("pwd: too many input arguments"));
    }
    let current =
        env::current_dir().map_err(|err| pwd_error(format!("pwd: unable to determine current directory ({err})")))?;
    emit_path_stdout(&current);
    Ok(path_to_value(&current))
}

fn path_to_value(path: &Path) -> Value {
    let text = path.to_string_lossy().into_owned();
    Value::CharArray(CharArray::new_row(&text))
}

fn emit_path_stdout(path: &Path) {
    record_console_output(ConsoleStream::Stdout, path.to_string_lossy().into_owned());
}

#[cfg(test)]
pub(crate) mod tests {
    use super::super::REPL_FS_TEST_LOCK;
    use super::*;
    use crate::{RuntimeControlFlow, RuntimeError};
    use runmat_builtins::CharArray;
    use std::convert::TryFrom;
    use std::path::PathBuf;
    use tempfile::tempdir;

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

    fn unwrap_error(flow: RuntimeControlFlow) -> RuntimeError {
        match flow {
            RuntimeControlFlow::Error(err) => err,
        }
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn pwd_returns_current_directory() {
        let _lock = REPL_FS_TEST_LOCK
            .lock()
            .unwrap_or_else(|poison| poison.into_inner());
        let _guard = DirGuard::new();

        let expected = env::current_dir().expect("current dir");
        let value = pwd_builtin(Vec::new()).expect("pwd");
        let actual = String::try_from(&value).expect("string conversion");
        assert_eq!(actual, expected.to_string_lossy());
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn pwd_reflects_directory_changes() {
        let _lock = REPL_FS_TEST_LOCK
            .lock()
            .unwrap_or_else(|poison| poison.into_inner());
        let guard = DirGuard::new();

        let temp = tempdir().expect("tempdir");
        env::set_current_dir(temp.path()).expect("set temp dir");

        let value = pwd_builtin(Vec::new()).expect("pwd");
        let actual = String::try_from(&value).expect("string conversion");
        let actual_path = PathBuf::from(actual);
        let expected_path =
            std::fs::canonicalize(temp.path()).unwrap_or_else(|_| temp.path().to_path_buf());
        let canonical_actual =
            std::fs::canonicalize(&actual_path).unwrap_or_else(|_| actual_path.clone());
        assert_eq!(canonical_actual, expected_path);

        drop(guard);
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn pwd_returns_char_array_row_vector() {
        let _lock = REPL_FS_TEST_LOCK
            .lock()
            .unwrap_or_else(|poison| poison.into_inner());
        let _guard = DirGuard::new();

        let value = pwd_builtin(Vec::new()).expect("pwd");
        match value {
            Value::CharArray(CharArray { rows, cols, .. }) => {
                assert_eq!(rows, 1);
                assert!(cols >= 1, "expected at least one column in pwd output");
            }
            other => panic!("expected CharArray result, got {other:?}"),
        }
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn pwd_errors_when_arguments_provided() {
        let _lock = REPL_FS_TEST_LOCK
            .lock()
            .unwrap_or_else(|poison| poison.into_inner());
        let _guard = DirGuard::new();

        let err = unwrap_error(pwd_builtin(vec![Value::Num(1.0)]).expect_err("expected error"));
        assert_eq!(err.message(), "pwd: too many input arguments");
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn doc_examples_present() {
        let blocks = crate::builtins::common::test_support::doc_examples(DOC_MD);
        assert!(!blocks.is_empty());
    }
}
