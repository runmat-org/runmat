//! MATLAB-compatible `savepath` builtin for persisting the session search path.

use runmat_builtins::{
    BuiltinCompletionPolicy, BuiltinDescriptor, BuiltinErrorDescriptor, BuiltinOutputMode,
    BuiltinParamArity, BuiltinParamDescriptor, BuiltinParamType, BuiltinSignatureDescriptor,
    CharArray, StringArray, Tensor, Value,
};
use runmat_macros::runtime_builtin;

use crate::builtins::common::fs::{expand_user_path, home_directory};
use crate::builtins::common::path_state::{current_path_string, PATH_LIST_SEPARATOR};
use crate::builtins::common::spec::{
    BroadcastSemantics, BuiltinFusionSpec, BuiltinGpuSpec, ConstantStrategy, GpuOpKind,
    ReductionNaN, ResidencyPolicy, ShapeRequirements,
};
use crate::{build_runtime_error, gather_if_needed_async, BuiltinResult, RuntimeError};

use runmat_filesystem as vfs;
use std::env;
use std::io;
use std::path::{Path, PathBuf};

const DEFAULT_FILENAME: &str = "pathdef.m";

#[runmat_macros::register_gpu_spec(builtin_path = "crate::builtins::io::repl_fs::savepath")]
pub const GPU_SPEC: BuiltinGpuSpec = BuiltinGpuSpec {
    name: "savepath",
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
        "Filesystem persistence executes on the host; GPU-resident filenames are gathered before writing pathdef.m.",
};

#[runmat_macros::register_fusion_spec(builtin_path = "crate::builtins::io::repl_fs::savepath")]
pub const FUSION_SPEC: BuiltinFusionSpec = BuiltinFusionSpec {
    name: "savepath",
    shape: ShapeRequirements::Any,
    constant_strategy: ConstantStrategy::InlineLiteral,
    elementwise: None,
    reduction: None,
    emits_nan: false,
    notes:
        "Filesystem side-effects are not eligible for fusion; metadata registered for completeness.",
};

const BUILTIN_NAME: &str = "savepath";

const SAVEPATH_OUTPUT_STATUS: [BuiltinParamDescriptor; 1] = [BuiltinParamDescriptor {
    name: "status",
    ty: BuiltinParamType::NumericScalar,
    arity: BuiltinParamArity::Required,
    default: None,
    description: "0 on success, 1 on failure.",
}];
const SAVEPATH_OUTPUT_STATUS_MESSAGE_ID: [BuiltinParamDescriptor; 3] = [
    BuiltinParamDescriptor {
        name: "status",
        ty: BuiltinParamType::NumericScalar,
        arity: BuiltinParamArity::Required,
        default: None,
        description: "0 on success, 1 on failure.",
    },
    BuiltinParamDescriptor {
        name: "message",
        ty: BuiltinParamType::StringScalar,
        arity: BuiltinParamArity::Required,
        default: None,
        description: "Failure message text, or empty on success.",
    },
    BuiltinParamDescriptor {
        name: "message_id",
        ty: BuiltinParamType::StringScalar,
        arity: BuiltinParamArity::Required,
        default: None,
        description: "Failure identifier, or empty on success.",
    },
];
const SAVEPATH_INPUTS_NONE: [BuiltinParamDescriptor; 0] = [];
const SAVEPATH_INPUTS_FILENAME: [BuiltinParamDescriptor; 1] = [BuiltinParamDescriptor {
    name: "filename",
    ty: BuiltinParamType::StringScalar,
    arity: BuiltinParamArity::Required,
    default: Some("\"pathdef.m\""),
    description: "Target file or target directory for persisted pathdef output.",
}];
const SAVEPATH_SIGNATURES: [BuiltinSignatureDescriptor; 4] = [
    BuiltinSignatureDescriptor {
        label: "status = savepath()",
        inputs: &SAVEPATH_INPUTS_NONE,
        outputs: &SAVEPATH_OUTPUT_STATUS,
    },
    BuiltinSignatureDescriptor {
        label: "status = savepath(filename)",
        inputs: &SAVEPATH_INPUTS_FILENAME,
        outputs: &SAVEPATH_OUTPUT_STATUS,
    },
    BuiltinSignatureDescriptor {
        label: "[status, message, message_id] = savepath()",
        inputs: &SAVEPATH_INPUTS_NONE,
        outputs: &SAVEPATH_OUTPUT_STATUS_MESSAGE_ID,
    },
    BuiltinSignatureDescriptor {
        label: "[status, message, message_id] = savepath(filename)",
        inputs: &SAVEPATH_INPUTS_FILENAME,
        outputs: &SAVEPATH_OUTPUT_STATUS_MESSAGE_ID,
    },
];
const SAVEPATH_ERROR_ARG_TYPE: BuiltinErrorDescriptor = BuiltinErrorDescriptor {
    code: "RM.SAVEPATH.ARG_TYPE",
    identifier: None,
    when: "Filename input is not a character vector, string scalar/array scalar, or tensor of character codes.",
    message: "savepath: filename must be a character vector or string scalar",
};
const SAVEPATH_ERROR_EMPTY_FILENAME: BuiltinErrorDescriptor = BuiltinErrorDescriptor {
    code: "RM.SAVEPATH.EMPTY_FILENAME",
    identifier: None,
    when: "Explicit filename argument is empty.",
    message: "savepath: filename must not be empty",
};
const SAVEPATH_ERROR_TOO_MANY_INPUTS: BuiltinErrorDescriptor = BuiltinErrorDescriptor {
    code: "RM.SAVEPATH.TOO_MANY_INPUTS",
    identifier: None,
    when: "More than one positional input argument is provided.",
    message: "savepath: too many input arguments",
};
const SAVEPATH_ERROR_CANNOT_WRITE: BuiltinErrorDescriptor = BuiltinErrorDescriptor {
    code: "RM.SAVEPATH.CANNOT_WRITE",
    identifier: Some("RunMat:savepath:cannotWriteFile"),
    when: "Pathdef file could not be written.",
    message: "savepath: unable to write file",
};
const SAVEPATH_ERROR_CANNOT_RESOLVE: BuiltinErrorDescriptor = BuiltinErrorDescriptor {
    code: "RM.SAVEPATH.CANNOT_RESOLVE",
    identifier: Some("RunMat:savepath:cannotResolveFile"),
    when: "Pathdef output path could not be resolved.",
    message: "savepath: unable to resolve output path",
};
const SAVEPATH_ERRORS: [BuiltinErrorDescriptor; 5] = [
    SAVEPATH_ERROR_ARG_TYPE,
    SAVEPATH_ERROR_EMPTY_FILENAME,
    SAVEPATH_ERROR_TOO_MANY_INPUTS,
    SAVEPATH_ERROR_CANNOT_WRITE,
    SAVEPATH_ERROR_CANNOT_RESOLVE,
];
pub const SAVEPATH_DESCRIPTOR: BuiltinDescriptor = BuiltinDescriptor {
    signatures: &SAVEPATH_SIGNATURES,
    output_mode: BuiltinOutputMode::ByRequestedOutputCount,
    completion_policy: BuiltinCompletionPolicy::Public,
    errors: &SAVEPATH_ERRORS,
};

fn savepath_error(error: &'static BuiltinErrorDescriptor) -> RuntimeError {
    savepath_error_with_message(error.message, error)
}

fn savepath_error_with_message(
    message: impl Into<String>,
    error: &'static BuiltinErrorDescriptor,
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
    name = "savepath",
    category = "io/repl_fs",
    summary = "Persist the current MATLAB search path to pathdef.m with status outputs.",
    keywords = "savepath,pathdef,search path,runmat path,persist path",
    accel = "cpu",
    suppress_auto_output = true,
    type_resolver(crate::builtins::io::type_resolvers::savepath_type),
    descriptor(crate::builtins::io::repl_fs::savepath::SAVEPATH_DESCRIPTOR),
    builtin_path = "crate::builtins::io::repl_fs::savepath"
)]
async fn savepath_builtin(args: Vec<Value>) -> crate::BuiltinResult<Value> {
    let eval = evaluate(&args).await?;
    if let Some(out_count) = crate::output_count::current_output_count() {
        if out_count == 0 {
            return Ok(Value::OutputList(Vec::new()));
        }
        return Ok(crate::output_count::output_list_with_padding(
            out_count,
            eval.outputs(),
        ));
    }
    Ok(eval.first_output())
}

/// Evaluate `savepath` and expose all MATLAB-style outputs.
pub async fn evaluate(args: &[Value]) -> BuiltinResult<SavepathResult> {
    let gathered = gather_arguments(args).await?;
    let target = match gathered.len() {
        0 => match default_target_path().await {
            Ok(path) => path,
            Err(err) => return Ok(SavepathResult::failure(err.message, err.message_error)),
        },
        1 => {
            let raw = extract_filename(&gathered[0])?;
            if raw.is_empty() {
                return Err(savepath_error(&SAVEPATH_ERROR_EMPTY_FILENAME));
            }
            match resolve_explicit_path(&raw).await {
                Ok(path) => path,
                Err(err) => return Ok(SavepathResult::failure(err.message, err.message_error)),
            }
        }
        _ => return Err(savepath_error(&SAVEPATH_ERROR_TOO_MANY_INPUTS)),
    };

    let path_string = current_path_string();
    match persist_path(&target, &path_string).await {
        Ok(()) => Ok(SavepathResult::success()),
        Err(err) => Ok(SavepathResult::failure(err.message, err.message_error)),
    }
}

#[derive(Debug, Clone)]
pub struct SavepathResult {
    status: f64,
    message: String,
    message_id: String,
}

impl SavepathResult {
    fn success() -> Self {
        Self {
            status: 0.0,
            message: String::new(),
            message_id: String::new(),
        }
    }

    fn failure(message: String, message_error: &'static BuiltinErrorDescriptor) -> Self {
        Self {
            status: 1.0,
            message,
            message_id: message_error.identifier.unwrap_or_default().to_string(),
        }
    }

    pub fn first_output(&self) -> Value {
        Value::Num(self.status)
    }

    pub fn outputs(&self) -> Vec<Value> {
        vec![
            Value::Num(self.status),
            char_array_value(&self.message),
            char_array_value(&self.message_id),
        ]
    }

    #[cfg(test)]
    pub(crate) fn status(&self) -> f64 {
        self.status
    }

    #[cfg(test)]
    pub(crate) fn message(&self) -> &str {
        &self.message
    }

    #[cfg(test)]
    pub(crate) fn message_id(&self) -> &str {
        &self.message_id
    }
}

struct SavepathFailure {
    message: String,
    message_error: &'static BuiltinErrorDescriptor,
}

impl SavepathFailure {
    fn new(message: String, message_error: &'static BuiltinErrorDescriptor) -> Self {
        Self {
            message,
            message_error,
        }
    }

    fn cannot_write(path: &Path, error: io::Error) -> Self {
        Self::new(
            format!(
                "savepath: unable to write \"{}\": {}",
                path.display(),
                error
            ),
            &SAVEPATH_ERROR_CANNOT_WRITE,
        )
    }
}

async fn persist_path(target: &Path, path_string: &str) -> Result<(), SavepathFailure> {
    if let Some(parent) = target.parent() {
        if let Err(err) = vfs::create_dir_all_async(parent).await {
            return Err(SavepathFailure::cannot_write(target, err));
        }
    }

    let contents = build_pathdef_contents(path_string);
    vfs::write_async(target, contents.as_bytes())
        .await
        .map_err(|err| SavepathFailure::cannot_write(target, err))
}

async fn default_target_path() -> Result<PathBuf, SavepathFailure> {
    if let Ok(override_path) = env::var("RUNMAT_PATHDEF") {
        if override_path.trim().is_empty() {
            return Err(SavepathFailure::new(
                "savepath: RUNMAT_PATHDEF is empty".to_string(),
                &SAVEPATH_ERROR_CANNOT_RESOLVE,
            ));
        }
        return resolve_explicit_path(&override_path).await;
    }

    let home = home_directory().ok_or_else(|| {
        SavepathFailure::new(
            "savepath: unable to determine default pathdef location".to_string(),
            &SAVEPATH_ERROR_CANNOT_RESOLVE,
        )
    })?;
    Ok(home.join(".runmat").join(DEFAULT_FILENAME))
}

async fn resolve_explicit_path(text: &str) -> Result<PathBuf, SavepathFailure> {
    let expanded = match expand_user_path(text, "savepath") {
        Ok(path) => path,
        Err(err) => return Err(SavepathFailure::new(err, &SAVEPATH_ERROR_CANNOT_RESOLVE)),
    };
    let mut path = PathBuf::from(&expanded);
    if path_should_be_directory(&path, text).await {
        path.push(DEFAULT_FILENAME);
    }
    Ok(path)
}

async fn path_should_be_directory(path: &Path, original: &str) -> bool {
    if original.ends_with(std::path::MAIN_SEPARATOR) || original.ends_with('/') {
        return true;
    }
    if cfg!(windows) && original.ends_with('\\') {
        return true;
    }
    match vfs::metadata_async(path).await {
        Ok(metadata) => metadata.is_dir(),
        Err(_) => false,
    }
}

fn build_pathdef_contents(path_string: &str) -> String {
    let mut contents = String::new();
    contents.push_str("function p = pathdef\n");
    contents.push_str("%PATHDEF Search path defaults generated by RunMat savepath.\n");
    contents.push_str(
        "%   This file reproduces the MATLAB search path at the time savepath was called.\n",
    );
    if !path_string.is_empty() {
        contents.push_str("%\n");
        contents.push_str("%   Directories on the saved path (in order):\n");
        for entry in path_string.split(PATH_LIST_SEPARATOR) {
            contents.push_str("%   ");
            contents.push_str(entry);
            contents.push('\n');
        }
    }
    contents.push('\n');
    let escaped = path_string.replace('\'', "''");
    contents.push_str("p = '");
    contents.push_str(&escaped);
    contents.push_str("';\n");
    contents.push_str("end\n");
    contents
}

fn extract_filename(value: &Value) -> BuiltinResult<String> {
    match value {
        Value::String(text) => Ok(text.clone()),
        Value::StringArray(StringArray { data, .. }) => {
            if data.len() != 1 {
                Err(savepath_error(&SAVEPATH_ERROR_ARG_TYPE))
            } else {
                Ok(data[0].clone())
            }
        }
        Value::CharArray(chars) => {
            if chars.rows != 1 {
                return Err(savepath_error(&SAVEPATH_ERROR_ARG_TYPE));
            }
            Ok(chars.data.iter().collect())
        }
        Value::Tensor(tensor) => tensor_to_string(tensor),
        Value::GpuTensor(_) => Err(savepath_error(&SAVEPATH_ERROR_ARG_TYPE)),
        _ => Err(savepath_error(&SAVEPATH_ERROR_ARG_TYPE)),
    }
}

fn tensor_to_string(tensor: &Tensor) -> BuiltinResult<String> {
    if tensor.shape.len() > 2 {
        return Err(savepath_error(&SAVEPATH_ERROR_ARG_TYPE));
    }
    if tensor.rows() > 1 {
        return Err(savepath_error(&SAVEPATH_ERROR_ARG_TYPE));
    }

    let mut text = String::with_capacity(tensor.data.len());
    for &code in &tensor.data {
        if !code.is_finite() {
            return Err(savepath_error(&SAVEPATH_ERROR_ARG_TYPE));
        }
        let rounded = code.round();
        if (code - rounded).abs() > 1e-6 {
            return Err(savepath_error(&SAVEPATH_ERROR_ARG_TYPE));
        }
        let int_code = rounded as i64;
        if !(0..=0x10FFFF).contains(&int_code) {
            return Err(savepath_error(&SAVEPATH_ERROR_ARG_TYPE));
        }
        let ch = char::from_u32(int_code as u32)
            .ok_or_else(|| savepath_error(&SAVEPATH_ERROR_ARG_TYPE))?;
        text.push(ch);
    }
    Ok(text)
}

async fn gather_arguments(args: &[Value]) -> BuiltinResult<Vec<Value>> {
    let mut gathered = Vec::with_capacity(args.len());
    for value in args {
        gathered.push(
            gather_if_needed_async(value)
                .await
                .map_err(map_control_flow)?,
        );
    }
    Ok(gathered)
}

fn char_array_value(text: &str) -> Value {
    Value::CharArray(CharArray::new_row(text))
}

#[cfg(test)]
pub(crate) mod tests {
    use super::super::REPL_FS_TEST_LOCK;
    use super::*;
    use crate::builtins::common::path_state::{current_path_string, set_path_string};
    use crate::builtins::common::test_support;
    #[cfg(feature = "wgpu")]
    use runmat_accelerate_api::AccelProvider;
    use runmat_accelerate_api::HostTensorView;
    use std::fs;
    use tempfile::tempdir;

    fn savepath_builtin(args: Vec<Value>) -> BuiltinResult<Value> {
        futures::executor::block_on(super::savepath_builtin(args))
    }

    fn evaluate(args: &[Value]) -> BuiltinResult<SavepathResult> {
        futures::executor::block_on(super::evaluate(args))
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn savepath_descriptor_signatures_cover_core_forms() {
        let labels: Vec<&str> = SAVEPATH_DESCRIPTOR
            .signatures
            .iter()
            .map(|sig| sig.label)
            .collect();
        assert!(labels.contains(&"status = savepath()"));
        assert!(labels.contains(&"status = savepath(filename)"));
        assert!(labels.contains(&"[status, message, message_id] = savepath()"));
        assert!(labels.contains(&"[status, message, message_id] = savepath(filename)"));
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

    struct PathdefEnvGuard {
        previous: Option<String>,
    }

    impl PathdefEnvGuard {
        fn set(path: &Path) -> Self {
            let previous = env::var("RUNMAT_PATHDEF").ok();
            env::set_var("RUNMAT_PATHDEF", path.to_string_lossy().to_string());
            Self { previous }
        }

        fn set_raw(value: &str) -> Self {
            let previous = env::var("RUNMAT_PATHDEF").ok();
            env::set_var("RUNMAT_PATHDEF", value);
            Self { previous }
        }
    }

    impl Drop for PathdefEnvGuard {
        fn drop(&mut self) {
            if let Some(ref value) = self.previous {
                env::set_var("RUNMAT_PATHDEF", value);
            } else {
                env::remove_var("RUNMAT_PATHDEF");
            }
        }
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn savepath_writes_to_default_location_with_env_override() {
        let _lock = REPL_FS_TEST_LOCK
            .lock()
            .unwrap_or_else(|poison| poison.into_inner());
        let _guard = PathGuard::new();

        let temp = tempdir().expect("tempdir");
        let target = temp.path().join("pathdef_default.m");
        let _env_guard = PathdefEnvGuard::set(&target);

        let path_a = temp.path().join("toolbox");
        let path_b = temp.path().join("utils");
        let path_string = format!(
            "{}{}{}",
            path_a.to_string_lossy(),
            PATH_LIST_SEPARATOR,
            path_b.to_string_lossy()
        );
        set_path_string(&path_string);

        let eval = evaluate(&[]).expect("evaluate");
        assert_eq!(eval.status(), 0.0);
        assert!(eval.message().is_empty());
        assert!(eval.message_id().is_empty());

        let contents = fs::read_to_string(&target).expect("pathdef contents");
        assert!(contents.contains("function p = pathdef"));
        assert!(contents.contains(path_a.to_string_lossy().as_ref()));
        assert!(contents.contains(path_b.to_string_lossy().as_ref()));
        assert_eq!(current_path_string(), path_string);
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn savepath_env_override_empty_returns_failure() {
        let _lock = REPL_FS_TEST_LOCK
            .lock()
            .unwrap_or_else(|poison| poison.into_inner());
        let _guard = PathGuard::new();

        let _env_guard = PathdefEnvGuard::set_raw("");
        set_path_string("");

        let eval = evaluate(&[]).expect("evaluate");
        assert_eq!(eval.status(), 1.0);
        assert!(eval.message().contains("RUNMAT_PATHDEF is empty"));
        assert_eq!(
            eval.message_id(),
            SAVEPATH_ERROR_CANNOT_RESOLVE.identifier.unwrap_or_default()
        );
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn savepath_accepts_explicit_filename_argument() {
        let _lock = REPL_FS_TEST_LOCK
            .lock()
            .unwrap_or_else(|poison| poison.into_inner());
        let _guard = PathGuard::new();

        let temp = tempdir().expect("tempdir");
        let target = temp.path().join("custom_pathdef.m");
        set_path_string("");

        let eval =
            evaluate(&[Value::from(target.to_string_lossy().to_string())]).expect("evaluate");
        assert_eq!(eval.status(), 0.0);
        assert!(target.exists());
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn savepath_appends_default_filename_for_directories() {
        let _lock = REPL_FS_TEST_LOCK
            .lock()
            .unwrap_or_else(|poison| poison.into_inner());
        let _guard = PathGuard::new();

        let temp = tempdir().expect("tempdir");
        let dir = temp.path().join("profile");
        fs::create_dir_all(&dir).expect("create dir");
        let expected = dir.join(DEFAULT_FILENAME);

        let eval = evaluate(&[Value::from(dir.to_string_lossy().to_string())]).expect("evaluate");
        assert_eq!(eval.status(), 0.0);
        assert!(expected.exists());
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn savepath_appends_default_filename_for_trailing_separator() {
        let _lock = REPL_FS_TEST_LOCK
            .lock()
            .unwrap_or_else(|poison| poison.into_inner());
        let _guard = PathGuard::new();

        let temp = tempdir().expect("tempdir");
        let dir = temp.path().join("profile_trailing");
        let mut raw = dir.to_string_lossy().to_string();
        raw.push(std::path::MAIN_SEPARATOR);

        set_path_string("");
        let eval = evaluate(&[Value::from(raw)]).expect("evaluate");
        assert_eq!(eval.status(), 0.0);
        assert!(dir.join(DEFAULT_FILENAME).exists());
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn savepath_returns_failure_when_write_fails() {
        let _lock = REPL_FS_TEST_LOCK
            .lock()
            .unwrap_or_else(|poison| poison.into_inner());
        let _guard = PathGuard::new();

        let temp = tempdir().expect("tempdir");
        let target = temp.path().join("readonly_pathdef.m");
        fs::write(&target, "locked").expect("write");
        let mut perms = fs::metadata(&target).expect("metadata").permissions();
        let original_perms = perms.clone();
        perms.set_readonly(true);
        fs::set_permissions(&target, perms).expect("set readonly");

        let eval =
            evaluate(&[Value::from(target.to_string_lossy().to_string())]).expect("evaluate");
        assert_eq!(eval.status(), 1.0);
        assert!(eval.message().contains("unable to write"));
        assert_eq!(
            eval.message_id(),
            SAVEPATH_ERROR_CANNOT_WRITE.identifier.unwrap_or_default()
        );

        // Restore permissions so tempdir cleanup succeeds.
        let _ = fs::set_permissions(&target, original_perms);
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn savepath_outputs_vector_contains_message_and_id() {
        let _lock = REPL_FS_TEST_LOCK
            .lock()
            .unwrap_or_else(|poison| poison.into_inner());
        let _guard = PathGuard::new();

        let temp = tempdir().expect("tempdir");
        let target = temp.path().join("outputs_pathdef.m");
        let eval =
            evaluate(&[Value::from(target.to_string_lossy().to_string())]).expect("evaluate");
        let outputs = eval.outputs();
        assert_eq!(outputs.len(), 3);
        assert!(matches!(outputs[0], Value::Num(0.0)));
        assert!(matches!(outputs[1], Value::CharArray(ref ca) if ca.cols == 0));
        assert!(matches!(outputs[2], Value::CharArray(ref ca) if ca.cols == 0));
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn savepath_rejects_empty_filename() {
        let _lock = REPL_FS_TEST_LOCK
            .lock()
            .unwrap_or_else(|poison| poison.into_inner());
        let _guard = PathGuard::new();

        let err = evaluate(&[Value::from(String::new())]).expect_err("expected error");
        assert_eq!(err.message(), SAVEPATH_ERROR_EMPTY_FILENAME.message);
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn savepath_rejects_non_string_input() {
        let err = savepath_builtin(vec![Value::Num(1.0)]).expect_err("expected error");
        assert!(err.message().contains("savepath"));
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn savepath_accepts_string_array_scalar_argument() {
        let _lock = REPL_FS_TEST_LOCK
            .lock()
            .unwrap_or_else(|poison| poison.into_inner());
        let _guard = PathGuard::new();

        let temp = tempdir().expect("tempdir");
        let target = temp.path().join("string_array_pathdef.m");
        let array = StringArray::new(vec![target.to_string_lossy().to_string()], vec![1])
            .expect("string array");

        set_path_string("");
        let eval = evaluate(&[Value::StringArray(array)]).expect("evaluate");
        assert_eq!(eval.status(), 0.0);
        assert!(target.exists());
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn savepath_rejects_multi_element_string_array() {
        let array = StringArray::new(vec!["a".to_string(), "b".to_string()], vec![1, 2])
            .expect("string array");
        let err = extract_filename(&Value::StringArray(array)).expect_err("expected error");
        assert_eq!(err.message(), SAVEPATH_ERROR_ARG_TYPE.message);
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn savepath_rejects_multi_row_char_array() {
        let chars = CharArray::new("abcd".chars().collect(), 2, 2).expect("char array");
        let err = extract_filename(&Value::CharArray(chars)).expect_err("expected error");
        assert_eq!(err.message(), SAVEPATH_ERROR_ARG_TYPE.message);
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn savepath_rejects_tensor_with_fractional_codes() {
        let tensor = Tensor::new(vec![65.5], vec![1, 1]).expect("tensor");
        let err = extract_filename(&Value::Tensor(tensor)).expect_err("expected error");
        assert_eq!(err.message(), SAVEPATH_ERROR_ARG_TYPE.message);
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn savepath_supports_gpu_tensor_filename() {
        let _lock = REPL_FS_TEST_LOCK
            .lock()
            .unwrap_or_else(|poison| poison.into_inner());
        let _guard = PathGuard::new();

        let temp = tempdir().expect("tempdir");
        let target = temp.path().join("gpu_tensor_pathdef.m");
        set_path_string("");

        test_support::with_test_provider(|provider| {
            let text = target.to_string_lossy().to_string();
            let ascii: Vec<f64> = text.chars().map(|ch| ch as u32 as f64).collect();
            let tensor = Tensor::new(ascii.clone(), vec![1, ascii.len()]).expect("tensor");
            let view = HostTensorView {
                data: &tensor.data,
                shape: &tensor.shape,
            };
            let handle = provider.upload(&view).expect("upload");

            let eval = evaluate(&[Value::GpuTensor(handle.clone())]).expect("evaluate");
            assert_eq!(eval.status(), 0.0);

            provider.free(&handle).expect("free");
        });

        assert!(target.exists());
    }

    #[cfg(feature = "wgpu")]
    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn savepath_supports_gpu_tensor_filename_with_wgpu_provider() {
        let _lock = REPL_FS_TEST_LOCK
            .lock()
            .unwrap_or_else(|poison| poison.into_inner());
        let _guard = PathGuard::new();

        let temp = tempdir().expect("tempdir");
        let target = temp.path().join("wgpu_tensor_pathdef.m");
        set_path_string("");

        let provider = runmat_accelerate::backend::wgpu::provider::register_wgpu_provider(
            runmat_accelerate::backend::wgpu::provider::WgpuProviderOptions::default(),
        )
        .expect("wgpu provider");

        let text = target.to_string_lossy().to_string();
        let ascii: Vec<f64> = text.chars().map(|ch| ch as u32 as f64).collect();
        let tensor = Tensor::new(ascii.clone(), vec![1, ascii.len()]).expect("tensor");
        let view = HostTensorView {
            data: &tensor.data,
            shape: &tensor.shape,
        };
        let handle = provider.upload(&view).expect("upload");

        let eval = evaluate(&[Value::GpuTensor(handle.clone())]).expect("evaluate");
        assert_eq!(eval.status(), 0.0);
        assert!(target.exists());

        provider.free(&handle).expect("free");
    }
}
