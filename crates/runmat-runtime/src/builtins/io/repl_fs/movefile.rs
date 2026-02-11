//! MATLAB-compatible `movefile` builtin for RunMat.

use runmat_filesystem as vfs;
use std::io;
use std::path::{Path, PathBuf};

use glob::Pattern;
use runmat_builtins::{CharArray, Value};
use runmat_macros::runtime_builtin;

use crate::builtins::common::fs::{contains_wildcards, expand_user_path};
use crate::builtins::common::spec::{
    BroadcastSemantics, BuiltinFusionSpec, BuiltinGpuSpec, ConstantStrategy, GpuOpKind,
    ReductionNaN, ResidencyPolicy, ShapeRequirements,
};
use crate::{build_runtime_error, gather_if_needed_async, BuiltinResult, RuntimeError};

const MESSAGE_ID_OS_ERROR: &str = "MATLAB:MOVEFILE:OSError";
const MESSAGE_ID_SOURCE_NOT_FOUND: &str = "MATLAB:MOVEFILE:FileDoesNotExist";
const MESSAGE_ID_DEST_EXISTS: &str = "MATLAB:MOVEFILE:DestinationExists";
const MESSAGE_ID_DEST_MISSING: &str = "MATLAB:MOVEFILE:DestinationNotFound";
const MESSAGE_ID_DEST_NOT_DIR: &str = "MATLAB:MOVEFILE:DestinationNotDirectory";
const MESSAGE_ID_EMPTY_SOURCE: &str = "MATLAB:MOVEFILE:EmptySource";
const MESSAGE_ID_EMPTY_DEST: &str = "MATLAB:MOVEFILE:EmptyDestination";
const MESSAGE_ID_PATTERN_ERROR: &str = "MATLAB:MOVEFILE:InvalidPattern";

const ERR_SOURCE_ARG: &str = "movefile: source must be a character vector or string scalar";
const ERR_DEST_ARG: &str = "movefile: destination must be a character vector or string scalar";
const ERR_FLAG_ARG: &str =
    "movefile: flag must be the character 'f' supplied as a char vector or string scalar";

#[runmat_macros::register_gpu_spec(builtin_path = "crate::builtins::io::repl_fs::movefile")]
pub const GPU_SPEC: BuiltinGpuSpec = BuiltinGpuSpec {
    name: "movefile",
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
        "Host-only filesystem builtin. GPU-resident path and flag arguments are gathered automatically before moving files.",
};

#[runmat_macros::register_fusion_spec(builtin_path = "crate::builtins::io::repl_fs::movefile")]
pub const FUSION_SPEC: BuiltinFusionSpec = BuiltinFusionSpec {
    name: "movefile",
    shape: ShapeRequirements::Any,
    constant_strategy: ConstantStrategy::InlineLiteral,
    elementwise: None,
    reduction: None,
    emits_nan: false,
    notes: "Filesystem side-effects materialise immediately; metadata registered for completeness.",
};

const BUILTIN_NAME: &str = "movefile";

fn movefile_error(message: impl Into<String>) -> RuntimeError {
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

#[runtime_builtin(
    name = "movefile",
    category = "io/repl_fs",
    summary = "Move or rename files and folders with MATLAB-compatible status, message, and message ID outputs.",
    keywords = "movefile,rename,move file,filesystem,status,message,messageid,force,overwrite",
    accel = "cpu",
    suppress_auto_output = true,
    type_resolver(crate::builtins::io::type_resolvers::movefile_type),
    builtin_path = "crate::builtins::io::repl_fs::movefile"
)]
async fn movefile_builtin(args: Vec<Value>) -> crate::BuiltinResult<Value> {
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

/// Evaluate `movefile` once and expose all outputs.
pub async fn evaluate(args: &[Value]) -> BuiltinResult<MovefileResult> {
    let gathered = gather_arguments(args).await?;
    match gathered.len() {
        0 | 1 => Err(movefile_error("movefile: not enough input arguments")),
        2 => move_operation(&gathered[0], &gathered[1], false),
        3 => {
            let force = parse_force_flag(&gathered[2])?;
            move_operation(&gathered[0], &gathered[1], force)
        }
        _ => Err(movefile_error("movefile: too many input arguments")),
    }
}

#[derive(Debug, Clone)]
pub struct MovefileResult {
    status: f64,
    message: String,
    message_id: String,
}

impl MovefileResult {
    fn success() -> Self {
        Self {
            status: 1.0,
            message: String::new(),
            message_id: String::new(),
        }
    }

    fn failure(message: String, message_id: &str) -> Self {
        Self {
            status: 0.0,
            message,
            message_id: message_id.to_string(),
        }
    }

    fn empty_source() -> Self {
        Self::failure(
            "Source file or folder name must not be empty.".to_string(),
            MESSAGE_ID_EMPTY_SOURCE,
        )
    }

    fn empty_destination() -> Self {
        Self::failure(
            "Destination file or folder name must not be empty.".to_string(),
            MESSAGE_ID_EMPTY_DEST,
        )
    }

    fn source_not_found(display: &str) -> Self {
        Self::failure(
            format!("Source \"{}\" does not exist.", display),
            MESSAGE_ID_SOURCE_NOT_FOUND,
        )
    }

    fn destination_exists(display: &str) -> Self {
        Self::failure(
            format!(
                "Cannot move to \"{}\": destination already exists.",
                display
            ),
            MESSAGE_ID_DEST_EXISTS,
        )
    }

    fn destination_missing(display: &str) -> Self {
        Self::failure(
            format!(
                "Destination folder \"{}\" must exist when moving multiple sources.",
                display
            ),
            MESSAGE_ID_DEST_MISSING,
        )
    }

    fn destination_not_directory(display: &str) -> Self {
        Self::failure(
            format!("Destination \"{}\" must refer to a folder.", display),
            MESSAGE_ID_DEST_NOT_DIR,
        )
    }

    fn glob_pattern_error(pattern: &str, err: &str) -> Self {
        Self::failure(
            format!("Invalid source pattern \"{}\": {}", pattern, err),
            MESSAGE_ID_PATTERN_ERROR,
        )
    }

    fn os_error(source: &str, target: &str, err: &io::Error) -> Self {
        Self::failure(
            format!("Unable to move \"{}\" to \"{}\": {}", source, target, err),
            MESSAGE_ID_OS_ERROR,
        )
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

fn move_operation(
    source: &Value,
    destination: &Value,
    force: bool,
) -> BuiltinResult<MovefileResult> {
    let source_raw = extract_path(source, ERR_SOURCE_ARG)?;
    if source_raw.is_empty() {
        return Ok(MovefileResult::empty_source());
    }

    let destination_raw = extract_path(destination, ERR_DEST_ARG)?;
    if destination_raw.is_empty() {
        return Ok(MovefileResult::empty_destination());
    }

    let source_expanded = expand_user_path(&source_raw, "movefile").map_err(movefile_error)?;
    let destination_expanded =
        expand_user_path(&destination_raw, "movefile").map_err(movefile_error)?;

    if contains_wildcards(&source_expanded) {
        Ok(move_with_pattern(
            &source_expanded,
            &destination_expanded,
            force,
        ))
    } else {
        Ok(move_single_source(
            &source_expanded,
            &destination_expanded,
            force,
        ))
    }
}

fn move_single_source(source: &str, destination: &str, force: bool) -> MovefileResult {
    let source_path = PathBuf::from(source);
    if vfs::metadata(&source_path).is_err() {
        return MovefileResult::source_not_found(source);
    }

    let destination_path = PathBuf::from(destination);
    if destination_path == source_path {
        return MovefileResult::success();
    }

    let destination_meta = vfs::metadata(&destination_path).ok();
    let mut target_path = destination_path.clone();
    let mut remove_target = false;
    let mut remove_is_dir = false;

    if let Some(meta) = &destination_meta {
        if meta.is_dir() {
            let Some(name) = source_path.file_name() else {
                return MovefileResult::os_error(
                    source,
                    destination,
                    &io::Error::other("Cannot determine source file name"),
                );
            };
            target_path = destination_path.join(name);
            if target_path == source_path {
                return MovefileResult::success();
            }
            match vfs::metadata(&target_path) {
                Ok(existing) => {
                    if !force {
                        return MovefileResult::destination_exists(&path_to_display(&target_path));
                    }
                    remove_target = true;
                    remove_is_dir = existing.is_dir();
                }
                Err(err) => {
                    if err.kind() != io::ErrorKind::NotFound {
                        return MovefileResult::os_error(
                            source,
                            &path_to_display(&target_path),
                            &err,
                        );
                    }
                }
            }
        } else if !force {
            if destination_path == source_path {
                return MovefileResult::success();
            }
            return MovefileResult::destination_exists(destination);
        } else {
            remove_target = true;
            remove_is_dir = meta.is_dir();
        }
    }

    let source_display = path_to_display(&source_path);
    let target_display = path_to_display(&target_path);
    let plan = vec![MovePlanEntry::new(
        source_path,
        source_display,
        target_path,
        target_display,
        remove_target,
        remove_is_dir,
    )];

    match execute_plan(&plan) {
        Ok(()) => MovefileResult::success(),
        Err(err) => MovefileResult::os_error(&err.source_display, &err.target_display, &err.error),
    }
}

fn move_with_pattern(pattern: &str, destination: &str, force: bool) -> MovefileResult {
    let pattern_path = Path::new(pattern);
    let (base_dir, name_pattern) = match pattern_path.file_name() {
        Some(name) => (
            pattern_path.parent().unwrap_or_else(|| Path::new(".")),
            name,
        ),
        None => {
            return MovefileResult::glob_pattern_error(pattern, "pattern has no file name");
        }
    };
    let matcher = match Pattern::new(&name_pattern.to_string_lossy()) {
        Ok(matcher) => matcher,
        Err(err) => return MovefileResult::glob_pattern_error(pattern, err.msg),
    };

    let mut matches = Vec::new();
    let entries = match vfs::read_dir(base_dir) {
        Ok(entries) => entries,
        Err(err) => {
            let display = path_to_display(base_dir);
            return MovefileResult::os_error(&display, destination, &err);
        }
    };

    for entry in entries {
        let file_name = entry.file_name().to_string_lossy();
        if matcher.matches(&file_name) {
            matches.push(entry.path().to_path_buf());
        }
    }

    if matches.is_empty() {
        return MovefileResult::source_not_found(pattern);
    }

    let destination_path = PathBuf::from(destination);
    let destination_meta = match vfs::metadata(&destination_path) {
        Ok(meta) => meta,
        Err(_) => return MovefileResult::destination_missing(destination),
    };

    if !destination_meta.is_dir() {
        return MovefileResult::destination_not_directory(destination);
    }

    let mut plan = Vec::with_capacity(matches.len());
    for source_path in matches {
        let display_source = path_to_display(&source_path);
        if vfs::metadata(&source_path).is_err() {
            return MovefileResult::source_not_found(&display_source);
        }
        let Some(name) = source_path.file_name() else {
            return MovefileResult::os_error(
                &display_source,
                destination,
                &io::Error::other("Cannot determine source name"),
            );
        };
        let target_path = destination_path.join(name);
        if target_path == source_path {
            continue;
        }
        let target_display = path_to_display(&target_path);
        match vfs::metadata(&target_path) {
            Ok(existing) => {
                if !force {
                    return MovefileResult::destination_exists(&target_display);
                }
                plan.push(MovePlanEntry::new(
                    source_path.clone(),
                    display_source.clone(),
                    target_path.clone(),
                    target_display,
                    true,
                    existing.is_dir(),
                ));
            }
            Err(err) => {
                if err.kind() != io::ErrorKind::NotFound {
                    return MovefileResult::os_error(&display_source, &target_display, &err);
                }
                plan.push(MovePlanEntry::new(
                    source_path.clone(),
                    display_source,
                    target_path.clone(),
                    target_display,
                    false,
                    false,
                ));
            }
        }
    }

    match execute_plan(&plan) {
        Ok(()) => MovefileResult::success(),
        Err(err) => MovefileResult::os_error(&err.source_display, &err.target_display, &err.error),
    }
}

#[derive(Debug, Clone)]
struct MovePlanEntry {
    source_path: PathBuf,
    source_display: String,
    target_path: PathBuf,
    target_display: String,
    remove_target: bool,
    remove_is_dir: bool,
}

impl MovePlanEntry {
    fn new(
        source_path: PathBuf,
        source_display: String,
        target_path: PathBuf,
        target_display: String,
        remove_target: bool,
        remove_is_dir: bool,
    ) -> Self {
        Self {
            source_path,
            source_display,
            target_path,
            target_display,
            remove_target,
            remove_is_dir,
        }
    }
}

struct MoveError {
    source_display: String,
    target_display: String,
    error: io::Error,
}

fn execute_plan(plan: &[MovePlanEntry]) -> Result<(), MoveError> {
    for entry in plan {
        if entry.remove_target {
            let result = if entry.remove_is_dir {
                vfs::remove_dir_all(&entry.target_path)
            } else {
                vfs::remove_file(&entry.target_path)
            };
            if let Err(err) = result {
                if err.kind() != io::ErrorKind::NotFound {
                    return Err(MoveError {
                        source_display: entry.source_display.clone(),
                        target_display: entry.target_display.clone(),
                        error: err,
                    });
                }
            }
        }

        if let Err(err) = vfs::rename(&entry.source_path, &entry.target_path) {
            return Err(MoveError {
                source_display: entry.source_display.clone(),
                target_display: entry.target_display.clone(),
                error: err,
            });
        }
    }

    Ok(())
}

fn parse_force_flag(value: &Value) -> BuiltinResult<bool> {
    let text = extract_path(value, ERR_FLAG_ARG)?;
    if text.eq_ignore_ascii_case("f") {
        Ok(true)
    } else {
        Err(movefile_error(ERR_FLAG_ARG))
    }
}

fn extract_path(value: &Value, error_message: &str) -> BuiltinResult<String> {
    match value {
        Value::String(text) => Ok(text.clone()),
        Value::CharArray(array) => {
            if array.rows == 1 {
                Ok(array.data.iter().collect())
            } else {
                Err(movefile_error(error_message))
            }
        }
        Value::StringArray(array) => {
            if array.data.len() == 1 {
                Ok(array.data[0].clone())
            } else {
                Err(movefile_error(error_message))
            }
        }
        _ => Err(movefile_error(error_message)),
    }
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

fn char_array_value(text: &str) -> Value {
    Value::CharArray(CharArray::new_row(text))
}

fn path_to_display(path: &Path) -> String {
    path.display().to_string()
}

#[cfg(test)]
pub(crate) mod tests {
    use super::super::REPL_FS_TEST_LOCK;
    use super::*;
    use std::fs::{self, File};
    use tempfile::tempdir;

    fn evaluate(args: &[Value]) -> BuiltinResult<MovefileResult> {
        futures::executor::block_on(super::evaluate(args))
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn movefile_renames_file() {
        let _lock = REPL_FS_TEST_LOCK
            .lock()
            .unwrap_or_else(|poison| poison.into_inner());

        let temp = tempdir().expect("temp dir");
        let source = temp.path().join("source.txt");
        let dest = temp.path().join("dest.txt");
        File::create(&source).expect("create source");

        let eval = evaluate(&[
            Value::from(source.to_string_lossy().to_string()),
            Value::from(dest.to_string_lossy().to_string()),
        ])
        .expect("movefile");
        assert_eq!(eval.status(), 1.0);
        assert!(!source.exists());
        assert!(dest.exists());
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn movefile_moves_into_existing_directory() {
        let _lock = REPL_FS_TEST_LOCK
            .lock()
            .unwrap_or_else(|poison| poison.into_inner());

        let temp = tempdir().expect("temp dir");
        let source = temp.path().join("report.txt");
        let dest_dir = temp.path().join("reports");
        fs::create_dir(&dest_dir).expect("create dest dir");
        File::create(&source).expect("create source");

        let eval = evaluate(&[
            Value::from(source.to_string_lossy().to_string()),
            Value::from(dest_dir.to_string_lossy().to_string()),
        ])
        .expect("movefile");
        assert_eq!(eval.status(), 1.0);
        assert!(dest_dir.join("report.txt").exists());
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn movefile_force_overwrites_existing_file() {
        let _lock = REPL_FS_TEST_LOCK
            .lock()
            .unwrap_or_else(|poison| poison.into_inner());

        let temp = tempdir().expect("temp dir");
        let source = temp.path().join("draft.txt");
        let dest = temp.path().join("final.txt");
        File::create(&source).expect("create source");
        File::create(&dest).expect("create dest");

        let eval = evaluate(&[
            Value::from(source.to_string_lossy().to_string()),
            Value::from(dest.to_string_lossy().to_string()),
            Value::from("f"),
        ])
        .expect("movefile");
        assert_eq!(eval.status(), 1.0);
        assert!(!source.exists());
        assert!(dest.exists());
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn movefile_without_force_preserves_existing_file() {
        let _lock = REPL_FS_TEST_LOCK
            .lock()
            .unwrap_or_else(|poison| poison.into_inner());

        let temp = tempdir().expect("temp dir");
        let source = temp.path().join("draft.txt");
        let dest = temp.path().join("final.txt");
        File::create(&source).expect("create source");
        File::create(&dest).expect("create dest");

        let eval = evaluate(&[
            Value::from(source.to_string_lossy().to_string()),
            Value::from(dest.to_string_lossy().to_string()),
        ])
        .expect("movefile");
        assert_eq!(eval.status(), 0.0);
        assert_eq!(eval.message_id(), MESSAGE_ID_DEST_EXISTS);
        assert!(eval.message().contains("destination already exists."));
        assert!(source.exists());
        assert!(dest.exists());
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn movefile_moves_multiple_files_with_wildcard() {
        let _lock = REPL_FS_TEST_LOCK
            .lock()
            .unwrap_or_else(|poison| poison.into_inner());

        let temp = tempdir().expect("temp dir");
        let dest_dir = temp.path().join("logs");
        fs::create_dir(&dest_dir).expect("create dest dir");
        let file_a = temp.path().join("a.log");
        let file_b = temp.path().join("b.log");
        File::create(&file_a).expect("create a");
        File::create(&file_b).expect("create b");

        let pattern = temp.path().join("*.log");
        let eval = evaluate(&[
            Value::from(pattern.to_string_lossy().to_string()),
            Value::from(dest_dir.to_string_lossy().to_string()),
        ])
        .expect("movefile");
        assert_eq!(eval.status(), 1.0);
        assert!(dest_dir.join("a.log").exists());
        assert!(dest_dir.join("b.log").exists());
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn movefile_reports_missing_source() {
        let _lock = REPL_FS_TEST_LOCK
            .lock()
            .unwrap_or_else(|poison| poison.into_inner());

        let temp = tempdir().expect("temp dir");
        let source = temp.path().join("missing.txt");
        let dest = temp.path().join("dest.txt");

        let eval = evaluate(&[
            Value::from(source.to_string_lossy().to_string()),
            Value::from(dest.to_string_lossy().to_string()),
        ])
        .expect("movefile");
        assert_eq!(eval.status(), 0.0);
        assert_eq!(eval.message_id(), MESSAGE_ID_SOURCE_NOT_FOUND);
        assert!(eval.message().contains("does not exist"));
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn movefile_outputs_char_arrays() {
        let _lock = REPL_FS_TEST_LOCK
            .lock()
            .unwrap_or_else(|poison| poison.into_inner());

        let temp = tempdir().expect("temp dir");
        let source = temp.path().join("source.txt");
        let dest = temp.path().join("dest.txt");
        File::create(&source).expect("create source");

        let eval = evaluate(&[
            Value::from(source.to_string_lossy().to_string()),
            Value::from(dest.to_string_lossy().to_string()),
        ])
        .expect("movefile");
        let outputs = eval.outputs();
        assert_eq!(outputs.len(), 3);
        assert!(matches!(outputs[0], Value::Num(1.0)));
        assert!(matches!(outputs[1], Value::CharArray(ref ca) if ca.cols == 0));
        assert!(matches!(outputs[2], Value::CharArray(ref ca) if ca.cols == 0));
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn movefile_rejects_invalid_flag() {
        let _lock = REPL_FS_TEST_LOCK
            .lock()
            .unwrap_or_else(|poison| poison.into_inner());

        let err = evaluate(&[Value::from("a"), Value::from("b"), Value::Num(1.0)])
            .expect_err("expected error");
        assert_eq!(err.message(), ERR_FLAG_ARG);
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn movefile_force_flag_accepts_uppercase_char_array() {
        let _lock = REPL_FS_TEST_LOCK
            .lock()
            .unwrap_or_else(|poison| poison.into_inner());

        let temp = tempdir().expect("temp dir");
        let source = temp.path().join("draft.txt");
        let dest = temp.path().join("final.txt");
        File::create(&source).expect("create source");
        File::create(&dest).expect("create dest");

        let eval = evaluate(&[
            Value::from(source.to_string_lossy().to_string()),
            Value::from(dest.to_string_lossy().to_string()),
            Value::CharArray(CharArray::new_row("F")),
        ])
        .expect("movefile");
        assert_eq!(eval.status(), 1.0);
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn movefile_same_path_is_success() {
        let _lock = REPL_FS_TEST_LOCK
            .lock()
            .unwrap_or_else(|poison| poison.into_inner());

        let temp = tempdir().expect("temp dir");
        let source = temp.path().join("note.txt");
        File::create(&source).expect("create source");

        let eval = evaluate(&[
            Value::from(source.to_string_lossy().to_string()),
            Value::from(source.to_string_lossy().to_string()),
        ])
        .expect("movefile");
        assert_eq!(eval.status(), 1.0);
        assert!(source.exists());
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn movefile_moving_into_same_directory_is_success() {
        let _lock = REPL_FS_TEST_LOCK
            .lock()
            .unwrap_or_else(|poison| poison.into_inner());

        let temp = tempdir().expect("temp dir");
        let dir = temp.path().join("docs");
        fs::create_dir(&dir).expect("create dir");
        let source = dir.join("readme.txt");
        File::create(&source).expect("create source");

        let eval = evaluate(&[
            Value::from(source.to_string_lossy().to_string()),
            Value::from(dir.to_string_lossy().to_string()),
        ])
        .expect("movefile");
        assert_eq!(eval.status(), 1.0);
        assert!(dir.join("readme.txt").exists());
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn movefile_reports_empty_source() {
        let _lock = REPL_FS_TEST_LOCK
            .lock()
            .unwrap_or_else(|poison| poison.into_inner());

        let eval = evaluate(&[Value::from(""), Value::from("dest.txt")]).expect("movefile");
        assert_eq!(eval.status(), 0.0);
        assert_eq!(eval.message_id(), MESSAGE_ID_EMPTY_SOURCE);
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn movefile_reports_empty_destination() {
        let _lock = REPL_FS_TEST_LOCK
            .lock()
            .unwrap_or_else(|poison| poison.into_inner());

        let eval = evaluate(&[Value::from("source.txt"), Value::from("")]).expect("movefile");
        assert_eq!(eval.status(), 0.0);
        assert_eq!(eval.message_id(), MESSAGE_ID_EMPTY_DEST);
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn movefile_requires_existing_destination_directory_for_pattern() {
        let _lock = REPL_FS_TEST_LOCK
            .lock()
            .unwrap_or_else(|poison| poison.into_inner());

        let temp = tempdir().expect("temp dir");
        let file = temp.path().join("file.log");
        File::create(&file).expect("create file");
        let pattern = temp.path().join("*.log");
        let dest = temp.path().join("missing");

        let eval = evaluate(&[
            Value::from(pattern.to_string_lossy().to_string()),
            Value::from(dest.to_string_lossy().to_string()),
        ])
        .expect("movefile");
        assert_eq!(eval.status(), 0.0);
        assert_eq!(eval.message_id(), MESSAGE_ID_DEST_MISSING);
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn movefile_requires_directory_destination_for_pattern() {
        let _lock = REPL_FS_TEST_LOCK
            .lock()
            .unwrap_or_else(|poison| poison.into_inner());

        let temp = tempdir().expect("temp dir");
        let file = temp.path().join("file.log");
        let dest_file = temp.path().join("dest.log");
        File::create(&file).expect("create file");
        File::create(&dest_file).expect("create dest");
        let pattern = temp.path().join("*.log");

        let eval = evaluate(&[
            Value::from(pattern.to_string_lossy().to_string()),
            Value::from(dest_file.to_string_lossy().to_string()),
        ])
        .expect("movefile");
        assert_eq!(eval.status(), 0.0);
        assert_eq!(eval.message_id(), MESSAGE_ID_DEST_NOT_DIR);
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn movefile_reports_invalid_pattern() {
        let _lock = REPL_FS_TEST_LOCK
            .lock()
            .unwrap_or_else(|poison| poison.into_inner());

        let eval = evaluate(&[
            Value::from("[*.txt"), // unmatched '[' produces glob PatternError
            Value::from("dest"),
        ])
        .expect("movefile");
        assert_eq!(eval.status(), 0.0);
        assert_eq!(eval.message_id(), MESSAGE_ID_PATTERN_ERROR);
    }
}
