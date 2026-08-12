//! MATLAB-compatible `diary` builtin for command-window text logging.

use std::path::PathBuf;

use runmat_builtins::{
    BuiltinCompletionPolicy, BuiltinDescriptor, BuiltinErrorDescriptor, BuiltinOutputMode,
    BuiltinParamArity, BuiltinParamDescriptor, BuiltinParamType, BuiltinSignatureDescriptor,
    Tensor, Value,
};
use runmat_macros::runtime_builtin;

use crate::builtins::common::spec::{
    BroadcastSemantics, BuiltinFusionSpec, BuiltinGpuSpec, ConstantStrategy, GpuOpKind,
    ReductionNaN, ResidencyPolicy, ShapeRequirements,
};
use crate::console;
use crate::{build_runtime_error, BuiltinResult};

#[runmat_macros::register_gpu_spec(builtin_path = "crate::builtins::io::diary")]
pub const GPU_SPEC: BuiltinGpuSpec = BuiltinGpuSpec {
    name: "diary",
    op_kind: GpuOpKind::Custom("file-io"),
    supported_precisions: &[],
    broadcast: BroadcastSemantics::None,
    provider_hooks: &[],
    constant_strategy: ConstantStrategy::InlineLiteral,
    residency: ResidencyPolicy::GatherImmediately,
    nan_mode: ReductionNaN::Include,
    two_pass_threshold: None,
    workgroup_size: None,
    accepts_nan_mode: false,
    notes: "Host-side command-window logging. Diary writes through the active filesystem provider and never operates on GPU buffers.",
};

#[runmat_macros::register_fusion_spec(builtin_path = "crate::builtins::io::diary")]
pub const FUSION_SPEC: BuiltinFusionSpec = BuiltinFusionSpec {
    name: "diary",
    shape: ShapeRequirements::Any,
    constant_strategy: ConstantStrategy::InlineLiteral,
    elementwise: None,
    reduction: None,
    emits_nan: false,
    notes: "Side-effecting file sink; excluded from fusion planning.",
};

const BUILTIN_NAME: &str = "diary";

const DIARY_OUTPUTS_EMPTY: [BuiltinParamDescriptor; 0] = [];
const DIARY_INPUTS_EMPTY: [BuiltinParamDescriptor; 0] = [];
const DIARY_INPUTS_ARG: [BuiltinParamDescriptor; 1] = [BuiltinParamDescriptor {
    name: "filenameOrState",
    ty: BuiltinParamType::StringScalar,
    arity: BuiltinParamArity::Required,
    default: None,
    description: "Log filename, `on`, or `off`.",
}];
const DIARY_SIGNATURES: [BuiltinSignatureDescriptor; 4] = [
    BuiltinSignatureDescriptor {
        label: "diary",
        inputs: &DIARY_INPUTS_EMPTY,
        outputs: &DIARY_OUTPUTS_EMPTY,
    },
    BuiltinSignatureDescriptor {
        label: "diary(filename)",
        inputs: &DIARY_INPUTS_ARG,
        outputs: &DIARY_OUTPUTS_EMPTY,
    },
    BuiltinSignatureDescriptor {
        label: "diary off",
        inputs: &DIARY_INPUTS_ARG,
        outputs: &DIARY_OUTPUTS_EMPTY,
    },
    BuiltinSignatureDescriptor {
        label: "diary on",
        inputs: &DIARY_INPUTS_ARG,
        outputs: &DIARY_OUTPUTS_EMPTY,
    },
];

const DIARY_ERROR_ARG_COUNT: BuiltinErrorDescriptor = BuiltinErrorDescriptor {
    code: "RM.DIARY.ARG_COUNT",
    identifier: None,
    when: "More than one input argument is passed to diary.",
    message: "diary: expected zero or one input argument",
};
const DIARY_ERROR_ARG_TYPE: BuiltinErrorDescriptor = BuiltinErrorDescriptor {
    code: "RM.DIARY.ARG_TYPE",
    identifier: None,
    when: "The diary filename/state argument is not a string scalar or character row.",
    message: "diary: expected filename, 'on', or 'off'",
};
const DIARY_ERROR_IO: BuiltinErrorDescriptor = BuiltinErrorDescriptor {
    code: "RM.DIARY.IO",
    identifier: None,
    when: "The selected diary file cannot be opened for append through the active filesystem provider.",
    message: "diary: failed to open diary file",
};
const DIARY_ERROR_TOO_MANY_OUTPUTS: BuiltinErrorDescriptor = BuiltinErrorDescriptor {
    code: "RM.DIARY.TOO_MANY_OUTPUTS",
    identifier: Some("RunMat:diary:TooManyOutputs"),
    when: "One or more output arguments are requested from diary.",
    message: "diary: expected no output arguments",
};
const DIARY_ERRORS: [BuiltinErrorDescriptor; 4] = [
    DIARY_ERROR_ARG_COUNT,
    DIARY_ERROR_ARG_TYPE,
    DIARY_ERROR_IO,
    DIARY_ERROR_TOO_MANY_OUTPUTS,
];

pub const DIARY_DESCRIPTOR: BuiltinDescriptor = BuiltinDescriptor {
    signatures: &DIARY_SIGNATURES,
    output_mode: BuiltinOutputMode::Fixed,
    completion_policy: BuiltinCompletionPolicy::Public,
    errors: &DIARY_ERRORS,
};

fn diary_error(error: &'static BuiltinErrorDescriptor) -> crate::RuntimeError {
    diary_error_with(error, error.message)
}

fn diary_error_with(
    error: &'static BuiltinErrorDescriptor,
    message: impl Into<String>,
) -> crate::RuntimeError {
    let mut builder = build_runtime_error(message).with_builtin(BUILTIN_NAME);
    if let Some(identifier) = error.identifier {
        builder = builder.with_identifier(identifier);
    }
    builder.build()
}

#[runtime_builtin(
    name = "diary",
    category = "io",
    summary = "Log Command Window text to a file.",
    keywords = "diary,log,command window,console,file",
    sink = true,
    suppress_auto_output = true,
    descriptor(crate::builtins::io::diary::DIARY_DESCRIPTOR),
    builtin_path = "crate::builtins::io::diary"
)]
async fn diary_builtin(args: Vec<Value>) -> BuiltinResult<Value> {
    if crate::output_count::current_output_count().unwrap_or(0) > 0 {
        return Err(diary_error(&DIARY_ERROR_TOO_MANY_OUTPUTS));
    }
    match args.as_slice() {
        [] => console::toggle_diary().map_err(|err| {
            diary_error_with(
                &DIARY_ERROR_IO,
                format!("diary: failed to open diary file ({err})"),
            )
        })?,
        [arg] => match parse_diary_arg(arg)? {
            DiaryAction::On => {
                console::set_diary_enabled_checked(true).map_err(|err| {
                    diary_error_with(
                        &DIARY_ERROR_IO,
                        format!("diary: failed to open diary file ({err})"),
                    )
                })?;
            }
            DiaryAction::Off => console::set_diary_enabled(false),
            DiaryAction::Filename(path) => {
                console::set_diary_filename_checked(path).map_err(|err| {
                    diary_error_with(
                        &DIARY_ERROR_IO,
                        format!("diary: failed to open diary file ({err})"),
                    )
                })?;
            }
        },
        _ => return Err(diary_error(&DIARY_ERROR_ARG_COUNT)),
    }
    Ok(empty_return_value())
}

enum DiaryAction {
    On,
    Off,
    Filename(PathBuf),
}

fn parse_diary_arg(value: &Value) -> BuiltinResult<DiaryAction> {
    let text = scalar_text(value).ok_or_else(|| diary_error(&DIARY_ERROR_ARG_TYPE))?;
    let trimmed = text.trim();
    if trimmed.eq_ignore_ascii_case("on") {
        Ok(DiaryAction::On)
    } else if trimmed.eq_ignore_ascii_case("off") {
        Ok(DiaryAction::Off)
    } else if trimmed.is_empty() {
        Err(diary_error(&DIARY_ERROR_ARG_TYPE))
    } else {
        Ok(DiaryAction::Filename(PathBuf::from(trimmed)))
    }
}

fn scalar_text(value: &Value) -> Option<String> {
    match value {
        Value::String(text) => Some(text.clone()),
        Value::CharArray(chars) if chars.rows == 1 => Some(chars.data.iter().collect()),
        Value::StringArray(array) if array.data.len() == 1 => Some(array.data[0].clone()),
        _ => None,
    }
}

fn empty_return_value() -> Value {
    Value::Tensor(Tensor::zeros(vec![0, 0]))
}

#[cfg(all(test, not(target_arch = "wasm32")))]
mod tests {
    use super::*;
    use futures::executor::block_on;
    use std::sync::Arc;

    #[test]
    fn diary_descriptor_covers_core_forms() {
        let labels: Vec<&str> = DIARY_DESCRIPTOR
            .signatures
            .iter()
            .map(|sig| sig.label)
            .collect();
        assert!(labels.contains(&"diary"));
        assert!(labels.contains(&"diary(filename)"));
        assert!(labels.contains(&"diary off"));
        assert!(labels.contains(&"diary on"));
    }

    #[test]
    fn diary_logs_console_output_to_active_provider() {
        let _lock = runmat_filesystem::provider_override_lock();
        let dir = tempfile::tempdir().expect("tempdir");
        let provider =
            runmat_filesystem::SandboxFsProvider::new(dir.path().to_path_buf()).expect("sandbox");
        let _guard = runmat_filesystem::replace_provider(Arc::new(provider));

        block_on(diary_builtin(vec![Value::from("session.log")])).expect("diary on");
        console::record_console_line(console::ConsoleStream::Stdout, "alpha");
        block_on(diary_builtin(vec![Value::from("off")])).expect("diary off");
        console::record_console_line(console::ConsoleStream::Stdout, "beta");

        let text = std::fs::read_to_string(dir.path().join("session.log")).expect("read log");
        assert!(text.contains("alpha\n"));
        assert!(!text.contains("beta\n"));
    }

    #[test]
    fn diary_filename_appends_existing_log() {
        let _lock = runmat_filesystem::provider_override_lock();
        let dir = tempfile::tempdir().expect("tempdir");
        let provider =
            runmat_filesystem::SandboxFsProvider::new(dir.path().to_path_buf()).expect("sandbox");
        let _guard = runmat_filesystem::replace_provider(Arc::new(provider));
        std::fs::write(dir.path().join("session.log"), "seed\n").expect("seed");

        block_on(diary_builtin(vec![Value::from("session.log")])).expect("diary on");
        console::record_console_line(console::ConsoleStream::Stdout, "next");
        block_on(diary_builtin(vec![Value::from("off")])).expect("diary off");

        let text = std::fs::read_to_string(dir.path().join("session.log")).expect("read log");
        assert_eq!(text, "seed\nnext\n");
    }

    #[test]
    fn diary_does_not_log_active_evalc_capture_scope() {
        let _lock = runmat_filesystem::provider_override_lock();
        let dir = tempfile::tempdir().expect("tempdir");
        let provider =
            runmat_filesystem::SandboxFsProvider::new(dir.path().to_path_buf()).expect("sandbox");
        let _guard = runmat_filesystem::replace_provider(Arc::new(provider));

        block_on(diary_builtin(vec![Value::from("session.log")])).expect("diary on");
        let capture = console::begin_capture();
        console::record_console_line(console::ConsoleStream::Stdout, "captured-only");
        let captured = capture.finish();
        console::record_console_line(console::ConsoleStream::Stdout, "logged");
        block_on(diary_builtin(vec![Value::from("off")])).expect("diary off");

        assert_eq!(captured, "captured-only\n");
        let text = std::fs::read_to_string(dir.path().join("session.log")).expect("read log");
        assert!(!text.contains("captured-only"));
        assert!(text.contains("logged\n"));
    }
}
