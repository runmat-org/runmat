//! MATLAB-compatible `diary` builtin for command-window text logging.

#[cfg(all(test, not(target_arch = "wasm32")))]
use runmat_builtins::DIARY_DESCRIPTOR;
use runmat_builtins::{
    BuiltinErrorDescriptor, DiaryCommand, DIARY_ERROR_ARG_COUNT, DIARY_ERROR_ARG_TYPE,
    DIARY_ERROR_IO, DIARY_ERROR_TOO_MANY_OUTPUTS,
};
use runmat_macros::runtime_builtin;
use runmat_value::{Tensor, Value};

use crate::console;
use crate::{build_runtime_error, BuiltinResult};

const BUILTIN_NAME: &str = "diary";

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

fn diary_io_error(source: std::io::Error) -> crate::RuntimeError {
    let message = format!("diary: failed to open diary file ({source})");
    let mut builder = build_runtime_error(message)
        .with_builtin(BUILTIN_NAME)
        .with_source(source);
    if let Some(identifier) = DIARY_ERROR_IO.identifier {
        builder = builder.with_identifier(identifier);
    }
    builder.build()
}

#[runtime_builtin(
    name = "diary",
    binding_variant = "default",
    builtin_path = "crate::builtins::io::diary"
)]
async fn diary_builtin(args: crate::call::BuiltinArguments) -> BuiltinResult<Value> {
    if crate::output_count::current_output_count().unwrap_or(0) > 0 {
        return Err(diary_error(&DIARY_ERROR_TOO_MANY_OUTPUTS));
    }
    match (args.len(), args.first()) {
        (0, _) => console::toggle_diary().map_err(diary_io_error)?,
        (1, Some(arg)) => match parse_diary_arg(arg)? {
            DiaryCommand::On => {
                console::set_diary_enabled_checked(true).map_err(diary_io_error)?;
            }
            DiaryCommand::Off => console::set_diary_enabled(false),
            DiaryCommand::Filename(path) => {
                console::set_diary_filename_checked(path).map_err(diary_io_error)?;
            }
        },
        _ => return Err(diary_error(&DIARY_ERROR_ARG_COUNT)),
    }
    Ok(empty_return_value())
}

fn parse_diary_arg(value: &Value) -> BuiltinResult<DiaryCommand> {
    let text = scalar_text(value).ok_or_else(|| diary_error(&DIARY_ERROR_ARG_TYPE))?;
    DiaryCommand::decode(&text).ok_or_else(|| diary_error(&DIARY_ERROR_ARG_TYPE))
}

fn scalar_text(value: &Value) -> Option<String> {
    crate::builtins::common::validation::value_text_scalar_string(value)
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

        block_on(diary_builtin(
            crate::builtins::common::test_support::builtin_arguments(vec![Value::from(
                "session.log",
            )]),
        ))
        .expect("diary on");
        console::record_console_line(console::ConsoleStream::Stdout, "alpha")
            .expect("record console line");
        block_on(diary_builtin(
            crate::builtins::common::test_support::builtin_arguments(vec![Value::from("off")]),
        ))
        .expect("diary off");
        console::record_console_line(console::ConsoleStream::Stdout, "beta")
            .expect("record console line");

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

        block_on(diary_builtin(
            crate::builtins::common::test_support::builtin_arguments(vec![Value::from(
                "session.log",
            )]),
        ))
        .expect("diary on");
        console::record_console_line(console::ConsoleStream::Stdout, "next")
            .expect("record console line");
        block_on(diary_builtin(
            crate::builtins::common::test_support::builtin_arguments(vec![Value::from("off")]),
        ))
        .expect("diary off");

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

        block_on(diary_builtin(
            crate::builtins::common::test_support::builtin_arguments(vec![Value::from(
                "session.log",
            )]),
        ))
        .expect("diary on");
        let capture = console::begin_capture().expect("console capture");
        console::record_console_line(console::ConsoleStream::Stdout, "captured-only")
            .expect("record console line");
        let captured = capture.finish().expect("captured text");
        console::record_console_line(console::ConsoleStream::Stdout, "logged")
            .expect("record console line");
        block_on(diary_builtin(
            crate::builtins::common::test_support::builtin_arguments(vec![Value::from("off")]),
        ))
        .expect("diary off");

        assert_eq!(captured, "captured-only\n");
        let text = std::fs::read_to_string(dir.path().join("session.log")).expect("read log");
        assert!(!text.contains("captured-only"));
        assert!(text.contains("logged\n"));
    }

    #[test]
    fn requested_output_rejects_before_opening_file_or_changing_state() {
        let _lock = runmat_filesystem::provider_override_lock();
        let dir = tempfile::tempdir().expect("tempdir");
        let provider =
            runmat_filesystem::SandboxFsProvider::new(dir.path().to_path_buf()).expect("sandbox");
        let _guard = runmat_filesystem::replace_provider(Arc::new(provider));
        let _outputs = crate::output_count::push_output_count(Some(1)).expect("output-count scope");
        let error = block_on(diary_builtin(
            crate::builtins::common::test_support::builtin_arguments(vec![Value::from(
                "should-not-open.log",
            )]),
        ))
        .expect_err("diary has no public output");
        assert_eq!(error.identifier(), Some("RunMat:diary:TooManyOutputs"));
        assert!(!dir.path().join("should-not-open.log").exists());
    }

    #[test]
    fn paged_character_name_rejected_but_trailing_singleton_admitted() {
        let paged = runmat_value::CharArray::from_unicode_scalars_with_shape(
            "ab".chars().collect(),
            vec![1, 1, 2],
        )
        .expect("paged char");
        assert!(parse_diary_arg(&Value::CharArray(paged)).is_err());
        let trailing = runmat_value::CharArray::from_unicode_scalars_with_shape(
            "log".chars().collect(),
            vec![1, 3, 1],
        )
        .expect("trailing singleton");
        assert_eq!(
            parse_diary_arg(&Value::CharArray(trailing)).expect("supported row"),
            DiaryCommand::Filename("log".into())
        );
    }
}
