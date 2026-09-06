use std::env;
use std::path::Path;

use runmat_value::Value;

pub(super) fn lock() -> std::sync::MutexGuard<'static, ()> {
    super::super::super::REPL_FS_TEST_LOCK
        .lock()
        .unwrap_or_else(|poison| poison.into_inner())
}

pub(super) fn run(
    args: Vec<Value>,
    outputs: usize,
    extensions: bool,
) -> crate::BuiltinResult<Value> {
    let _outputs = crate::output_count::push_output_count(Some(outputs));
    let _extensions = crate::compatibility::push_runmat_extensions_enabled(extensions);
    futures::executor::block_on(super::super::savepath_builtin(args))
}

pub(super) fn run_without_output_context(
    args: Vec<Value>,
    extensions: bool,
) -> crate::BuiltinResult<Value> {
    let _extensions = crate::compatibility::push_runmat_extensions_enabled(extensions);
    futures::executor::block_on(super::super::savepath_builtin(args))
}

pub(super) fn set_path(path: &str) -> PathGuard {
    let previous = crate::builtins::common::path_state::current_path_string();
    crate::builtins::common::path_state::set_path_string(path);
    PathGuard { previous }
}

pub(super) struct PathGuard {
    previous: String,
}

impl Drop for PathGuard {
    fn drop(&mut self) {
        crate::builtins::common::path_state::set_path_string(&self.previous);
    }
}

pub(super) struct EnvironmentGuard {
    previous: Option<String>,
}

impl EnvironmentGuard {
    pub(super) fn set(path: &Path) -> Self {
        let previous = env::var("RUNMAT_PATHDEF").ok();
        env::set_var("RUNMAT_PATHDEF", path.to_string_lossy().as_ref());
        Self { previous }
    }

    pub(super) fn set_text(value: &str) -> Self {
        let previous = env::var("RUNMAT_PATHDEF").ok();
        env::set_var("RUNMAT_PATHDEF", value);
        Self { previous }
    }
}

impl Drop for EnvironmentGuard {
    fn drop(&mut self) {
        match &self.previous {
            Some(value) => env::set_var("RUNMAT_PATHDEF", value),
            None => env::remove_var("RUNMAT_PATHDEF"),
        }
    }
}

pub(super) fn status(value: &Value) -> f64 {
    let Value::OutputList(outputs) = value else {
        panic!("expected output list");
    };
    let Value::Num(status) = outputs[0] else {
        panic!("expected numeric status");
    };
    status
}
