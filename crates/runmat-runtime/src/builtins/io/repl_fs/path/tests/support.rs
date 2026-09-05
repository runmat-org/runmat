use runmat_value::Value;

use crate::builtins::common::path_state::{current_path_string, set_path_string};
use crate::BuiltinResult;

pub(super) fn call(args: Vec<Value>) -> BuiltinResult<Value> {
    futures::executor::block_on(super::super::path_builtin(args))
}

pub(super) struct PathGuard {
    pub(super) previous: String,
}

impl PathGuard {
    pub(super) fn new() -> Self {
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
