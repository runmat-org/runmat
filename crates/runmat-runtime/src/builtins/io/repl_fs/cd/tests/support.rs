use std::env;
use std::path::{Path, PathBuf};

use runmat_value::Value;

use crate::BuiltinResult;

pub(super) fn call(args: Vec<Value>) -> BuiltinResult<Value> {
    futures::executor::block_on(super::super::cd_builtin(args))
}

pub(super) fn canonical(path: &Path) -> PathBuf {
    std::fs::canonicalize(path).unwrap_or_else(|_| path.to_path_buf())
}

pub(super) struct DirGuard {
    pub(super) original: PathBuf,
}

impl DirGuard {
    pub(super) fn new() -> Self {
        Self {
            original: env::current_dir().expect("current directory"),
        }
    }
}

impl Drop for DirGuard {
    fn drop(&mut self) {
        let _ = env::set_current_dir(&self.original);
    }
}
