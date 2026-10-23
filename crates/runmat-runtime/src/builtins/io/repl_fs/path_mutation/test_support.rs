use std::path::Path;

pub(in crate::builtins::io::repl_fs) struct PathGuard {
    previous: String,
    _lock: std::sync::MutexGuard<'static, ()>,
}

impl PathGuard {
    pub(in crate::builtins::io::repl_fs) fn new() -> Self {
        let lock = runmat_filesystem::provider_override_lock();
        Self {
            previous: crate::builtins::common::path_state::current_path_string(),
            _lock: lock,
        }
    }
}

impl Drop for PathGuard {
    fn drop(&mut self) {
        crate::builtins::common::path_state::set_path_string(&self.previous)
            .expect("test search path update");
    }
}

pub(in crate::builtins::io::repl_fs) fn canonical(path: &Path) -> String {
    crate::builtins::common::fs::path_to_string(&super::lexical::normalize(path))
}
