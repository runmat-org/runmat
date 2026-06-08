pub mod app;
pub mod cli;
pub mod commands;
pub mod diagnostics;
pub mod logging;
pub mod remote;
pub mod telemetry;

use std::fmt;

pub use cli::{Cli, CliOverrideSources};

#[derive(Debug)]
pub struct AlreadyReportedCliError;

impl fmt::Display for AlreadyReportedCliError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str("command failed")
    }
}

impl std::error::Error for AlreadyReportedCliError {}

pub async fn run_cli(cli: Cli, sources: CliOverrideSources) -> anyhow::Result<()> {
    let result = app::bootstrap::run_cli(cli, sources).await;
    telemetry::shutdown();
    result
}

#[cfg(test)]
pub(crate) mod test_support {
    use once_cell::sync::Lazy;
    use std::path::{Path, PathBuf};
    use std::sync::{Mutex, MutexGuard};

    static CWD_LOCK: Lazy<Mutex<()>> = Lazy::new(|| Mutex::new(()));

    pub(crate) struct ScopedCurrentDir {
        _lock: MutexGuard<'static, ()>,
        original: PathBuf,
    }

    impl ScopedCurrentDir {
        pub(crate) fn enter(path: &Path) -> Self {
            let lock = CWD_LOCK.lock().unwrap_or_else(|poison| poison.into_inner());
            let original = std::env::current_dir().expect("current directory");
            std::env::set_current_dir(path).expect("set current directory");
            Self {
                _lock: lock,
                original,
            }
        }
    }

    impl Drop for ScopedCurrentDir {
        fn drop(&mut self) {
            let _ = std::env::set_current_dir(&self.original);
        }
    }
}
