use std::collections::BTreeMap;
use std::path::PathBuf;

use runmat_process_host::environment::{EnvironmentAllowlist, EnvironmentPolicy};

#[derive(Clone, Debug)]
pub struct ProcessBackendConfig {
    pub executable: PathBuf,
    pub worker_arguments: Vec<String>,
    pub environment: BTreeMap<String, String>,
    pub environment_policy: EnvironmentPolicy,
    pub max_workers: usize,
    pub max_stderr_bytes: usize,
    pub project_handoff: Option<runmat_package::FrozenProjectHandoff>,
}

impl ProcessBackendConfig {
    pub fn same_binary(executable: impl Into<PathBuf>) -> Self {
        Self {
            executable: executable.into(),
            worker_arguments: vec!["--__runmat-test-worker".into()],
            environment: BTreeMap::new(),
            environment_policy: EnvironmentPolicy::Allow(EnvironmentAllowlist::platform_runtime()),
            max_workers: std::thread::available_parallelism()
                .map(usize::from)
                .unwrap_or(1),
            max_stderr_bytes: 1024 * 1024,
            project_handoff: None,
        }
    }
}
