use std::collections::BTreeMap;
use std::path::PathBuf;

use crate::child::{self, ChildProcess};
use crate::environment::EnvironmentPolicy;
use crate::{ProcessHostError, ProcessHostResult};

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum ChildLifetime {
    Owned,
    Detached,
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub enum StdioPolicy {
    Piped,
    Inherit,
    Null,
    Files { stdout: PathBuf, stderr: PathBuf },
}

#[derive(Clone, Debug)]
pub struct HostCommand {
    pub executable: PathBuf,
    pub arguments: Vec<String>,
    pub working_directory: Option<PathBuf>,
    pub environment: BTreeMap<String, String>,
    pub environment_policy: EnvironmentPolicy,
    pub lifetime: ChildLifetime,
    pub stdio: StdioPolicy,
    pub max_stderr_bytes: usize,
}

impl HostCommand {
    pub fn new(executable: impl Into<PathBuf>) -> Self {
        Self {
            executable: executable.into(),
            arguments: Vec::new(),
            working_directory: None,
            environment: BTreeMap::new(),
            environment_policy: EnvironmentPolicy::Clear,
            lifetime: ChildLifetime::Owned,
            stdio: StdioPolicy::Piped,
            max_stderr_bytes: 1024 * 1024,
        }
    }

    pub fn validate(&self) -> ProcessHostResult<()> {
        if self.executable.as_os_str().is_empty() {
            return Err(ProcessHostError::Configuration(
                "child executable must not be empty".into(),
            ));
        }
        if self
            .working_directory
            .as_ref()
            .is_some_and(|path| !path.is_absolute() || !path.is_dir())
        {
            return Err(ProcessHostError::Configuration(
                "child working directory must be an existing absolute directory".into(),
            ));
        }
        if self.max_stderr_bytes == 0 {
            return Err(ProcessHostError::Configuration(
                "child stderr bound must be greater than zero".into(),
            ));
        }
        if self.lifetime == ChildLifetime::Detached
            && matches!(self.stdio, StdioPolicy::Piped | StdioPolicy::Inherit)
        {
            return Err(ProcessHostError::Configuration(
                "detached children require null or file-backed stdio".into(),
            ));
        }
        Ok(())
    }

    pub async fn spawn(self) -> ProcessHostResult<ChildProcess> {
        self.validate()?;
        child::spawn(self).await
    }
}
