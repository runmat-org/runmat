use std::collections::BTreeMap;
use std::path::PathBuf;

use crate::child::{self, ChildProcess};
use crate::environment::EnvironmentPolicy;
use crate::{ProcessHostError, ProcessHostResult};

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum StdioPolicy {
    Piped,
}

#[derive(Clone, Debug)]
pub struct HostCommand {
    pub executable: PathBuf,
    pub arguments: Vec<String>,
    pub environment: BTreeMap<String, String>,
    pub environment_policy: EnvironmentPolicy,
    pub stdio: StdioPolicy,
    pub max_stderr_bytes: usize,
}

impl HostCommand {
    pub fn new(executable: impl Into<PathBuf>) -> Self {
        Self {
            executable: executable.into(),
            arguments: Vec::new(),
            environment: BTreeMap::new(),
            environment_policy: EnvironmentPolicy::Clear,
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
        if self.max_stderr_bytes == 0 {
            return Err(ProcessHostError::Configuration(
                "child stderr bound must be greater than zero".into(),
            ));
        }
        Ok(())
    }

    pub async fn spawn(self) -> ProcessHostResult<ChildProcess> {
        self.validate()?;
        child::spawn(self).await
    }
}
