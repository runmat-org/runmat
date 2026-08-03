use std::path::PathBuf;
use std::time::Duration;

use crate::{AgentError, AgentResult};

#[derive(Debug, Clone)]
pub struct AgentConfig {
    pub state_directory: PathBuf,
    pub server_url: String,
    pub runmat_executable: PathBuf,
    pub heartbeat_interval: Duration,
    pub heartbeat_ttl: Duration,
    pub drain_timeout: Duration,
    pub maximum_allocations: usize,
    pub trust_tier: runmat_execution::security::ExecutionTrustTier,
}

impl AgentConfig {
    pub const DEFAULT_HEARTBEAT_INTERVAL: Duration = Duration::from_secs(15);
    pub const DEFAULT_HEARTBEAT_TTL: Duration = Duration::from_secs(60);
    pub const DEFAULT_DRAIN_TIMEOUT: Duration = Duration::from_secs(30);
    pub const DEFAULT_MAXIMUM_ALLOCATIONS: usize = 1;

    pub fn default_state_directory() -> AgentResult<PathBuf> {
        dirs::data_local_dir()
            .map(|path| path.join("runmat").join("node-agent"))
            .ok_or_else(|| AgentError::Configuration("no local data directory".to_string()))
    }

    pub fn validate(&self) -> AgentResult<()> {
        if !(self.server_url.starts_with("https://")
            || self.server_url.starts_with("http://127.0.0.1")
            || self.server_url.starts_with("http://localhost"))
        {
            return Err(AgentError::Configuration(
                "server URL must use HTTPS except on loopback".to_string(),
            ));
        }
        if !self.runmat_executable.is_absolute()
            || !self.runmat_executable.is_file()
            || self.heartbeat_interval.is_zero()
            || self.heartbeat_ttl <= self.heartbeat_interval
            || self.drain_timeout.is_zero()
            || self.maximum_allocations == 0
        {
            return Err(AgentError::Configuration(
                "paths, heartbeat, drain, or allocation bounds are invalid".to_string(),
            ));
        }
        Ok(())
    }
}
