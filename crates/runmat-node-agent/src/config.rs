use std::path::PathBuf;
use std::time::Duration;

use serde::{Deserialize, Serialize};

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
        if !self.state_directory.is_absolute()
            || !self.runmat_executable.is_absolute()
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

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase", deny_unknown_fields)]
pub struct AgentFileConfig {
    pub server_url: String,
    pub runmat_executable: PathBuf,
    pub state_directory: PathBuf,
    pub heartbeat_interval_seconds: u64,
    pub heartbeat_ttl_seconds: u64,
    pub drain_timeout_seconds: u64,
    pub maximum_allocations: usize,
    pub trust_tier: runmat_execution::security::ExecutionTrustTier,
}

impl AgentFileConfig {
    pub fn from_runtime(config: &AgentConfig) -> Self {
        Self {
            server_url: config.server_url.clone(),
            runmat_executable: config.runmat_executable.clone(),
            state_directory: config.state_directory.clone(),
            heartbeat_interval_seconds: config.heartbeat_interval.as_secs(),
            heartbeat_ttl_seconds: config.heartbeat_ttl.as_secs(),
            drain_timeout_seconds: config.drain_timeout.as_secs(),
            maximum_allocations: config.maximum_allocations,
            trust_tier: config.trust_tier,
        }
    }

    pub fn into_runtime(self) -> AgentResult<AgentConfig> {
        let config = AgentConfig {
            state_directory: self.state_directory,
            server_url: self.server_url,
            runmat_executable: self.runmat_executable,
            heartbeat_interval: Duration::from_secs(self.heartbeat_interval_seconds),
            heartbeat_ttl: Duration::from_secs(self.heartbeat_ttl_seconds),
            drain_timeout: Duration::from_secs(self.drain_timeout_seconds),
            maximum_allocations: self.maximum_allocations,
            trust_tier: self.trust_tier,
        };
        config.validate()?;
        Ok(config)
    }

    pub fn load(path: &std::path::Path) -> AgentResult<Self> {
        let bytes = std::fs::read(path)?;
        serde_json::from_slice(&bytes).map_err(Into::into)
    }

    pub fn encode_pretty(&self) -> AgentResult<Vec<u8>> {
        serde_json::to_vec_pretty(self).map_err(Into::into)
    }
}
