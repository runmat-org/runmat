use serde::{Deserialize, Serialize};

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(default, deny_unknown_fields)]
pub struct CacheConfig {
    pub max_bytes: u64,
    pub lease_ttl_ms: u64,
    pub transaction_retries: usize,
}

impl Default for CacheConfig {
    fn default() -> Self {
        Self {
            max_bytes: 10 * 1024 * 1024 * 1024,
            lease_ttl_ms: 5 * 60 * 1000,
            transaction_retries: 8,
        }
    }
}

impl CacheConfig {
    pub fn validate(&self) -> Result<(), &'static str> {
        if self.max_bytes == 0 {
            return Err("max_bytes must be greater than zero");
        }
        if self.lease_ttl_ms == 0 {
            return Err("lease_ttl_ms must be greater than zero");
        }
        if self.transaction_retries == 0 {
            return Err("transaction_retries must be greater than zero");
        }
        Ok(())
    }
}
