use serde::{Deserialize, Serialize};

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(default, deny_unknown_fields)]
pub struct ArchiveLimits {
    pub max_entries: u64,
    pub max_files: u64,
    pub max_expanded_bytes: u64,
    pub max_file_bytes: u64,
    pub max_path_bytes: usize,
    pub max_component_bytes: usize,
    pub max_compression_ratio: u64,
}

impl Default for ArchiveLimits {
    fn default() -> Self {
        Self {
            max_entries: 200_000,
            max_files: 100_000,
            max_expanded_bytes: 4 * 1024 * 1024 * 1024,
            max_file_bytes: 1024 * 1024 * 1024,
            max_path_bytes: 4096,
            max_component_bytes: 255,
            max_compression_ratio: 1_000,
        }
    }
}

impl ArchiveLimits {
    pub fn validate(&self) -> Result<(), &'static str> {
        if self.max_entries == 0
            || self.max_files == 0
            || self.max_expanded_bytes == 0
            || self.max_file_bytes == 0
            || self.max_path_bytes == 0
            || self.max_component_bytes == 0
            || self.max_compression_ratio == 0
        {
            return Err("archive limits must all be greater than zero");
        }
        Ok(())
    }
}
