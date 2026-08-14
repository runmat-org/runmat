#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum CompilationMode {
    Background,
    DeterministicSynchronous,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct TieringConfig {
    pub generic_hot_threshold: u64,
    pub specialized_hot_threshold: u64,
    pub loop_hot_threshold: u64,
    pub max_sites: usize,
    pub max_profiles_per_site: usize,
    pub max_profile_bytes: usize,
    pub max_versions_per_entry: usize,
    pub max_pending_compilations: usize,
    pub max_code_bytes: u64,
    pub deterministic: bool,
}

impl Default for TieringConfig {
    fn default() -> Self {
        Self {
            generic_hot_threshold: 8,
            specialized_hot_threshold: 16,
            loop_hot_threshold: 32,
            max_sites: 1_024,
            max_profiles_per_site: 4,
            max_profile_bytes: 64 * 1024,
            max_versions_per_entry: 4,
            max_pending_compilations: 2,
            max_code_bytes: 64 * 1024 * 1024,
            deterministic: false,
        }
    }
}

impl TieringConfig {
    pub fn validate(self) -> Result<Self, &'static str> {
        if self.generic_hot_threshold == 0
            || self.specialized_hot_threshold < self.generic_hot_threshold
            || self.loop_hot_threshold == 0
            || self.max_sites == 0
            || self.max_profiles_per_site == 0
            || self.max_profile_bytes == 0
            || self.max_versions_per_entry < 2
            || self.max_pending_compilations == 0
            || self.max_code_bytes == 0
        {
            return Err("tiering configuration must be bounded and thresholds must be ordered");
        }
        Ok(self)
    }

    pub fn compilation_mode(self) -> CompilationMode {
        if self.deterministic {
            CompilationMode::DeterministicSynchronous
        } else {
            CompilationMode::Background
        }
    }
}
