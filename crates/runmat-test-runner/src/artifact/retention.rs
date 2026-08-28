#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct RetentionPolicy {
    pub keep_successful: bool,
    pub keep_failed: bool,
    pub max_runs: Option<u32>,
}

impl Default for RetentionPolicy {
    fn default() -> Self {
        Self {
            keep_successful: false,
            keep_failed: true,
            max_runs: None,
        }
    }
}
