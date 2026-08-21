#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct RetryPolicy {
    pub max_attempts: u32,
}

impl Default for RetryPolicy {
    fn default() -> Self {
        Self { max_attempts: 1 }
    }
}

impl RetryPolicy {
    pub fn should_retry(self, attempt: u32, infrastructure_failure: bool) -> bool {
        infrastructure_failure && attempt < self.max_attempts.max(1)
    }
}
