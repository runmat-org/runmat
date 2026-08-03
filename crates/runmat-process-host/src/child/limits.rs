use crate::{ProcessHostError, ProcessHostResult};

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct ChildLimits {
    pub max_stderr_bytes: usize,
}

impl ChildLimits {
    pub fn validate(self) -> ProcessHostResult<Self> {
        if self.max_stderr_bytes == 0 {
            return Err(ProcessHostError::Configuration(
                "child stderr bound must be greater than zero".into(),
            ));
        }
        Ok(self)
    }
}
