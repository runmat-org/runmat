use std::fmt;

#[derive(Debug)]
pub struct TestCommandError {
    code: u8,
}

impl TestCommandError {
    pub fn new(code: u8) -> Self {
        Self { code }
    }

    pub fn code(&self) -> u8 {
        self.code
    }
}

impl fmt::Display for TestCommandError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(formatter, "test command exited with status {}", self.code)
    }
}

impl std::error::Error for TestCommandError {}
