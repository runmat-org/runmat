#[derive(Debug)]
pub struct SyntaxError {
    pub message: String,
    pub position: usize,
    pub found_token: Option<String>,
    pub expected: Option<String>,
}

impl std::fmt::Display for SyntaxError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(
            f,
            "Syntax error at position {}: {}",
            self.position, self.message
        )?;
        if let Some(found) = &self.found_token {
            write!(f, " (found: '{found}')")?;
        }
        if let Some(expected) = &self.expected {
            write!(f, " (expected: {expected})")?;
        }
        Ok(())
    }
}

impl std::error::Error for SyntaxError {}

impl From<String> for SyntaxError {
    fn from(value: String) -> Self {
        SyntaxError {
            message: value,
            position: 0,
            found_token: None,
            expected: None,
        }
    }
}

impl From<SyntaxError> for String {
    fn from(error: SyntaxError) -> Self {
        error.to_string()
    }
}
