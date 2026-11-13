use serde::{Deserialize, Serialize};

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct SchemaValidationError {
    pub path: String,
    pub message: String,
}

impl SchemaValidationError {
    pub fn new(path: impl Into<String>, message: impl Into<String>) -> Self {
        Self {
            path: path.into(),
            message: message.into(),
        }
    }
}

impl std::fmt::Display for SchemaValidationError {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(formatter, "{}: {}", self.path, self.message)
    }
}

impl std::error::Error for SchemaValidationError {}

pub(crate) fn validate_token(
    path: &str,
    value: &str,
    maximum_bytes: usize,
) -> Result<(), SchemaValidationError> {
    if value.is_empty()
        || value.len() > maximum_bytes
        || !value.is_ascii()
        || value.chars().any(char::is_control)
    {
        return Err(SchemaValidationError::new(
            path,
            format!("must be 1..={maximum_bytes} bytes of non-control ASCII"),
        ));
    }
    Ok(())
}
