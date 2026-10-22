//! Grouping-variable normalization shared by grouping operations.

mod categorical;
mod column;
mod labels;
mod numeric;
mod observations;
mod retained;
mod temporal;
mod text;

pub(crate) use column::{columns_from_arguments, columns_from_value, GroupColumn};
pub(crate) use labels::{label_columns, select_group_rows};
pub(crate) use observations::build_index;

#[derive(Debug)]
pub(crate) struct VariableError {
    message: String,
    source: Option<Box<dyn std::error::Error + Send + Sync>>,
}

impl VariableError {
    pub(crate) fn with_source(
        message: impl Into<String>,
        source: impl std::error::Error + Send + Sync + 'static,
    ) -> Self {
        Self {
            message: message.into(),
            source: Some(Box::new(source)),
        }
    }

    pub(crate) fn attach(self, mut error: crate::RuntimeError) -> crate::RuntimeError {
        error.source = Some(Box::new(self));
        error
    }
}

impl From<String> for VariableError {
    fn from(message: String) -> Self {
        Self {
            message,
            source: None,
        }
    }
}

impl From<&str> for VariableError {
    fn from(message: &str) -> Self {
        message.to_owned().into()
    }
}

impl std::fmt::Display for VariableError {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        formatter.write_str(&self.message)
    }
}

impl std::error::Error for VariableError {
    fn source(&self) -> Option<&(dyn std::error::Error + 'static)> {
        self.source.as_deref().map(|source| source as _)
    }
}

pub(crate) type VariableResult<T> = Result<T, VariableError>;
