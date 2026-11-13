//! Domain errors carry classification without importing a builtin descriptor.
#[derive(Debug)]
pub(crate) enum ProductError {
    InvalidInput(String),
    Internal(String),
}
pub(super) fn invalid_input(message: impl Into<String>) -> ProductError {
    ProductError::InvalidInput(message.into())
}
pub(super) fn internal(message: impl Into<String>) -> ProductError {
    ProductError::Internal(message.into())
}
