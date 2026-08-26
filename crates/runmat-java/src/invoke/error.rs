use crate::{
    exception::capture_pending_exception, JavaException, JvmError, ObjectRegistryError,
    OverloadError,
};

#[derive(Debug, thiserror::Error)]
pub enum JavaInvocationError {
    #[error(transparent)]
    Jvm(#[from] JvmError),
    #[error(transparent)]
    Object(#[from] ObjectRegistryError),
    #[error("Java invocation failed: {0}")]
    Jni(String),
    #[error("{0}")]
    Exception(JavaException),
    #[error("Java invocation returned an unsupported value: {0}")]
    UnsupportedValue(String),
    #[error("Java callback failed: {0}")]
    Callback(String),
    #[error(transparent)]
    Overload(#[from] OverloadError),
}

pub(super) fn jni_error(
    environment: &mut jni::JNIEnv<'_>,
    error: jni::errors::Error,
) -> JavaInvocationError {
    if matches!(error, jni::errors::Error::JavaException) {
        match capture_pending_exception(environment) {
            Ok(exception) => JavaInvocationError::Exception(exception),
            Err(capture_error) => JavaInvocationError::Jni(format!(
                "{error}; Java exception capture failed: {capture_error}"
            )),
        }
    } else {
        JavaInvocationError::Jni(error.to_string())
    }
}
