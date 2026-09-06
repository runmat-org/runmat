use runmat_builtins::BuiltinErrorDescriptor;

pub(super) const NAME: &str = "rmpath";

pub(super) fn descriptor(error: &'static BuiltinErrorDescriptor) -> crate::RuntimeError {
    crate::runtime_descriptor_error(NAME, error)
}

pub(super) fn detail(
    error: &'static BuiltinErrorDescriptor,
    detail: impl AsRef<str>,
) -> crate::RuntimeError {
    crate::runtime_descriptor_error_with_detail(NAME, error, detail.as_ref())
}

pub(super) fn decode(error: super::super::path_mutation::text::DecodeError) -> crate::RuntimeError {
    use super::super::path_mutation::text::DecodeError;
    match error {
        DecodeError::InvalidInput | DecodeError::Provider(_) => {
            descriptor(&runmat_builtins::RMPATH_ERROR_ARGUMENT_TYPE)
        }
        DecodeError::Compatibility(error) => error,
    }
}
