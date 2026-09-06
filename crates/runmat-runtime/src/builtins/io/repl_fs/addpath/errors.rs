use runmat_builtins::BuiltinErrorDescriptor;

pub(super) const NAME: &str = "addpath";

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
        DecodeError::InvalidInput => descriptor(&runmat_builtins::ADDPATH_ERROR_ARGUMENT_TYPE),
        DecodeError::Provider(source) => {
            detail(&runmat_builtins::ADDPATH_ERROR_PROVIDER, source.message())
        }
        DecodeError::Compatibility(error) => error,
    }
}
