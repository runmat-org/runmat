use runmat_builtins::BuiltinErrorDescriptor;

pub(super) const NAME: &str = "genpath";

pub(super) fn descriptor(error: &'static BuiltinErrorDescriptor) -> crate::RuntimeError {
    crate::runtime_descriptor_error(NAME, error)
}

pub(super) fn detail(
    error: &'static BuiltinErrorDescriptor,
    detail: impl AsRef<str>,
) -> crate::RuntimeError {
    crate::runtime_descriptor_error_with_detail(NAME, error, detail.as_ref())
}
