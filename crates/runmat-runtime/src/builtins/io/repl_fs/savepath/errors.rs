use runmat_builtins::BuiltinErrorDescriptor;

pub(super) const NAME: &str = "savepath";

pub(super) fn descriptor(error: &'static BuiltinErrorDescriptor) -> crate::RuntimeError {
    crate::runtime_descriptor_error(NAME, error)
}

pub(super) fn detail(
    error: &'static BuiltinErrorDescriptor,
    detail: impl AsRef<str>,
) -> crate::RuntimeError {
    crate::runtime_descriptor_error_with_detail(NAME, error, detail.as_ref())
}

pub(super) struct Failure {
    pub(super) message: String,
    pub(super) descriptor: &'static BuiltinErrorDescriptor,
}

impl Failure {
    pub(super) fn new(
        descriptor: &'static BuiltinErrorDescriptor,
        message: impl Into<String>,
    ) -> Self {
        Self {
            message: message.into(),
            descriptor,
        }
    }

    pub(super) fn write(path: &std::path::Path, error: &std::io::Error) -> Self {
        Self::new(
            &runmat_builtins::SAVEPATH_ERROR_CANNOT_WRITE,
            format!("savepath: unable to write \"{}\": {error}", path.display()),
        )
    }
}
