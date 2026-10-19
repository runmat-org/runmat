//! Attach only an already-classified owning semantic descriptor. Operational
//! errors retain their existing source, identifier and replay disposition.
pub(super) fn declared(
    mut error: crate::RuntimeError,
    descriptor: &'static runmat_builtins::BuiltinErrorDescriptor,
) -> crate::RuntimeError {
    if error.identifier.is_none() {
        error.identifier = descriptor.identifier.map(str::to_owned);
    }
    error
}
