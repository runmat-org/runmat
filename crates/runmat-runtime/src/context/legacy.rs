//! Sole migration boundary for ambient runtime APIs.
//!
//! R09 routes existing scoped modules through the active [`RuntimeContext`]
//! here. New semantic code must receive `&RuntimeContext` explicitly. The
//! checked RM-1064 inventory assigns complete removal of this bridge to R29.

use super::{scope::active_runtime_context, RuntimeContext};

pub fn active() -> Option<RuntimeContext> {
    active_runtime_context()
}

/// Transitional adapter for args-only builtin bindings. Document semantics
/// receive the existing invocation context; this never creates host authority.
pub(crate) async fn request_document(
    request: super::HostDocumentRequest,
) -> Result<(), crate::RuntimeError> {
    let context = active().ok_or_else(super::host_document::unavailable)?;
    super::host_document::request(&context, request).await
}

pub(crate) fn error_namespace() -> Option<String> {
    active_runtime_context().map(|context| context.error_namespace())
}
