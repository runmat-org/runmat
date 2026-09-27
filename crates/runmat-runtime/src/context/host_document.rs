//! Typed document interactions owned by the active application host.
use super::RuntimeContext;
use crate::RuntimeError;
use std::{num::NonZeroUsize, path::PathBuf};

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum HostTextSelection {
    Caret,
    Select,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct HostTextLocation {
    pub line: NonZeroUsize,
    pub column: NonZeroUsize,
    pub selection: HostTextSelection,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum HostDocumentRequest {
    OpenFile {
        path: PathBuf,
    },
    OpenVariable {
        name: String,
    },
    Navigate {
        path: PathBuf,
        location: HostTextLocation,
    },
}

pub(crate) async fn request(
    context: &RuntimeContext,
    request: HostDocumentRequest,
) -> Result<(), RuntimeError> {
    let host = context.service_ports().host().ok_or_else(unavailable)?;
    host.open_document(request).await
}

pub(super) fn unavailable() -> RuntimeError {
    crate::build_runtime_error("The active host does not provide document or editor interaction")
        .with_identifier("RunMat:host:DocumentInteractionUnavailable")
        .build()
}
