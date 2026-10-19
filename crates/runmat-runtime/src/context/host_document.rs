//! Typed document interactions owned by the active application host.
use super::RuntimeContext;
use crate::RuntimeError;
use std::{num::NonZeroUsize, path::PathBuf};

#[derive(Debug, Clone, Copy, PartialEq, Eq, serde::Serialize, serde::Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum HostTextSelection {
    Caret,
    Select,
}

#[derive(Debug, Clone, PartialEq, Eq, serde::Serialize, serde::Deserialize)]
#[serde(deny_unknown_fields)]
pub struct HostTextLocation {
    pub line: NonZeroUsize,
    pub column: NonZeroUsize,
    pub selection: HostTextSelection,
}

#[derive(Debug, Clone, PartialEq, Eq, serde::Serialize, serde::Deserialize)]
#[serde(tag = "kind", rename_all = "snake_case", deny_unknown_fields)]
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
    context: Option<&RuntimeContext>,
    request: HostDocumentRequest,
) -> Result<(), RuntimeError> {
    let context = context.ok_or_else(unavailable)?;
    let host = context.service_ports().host().ok_or_else(unavailable)?;
    host.open_document(request).await
}

pub(super) fn unavailable() -> RuntimeError {
    crate::build_runtime_error("The active host does not provide document or editor interaction")
        .with_identifier("RunMat:host:DocumentInteractionUnavailable")
        .build()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn document_wire_preserves_typed_location_and_rejects_zero_positions() {
        for selection in [HostTextSelection::Caret, HostTextSelection::Select] {
            let request = HostDocumentRequest::Navigate {
                path: PathBuf::from("/sample.m"),
                location: HostTextLocation {
                    line: NonZeroUsize::new(3).unwrap(),
                    column: NonZeroUsize::new(7).unwrap(),
                    selection,
                },
            };
            let wire = serde_json::to_value(&request).unwrap();
            assert_eq!(
                serde_json::from_value::<HostDocumentRequest>(wire.clone()).unwrap(),
                request
            );
            let mut invalid = wire;
            invalid["location"]["line"] = serde_json::json!(0);
            assert!(serde_json::from_value::<HostDocumentRequest>(invalid).is_err());
        }
    }
}
