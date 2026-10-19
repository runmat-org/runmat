//! Shared canonical workspace codec.
pub(super) use runmat_workspace_contracts::wire::{decode, encode, MAX_WIRE_BYTES};
#[cfg(test)]
mod tests {
    use super::*;
    use crate::workspace::*;

    #[test]
    fn pending_recovery_receipt_survives_the_shared_closed_wire_contract() {
        let outcome = WorkspacePublicationOutcome::CommittedPendingRecovery {
            receipt: WorkspacePublicationReceipt {
                id: WorkspacePublicationId([19; 16]),
                revision: WorkspaceRevision([43; 32]),
            },
        };
        let bytes = encode(&outcome).unwrap();
        assert_eq!(
            decode::<WorkspacePublicationOutcome>(&bytes).unwrap(),
            outcome
        );
        let mut trailing = bytes;
        trailing.push(0);
        assert!(decode::<WorkspacePublicationOutcome>(&trailing).is_err());
    }

    #[test]
    fn uncertain_durability_retains_original_receipt_in_shared_wire() {
        let outcome = WorkspacePublicationOutcome::DurabilityIndeterminate {
            receipt: WorkspacePublicationReceipt {
                id: WorkspacePublicationId([23; 16]),
                revision: WorkspaceRevision([71; 32]),
            },
        };
        assert_eq!(
            decode::<WorkspacePublicationOutcome>(&encode(&outcome).unwrap()).unwrap(),
            outcome
        );
    }

    #[test]
    fn browser_cbor_uses_the_same_closed_rust_contract() {
        // Produced by the browser's standard cbor-x encoder, not this codec.
        let request: WorkspaceCaptureRequest = decode(include_bytes!(
            "../../tests/fixtures/workspace-browser-capture.cbor"
        ))
        .unwrap();
        assert_eq!(request.root, std::path::Path::new("/"));
        assert_eq!(request.max_bytes, 1048576);
        let request: WorkspacePublicationRequest = decode(include_bytes!(
            "../../tests/fixtures/workspace-browser-publication.cbor"
        ))
        .unwrap();
        assert_eq!(request.id, WorkspacePublicationId([1; 16]));
        assert_eq!(request.expected_revision, WorkspaceRevision([7; 32]));
        assert!(
            matches!(&request.edits[0], WorkspaceEdit::Write { bytes, .. } if bytes == &[0, 127, 128, 255])
        );
        assert!(
            matches!(&request.edits[1], WorkspaceEdit::Rename { from, to } if from == std::path::Path::new("old.fea") && to == std::path::Path::new("shared.fea"))
        );
    }

    #[test]
    fn unknown_publication_variants_and_trailing_responses_are_rejected() {
        let invalid = encode(&serde_json::json!({"type": "overwrite_without_check"})).unwrap();
        assert!(decode::<WorkspacePublicationOutcome>(&invalid).is_err());
        let mut bytes = encode(&WorkspacePublicationOutcome::IdReused).unwrap();
        bytes.push(0);
        assert!(decode::<WorkspacePublicationOutcome>(&bytes).is_err());
    }

    #[test]
    fn source_capture_transports_bytes_without_numeric_array_expansion() {
        let source = vec![255; 4 * 1024 * 1024];
        let snapshot = WorkspaceSnapshot {
            root: "/".into(),
            excluded_subtrees: vec![],
            revision: WorkspaceRevision([1; 32]),
            files: vec![WorkspaceFile {
                path: "main.fea".into(),
                version: "v1".into(),
                bytes: source.clone(),
                readonly: false,
            }],
        };
        let encoded = encode(&snapshot).unwrap();
        assert!(encoded.len() < source.len() + 512);
        let decoded: WorkspaceSnapshot = decode(&encoded).unwrap();
        assert_eq!(decoded, snapshot);
    }
}
