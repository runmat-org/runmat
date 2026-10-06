//! Provider-independent contracts shared by runtime hosts and project authorities.
pub use runmat_workspace_contracts::{
    publication_digest, wire, WorkspaceCaptureRequest, WorkspaceEdit, WorkspaceFile,
    WorkspacePublicationId, WorkspacePublicationOutcome, WorkspacePublicationReceipt,
    WorkspacePublicationRequest, WorkspaceRevision, WorkspaceSnapshot,
    MAX_WORKSPACE_PUBLICATION_BYTES, MAX_WORKSPACE_PUBLICATION_EDITS,
};

/// Original native journal receipt of a visible commit whose durability participating access
/// could not confirm. Its presence is neither successful recovery nor rejection.
pub fn indeterminate_receipt(error: &std::io::Error) -> Option<&WorkspacePublicationReceipt> {
    #[cfg(not(target_arch = "wasm32"))]
    {
        crate::workspace_native::journal_receipt(error)
    }
    #[cfg(target_arch = "wasm32")]
    {
        // Browser providers report publication outcomes through the portable
        // contract; they cannot attach a native journal receipt to an I/O error.
        let _ = error;
        None
    }
}
