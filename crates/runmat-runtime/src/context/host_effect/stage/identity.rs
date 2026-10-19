//! Identity of the one public invocation across pre-open and post-scan phases.

use super::*;

pub(super) fn request_identity(request: &HostEffectRequest<'_>) -> Digest {
    if let HostEffectWorkload::ReadMatrixText {
        source,
        controls,
        output,
    } = request.workload
    {
        // This new workload has no legacy stage identities to preserve. Its
        // typed tuple is streamed, so a caller-borrowed path needs no heap
        // allocation before the stage's initial reservation.
        return super::super::hash_serializable(&(
            request.operation,
            source.as_os_str().as_encoded_bytes(),
            controls,
            output,
        ));
    }
    // Preserve existing identities for previously staged domains. Domain
    // facts are excluded because final geometry replaces them after scan.
    Digest::sha256(
        serde_json::to_vec(&(request.operation, request.workload.identity_bytes()))
            .expect("typed stage request serializes"),
    )
}
