//! Original catalog bytes retain their content identity without JSON byte-array expansion.
use super::*;
use runmat_execution::value::ValueRef;

#[cfg(test)]
mod tests;

impl AnalysisPublicationBuilder {
    #[cfg(test)]
    pub(in crate::analysis) fn persist_catalog_object(
        &self,
        definition: Digest,
        reference: &ValueRef,
        bytes: &[u8],
    ) -> Result<String, AnalysisPublicationError> {
        self.persist_catalog_object_within(
            definition,
            reference,
            bytes,
            self.source_workspace_limit()?,
        )
    }

    /// Persist one catalog object whose buffers are charged against `maximum` bytes of the
    /// source workspace (the caller divides the allowance between concurrent objects).
    pub(in crate::analysis) fn persist_catalog_object_within(
        &self,
        definition: Digest,
        reference: &ValueRef,
        bytes: &[u8],
        maximum: u64,
    ) -> Result<String, AnalysisPublicationError> {
        if definition != self.definition {
            return Err("catalog object belongs to another publication definition".into());
        }
        if bytes.len() as u64 != reference.encoded_length
            || Digest::sha256(bytes) != reference.logical_digest
        {
            return Err("study catalog object digest or length mismatch".into());
        }
        // The sealed whole-study manifest carries the original catalog, including access,
        // media and schema metadata. This file is its exact immutable content object.
        let relative = format!(
            "sha256_{}/objects/sha256_{}.object",
            definition.to_string().trim_start_matches("sha256:"),
            reference
                .logical_digest
                .to_string()
                .trim_start_matches("sha256:"),
        );
        self.evidence_access
            .persist_bytes(&relative, bytes, maximum)
    }
}
