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
        let mut paths =
            self.persist_catalog_objects(definition, std::slice::from_ref(reference), |_| {
                Ok(bytes.to_vec())
            })?;
        Ok(paths.remove(0))
    }

    /// Persist catalog objects read one at a time through one immutable-object scope: one
    /// journal lock and recovery for the run, not one per object.
    pub(in crate::analysis) fn persist_catalog_objects(
        &self,
        definition: Digest,
        references: &[ValueRef],
        read: impl Fn(&ValueRef) -> Result<Vec<u8>, AnalysisPublicationError>,
    ) -> Result<Vec<String>, AnalysisPublicationError> {
        if definition != self.definition {
            return Err("catalog object belongs to another publication definition".into());
        }
        let Some(largest) = references
            .iter()
            .map(|reference| reference.encoded_length)
            .max()
        else {
            return Ok(Vec::new());
        };
        let mut scope = self
            .evidence_access
            .immutable_objects(self.source_workspace_limit()?, largest)?;
        references
            .iter()
            .map(|reference| {
                let bytes = read(reference)?;
                if bytes.len() as u64 != reference.encoded_length
                    || Digest::sha256(&bytes) != reference.logical_digest
                {
                    return Err("study catalog object digest or length mismatch".into());
                }
                // The sealed whole-study manifest carries the original catalog, including
                // access, media and schema metadata. This file is its exact immutable content
                // object.
                let relative = format!(
                    "sha256_{}/objects/sha256_{}.object",
                    definition.to_string().trim_start_matches("sha256:"),
                    reference
                        .logical_digest
                        .to_string()
                        .trim_start_matches("sha256:"),
                );
                scope.persist(&relative, &bytes)
            })
            .collect()
    }
}
