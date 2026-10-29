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

    /// Persist catalog objects read and verified one at a time through one workspace scope
    /// and one group commit: one journal lock, recovery and durability barrier for the run,
    /// not one per object. The first object is verified before the scope opens, so a
    /// mismatched catalog creates nothing.
    pub(in crate::analysis) fn persist_catalog_objects(
        &self,
        definition: Digest,
        references: &[ValueRef],
        read: impl Fn(&ValueRef) -> Result<Vec<u8>, AnalysisPublicationError>,
    ) -> Result<Vec<String>, AnalysisPublicationError> {
        if definition != self.definition {
            return Err("catalog object belongs to another publication definition".into());
        }
        let verified = |reference: &ValueRef| {
            let bytes = read(reference)?;
            if bytes.len() as u64 != reference.encoded_length
                || Digest::sha256(&bytes) != reference.logical_digest
            {
                return Err::<_, AnalysisPublicationError>(
                    "study catalog object digest or length mismatch".into(),
                );
            }
            Ok(bytes)
        };
        // The sealed whole-study manifest carries the original catalog, including access,
        // media and schema metadata. Each file is its exact immutable content object.
        let relative = |reference: &ValueRef| {
            format!(
                "sha256_{}/objects/sha256_{}.object",
                definition.to_string().trim_start_matches("sha256:"),
                reference
                    .logical_digest
                    .to_string()
                    .trim_start_matches("sha256:"),
            )
        };
        let Some((first, rest)) = references.split_first() else {
            return Ok(Vec::new());
        };
        let first_bytes = verified(first)?;
        let largest = references
            .iter()
            .map(|reference| reference.encoded_length)
            .max()
            .unwrap_or(0);
        self.evidence_access.persist_immutable_batch(
            self.source_workspace_limit()?,
            largest,
            |batch| {
                let mut paths = Vec::with_capacity(references.len());
                paths.push(batch.persist(&relative(first), &first_bytes)?);
                drop(first_bytes);
                for reference in rest {
                    paths.push(batch.persist(&relative(reference), &verified(reference)?)?);
                }
                Ok(paths)
            },
        )
    }
}
