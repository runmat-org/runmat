use super::*;
impl InMemoryAnalysisArtifactStore {
    pub(in crate::analysis::storage) fn load_assessment_metadata_with_workspace(
        &self,
        run_id: &str,
        maximum_bytes: u64,
        maximum_resident_bytes: u64,
    ) -> Result<Option<AssessmentRunEvidence>, AssessmentStoreError> {
        validate_run_id_token(run_id).map_err(AssessmentStoreError::Invalid)?;
        let guard = self.runs.read().map_err(|_| {
            AssessmentStoreError::Invalid("analysis artifact store lock poisoned".into())
        })?;
        match guard.get(run_id) {
            None => Ok(None),
            Some(PersistedAnalysisArtifact::Standard(persisted)) => {
                crate::analysis::storage::assessment::check_encoded_size(
                    persisted.as_ref(),
                    maximum_bytes,
                )?;
                validate_persisted_run(persisted, run_id).map_err(AssessmentStoreError::Invalid)?;
                Ok(Some(AssessmentRunEvidence::from(&persisted.metadata)))
            }
            Some(PersistedAnalysisArtifact::StructuralFrequency(persisted)) => {
                let metadata_bytes = crate::analysis::storage::assessment::encoded_size(
                    persisted.as_ref(),
                    maximum_bytes,
                )?;
                let objects = self.chunks.read().map_err(|_| {
                    AssessmentStoreError::Invalid("analysis object store lock poisoned".into())
                })?;
                frequency::assessment_evidence(
                    persisted,
                    maximum_bytes - metadata_bytes,
                    maximum_resident_bytes,
                    |digest, limit| {
                        let Some(bytes) = objects.get(&digest) else {
                            return Ok(None);
                        };
                        if bytes.len() as u64 > limit {
                            return Err(AssessmentStoreError::ResourceLimit);
                        }
                        Ok(Some(bytes.to_vec()))
                    },
                )
                .map(Some)
            }
            Some(PersistedAnalysisArtifact::AcousticFrequency(persisted)) => {
                crate::analysis::storage::assessment::check_encoded_size(
                    persisted.as_ref(),
                    maximum_bytes,
                )?;
                frequency_sample::assessment_evidence(persisted, maximum_resident_bytes).map(Some)
            }
        }
    }
}
