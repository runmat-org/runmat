//! Admit assessment copies beside original preparation and staged physical artifacts.
use super::*;

impl AnalysisPublicationBuilder {
    /// Snapshot a remaining phase allowance; manifest admission freezes the inventory.
    /// The evaluator reuses this bound rather than rescanning full fields for every report.
    pub(in crate::analysis) fn assessment_workspace_limit(
        &self,
    ) -> Result<u64, AnalysisPublicationError> {
        if !*self
            .admission
            .lock()
            .map_err(|_| "staging admission lock poisoned")?
        {
            return Err("assessment workspace requires a frozen publication inventory".into());
        }
        let retained = resources::retained(
            &*self
                .store
                .runs
                .read()
                .map_err(|_| "staging lock poisoned")?,
            &*self
                .store
                .chunks
                .read()
                .map_err(|_| "staging lock poisoned")?,
            &*self
                .store
                .observations
                .read()
                .map_err(|_| "staging lock poisoned")?,
            self.maximum_bytes,
        )?;
        let fixed = retained
            .checked_add(self.preparation_resident_bytes)
            .ok_or_else(|| AnalysisPublicationError::ResourceLimit {
                message: "assessment phase reservation overflow".into(),
            })?;
        self.maximum_bytes
            .checked_sub(fixed)
            .filter(|remaining| *remaining > 0)
            .ok_or_else(|| AnalysisPublicationError::ResourceLimit {
                message: "assessment phase exceeds captured publication allowance".into(),
            })
    }
    pub(in crate::analysis) fn load_assessment_metadata_admitted(
        &self,
        run_id: &str,
        maximum_bytes: u64,
        retained_workspace_bytes: u64,
    ) -> Result<Option<AssessmentRunEvidence>, AssessmentStoreError> {
        validate_run_id_token(run_id).map_err(AssessmentStoreError::Invalid)?;
        {
            let runs = self
                .store
                .runs
                .read()
                .map_err(|_| AssessmentStoreError::Invalid("staging lock poisoned".into()))?;
            if let Some(PersistedAnalysisArtifact::Standard(run)) = runs.get(run_id) {
                // Borrow the same original projection copied by the store, before cloning.
                let required = resident(
                    &(
                        &run.metadata.run.identity,
                        run.metadata.numerically_acceptable,
                        &run.metadata.observation_scalars,
                    ),
                    self.maximum_bytes,
                )?;
                let total = retained_workspace_bytes
                    .checked_add(required)
                    .ok_or(AssessmentStoreError::ResourceLimit)?;
                // Release the borrowed record guard before the common admission takes locks.
                drop(runs);
                self.admit(total)?;
            } else {
                drop(runs);
                self.admit(retained_workspace_bytes)?;
            }
        }
        let available = self
            .assessment_workspace_limit()?
            .checked_sub(retained_workspace_bytes)
            .ok_or(AssessmentStoreError::ResourceLimit)?;
        let result =
            self.store
                .load_assessment_metadata_with_workspace(run_id, maximum_bytes, available)?;
        // Catalog-backed extraction still owns its existing encoded/decode limits. The
        // returned metadata joins cumulative admission before entering the evaluator cache.
        let additional = resident(&result, self.maximum_bytes)?;
        self.admit(
            retained_workspace_bytes
                .checked_add(additional)
                .ok_or(AssessmentStoreError::ResourceLimit)?,
        )?;
        Ok(result)
    }
}
