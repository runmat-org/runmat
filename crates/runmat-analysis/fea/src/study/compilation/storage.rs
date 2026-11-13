//! Clone demand for an immutable compiled definition; original properties stay shared.
use super::CompiledAnalysisStudy;
use runmat_canonical_codec::{profile_typed_resident_bytes, CanonicalLimits};
impl CompiledAnalysisStudy {
    /// Original metadata and validated property owner, each charged once.
    pub fn retained_storage_bytes_estimate(&self, limits: CanonicalLimits) -> Result<u64, String> {
        self.shared_clone_storage_bytes_estimate(limits)?
            .checked_add(self.properties.retained_storage_bytes_estimate(limits)?)
            .ok_or_else(|| "compiled original storage overflow".into())
    }

    /// Conservative metadata projection plus newly introduced shared-owner storage.
    /// The original validated property inventories are retained separately; this
    /// estimates an additional clone, never their original heap or codec workspace.
    pub fn shared_clone_storage_bytes_estimate(
        &self,
        limits: CanonicalLimits,
    ) -> Result<u64, String> {
        // New contract fields must make this inventory fail to compile until
        // their ownership is accounted for explicitly.
        let Self {
            module_path,
            study_id,
            study_inputs,
            outputs,
            assessments,
            model,
            properties,
            solver_mesh_digest: _,
        } = self;
        // A clone owns typed values; it never builds a value tree.
        let metadata = profile_typed_resident_bytes(
            &(
                module_path,
                study_id,
                study_inputs,
                outputs,
                assessments,
                model,
            ),
            limits,
        )
        .map_err(|error| error.to_string())?;
        metadata
            .checked_add(
                properties
                    .shared_clone_storage_bytes()
                    .ok_or("property clone storage overflow")?,
            )
            .and_then(|bytes| bytes.checked_add(std::mem::size_of::<Self>() as u64))
            .ok_or_else(|| "compiled clone storage overflow".into())
    }
}

#[cfg(test)]
mod tests;
