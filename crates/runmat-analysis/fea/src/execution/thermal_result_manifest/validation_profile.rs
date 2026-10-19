//! Structural codec admission for metadata validation; no numerical partitions are reconstructed.
use super::*;

impl ThermalSolverResultManifest {
    /// Additional requested validation workspace. The shared codec profile covers original
    /// identity materialization/encoding and temporary definition construction; exact typed
    /// partition and entity duplicate inventories are composed conservatively.
    /// Borrowed manifest storage, workload validation, allocator bookkeeping and RSS are excluded.
    pub fn validation_resident_bytes_upper_bound(&self) -> AnalysisExecutionResult<u64> {
        // Reject embedded numerical Value payloads before invoking their custom serializers.
        if !self.run.fields.is_empty()
            || !self.run.observations.is_empty()
            || !self.run.observation_histories.is_empty()
            || !self.run.series_reductions.is_empty()
            || self.run.structural_static_balance.is_some()
            || self.run.structural_constraint_result.is_some()
            || self.run.structural_connector_result.is_some()
        {
            return Err(invalid("embedded numerical payload in thermal manifest"));
        }
        let profile = runmat_canonical_codec::profile_serialized(
            self,
            CanonicalLimits::new(usize::MAX, usize::MAX, usize::MAX, 64),
        )?;
        let partitions = self
            .fields
            .iter()
            .flatten()
            .try_fold(0usize, |count, entry| {
                count
                    .checked_add(entry.partitions.len())
                    .ok_or(AnalysisExecutionError::Limit(
                        "thermal manifest validation workspace",
                    ))
            })?;
        let entities = self
            .ordered_node_ids
            .len()
            .max(self.ordered_element_ids.len());
        let ids = crate::topology::storage::slots::<
            crate::results::field::RecoveredFieldPartitionIdentity,
        >(partitions)
        .map_err(invalid)?;
        let entities = crate::topology::storage::slots::<u64>(entities).map_err(invalid)?;
        profile
            .materialization_resident_bytes_estimate
            .checked_add(ids)
            .and_then(|bytes| bytes.checked_add(entities))
            .ok_or(AnalysisExecutionError::Limit(
                "thermal manifest validation workspace",
            ))
    }

    /// Admit original metadata validation before constructing its canonical identity or indexes.
    pub fn validate_admitted(&self, maximum_resident_bytes: u64) -> AnalysisExecutionResult<u64> {
        let bytes = self.validation_resident_bytes_upper_bound()?;
        if bytes > maximum_resident_bytes {
            return Err(AnalysisExecutionError::Limit(
                "thermal manifest validation workspace",
            ));
        }
        self.validate()?;
        Ok(bytes)
    }
}
