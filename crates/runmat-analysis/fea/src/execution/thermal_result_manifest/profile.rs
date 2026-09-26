use super::*;

/// Canonical payload construction facts, excluding transport and retained solver histories.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct ThermalResultEncodingProfile {
    pub contract_count: u64,
    pub encoded_bytes_upper_bound: u64,
    pub largest_contract_materialization_bytes: u64,
}

#[cfg(test)]
#[test]
fn composed_header_admission_uses_the_canonical_contract_byte_ceiling() {
    let maximum = LIMITS.maximum_encoded_bytes as u64;
    ThermalSolverResultManifest::require_header_byte_bound(maximum).unwrap();
    assert!(ThermalSolverResultManifest::require_header_byte_bound(maximum + 1).is_err());
}

impl ThermalSolverResultManifest {
    /// Admission of a composed pre-solve header bound against this contract's wire ceiling.
    pub(crate) fn require_header_byte_bound(bytes: u64) -> AnalysisExecutionResult<()> {
        if bytes > LIMITS.maximum_encoded_bytes as u64 {
            return Err(AnalysisExecutionError::Limit(
                "projected thermal result header",
            ));
        }
        Ok(())
    }
    /// Profile metadata plus every field partition before encoding numerical payloads.
    /// This is post-solve: runtime provenance is known. It is not pre-solve admission.
    pub fn encoding_profile(&self) -> AnalysisExecutionResult<ThermalResultEncodingProfile> {
        self.validate()?;
        let mut result = ThermalResultEncodingProfile {
            contract_count: 0,
            encoded_bytes_upper_bound: 0,
            largest_contract_materialization_bytes: 0,
        };
        self.visit_encoding_profiles(|_, profile| {
            result.contract_count = result
                .contract_count
                .checked_add(1)
                .ok_or_else(|| invalid("encoding contract count overflows"))?;
            result.encoded_bytes_upper_bound = result
                .encoded_bytes_upper_bound
                .checked_add(profile.encoded_bytes_upper_bound)
                .ok_or_else(|| invalid("encoding payload size overflows"))?;
            result.largest_contract_materialization_bytes = result
                .largest_contract_materialization_bytes
                .max(profile.materialization_resident_bytes_estimate);
            Ok(())
        })?;
        Ok(result)
    }

    /// Post-solve output projection governed by the actual transport chunk parameter.
    pub fn transport_profile(
        &self,
        chunk_bytes: usize,
    ) -> AnalysisExecutionResult<AnalysisTransportProfile> {
        let mut transport = AnalysisTransportProfiler::new(chunk_bytes)?;
        self.validate()?;
        self.visit_encoding_profiles(|_, profile| {
            transport.push(profile.encoded_bytes_upper_bound)
        })?;
        transport.finish(AnalysisNumericalObjectKind::SolverResult)
    }

    /// Ordered post-solve plan consumed by the shared bounded incremental builder.
    pub fn transport_plan(
        &self,
        chunk_bytes: usize,
    ) -> AnalysisExecutionResult<AnalysisPayloadPlan> {
        AnalysisTransportProfiler::new(chunk_bytes)?;
        self.validate()?;
        let count = self
            .fields
            .iter()
            .flatten()
            .try_fold(1_usize, |count, entry| {
                count
                    .checked_add(entry.partitions.len())
                    .ok_or_else(|| invalid("transport partition count overflows"))
            })?;
        let mut entries = AnalysisPayloadPlan::reserve_partitions(count)?;
        self.visit_encoding_profiles(|logical_identity, profile| {
            entries.push(AnalysisPayloadPartitionPlan {
                logical_identity,
                encoded_bytes_upper_bound: profile.encoded_bytes_upper_bound,
            });
            Ok(())
        })?;
        AnalysisPayloadPlan::new(
            AnalysisNumericalObjectKind::SolverResult,
            chunk_bytes,
            entries,
        )
    }

    fn visit_encoding_profiles(
        &self,
        mut visit: impl FnMut(
            Digest,
            runmat_canonical_codec::SerializedCanonicalProfile,
        ) -> AnalysisExecutionResult<()>,
    ) -> AnalysisExecutionResult<()> {
        visit(
            self.identity().digest(),
            runmat_canonical_codec::profile_serialized(self, LIMITS)?,
        )?;
        for entry in self.fields.iter().flatten() {
            // A canonical field consists of equal full blocks and at most one trailing block.
            // Cache the previous shape: profiling work is bounded by two shapes per field,
            // rather than traversing a hypothetical scalar for every retained history value.
            let mut previous = None;
            for partition in &entry.partitions {
                let profile = match previous {
                    Some((count, profile)) if count == partition.entity_count => profile,
                    _ => {
                        let profile = RecoveredFieldPartition::profile_shape(
                            &entry.definition,
                            entry.sample,
                            partition.entity_count,
                        )
                        .map_err(invalid)?;
                        previous = Some((partition.entity_count, profile));
                        profile
                    }
                };
                visit(partition.partition.digest(), profile)?;
            }
        }
        Ok(())
    }
}
