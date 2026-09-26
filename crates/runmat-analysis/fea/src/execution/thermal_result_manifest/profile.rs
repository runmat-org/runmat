use super::*;

/// Canonical payload construction facts, excluding transport and retained solver histories.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct ThermalResultEncodingProfile {
    pub contract_count: u64,
    pub encoded_bytes_upper_bound: u64,
    pub largest_contract_materialization_bytes: u64,
}

impl ThermalSolverResultManifest {
    /// Profile metadata plus every field partition before encoding numerical payloads.
    /// This is post-solve: runtime provenance is known. It is not pre-solve admission.
    pub fn encoding_profile(&self) -> AnalysisExecutionResult<ThermalResultEncodingProfile> {
        self.validate()?;
        let header = runmat_canonical_codec::profile_serialized(self, LIMITS)?;
        let mut result = ThermalResultEncodingProfile {
            contract_count: 1,
            encoded_bytes_upper_bound: header.encoded_bytes_upper_bound,
            largest_contract_materialization_bytes: header.materialization_resident_bytes_estimate,
        };
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
            }
        }
        Ok(result)
    }
}
