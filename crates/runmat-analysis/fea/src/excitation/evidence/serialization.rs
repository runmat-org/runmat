//! Shared retained/source-projection evidence wire authority.
use super::*;

#[derive(Serialize)]
#[serde(rename = "SpatialProjectionEvidence")]
struct EvidenceSerialization<'a> {
    excitation_id: &'a str,
    signal_id: &'a str,
    quantity: PhysicalQuantityKind,
    sample: &'a ProjectionSample,
    analysis_projection_digest: Digest,
    integration_rule: SpatialIntegrationRule,
    resolved_entity_count: usize,
    resolved_node_count: usize,
    reference_point_m: [f64; 3],
    balance: SpatialProjectionBalance,
    requirements: &'a ProjectionRequirement,
    accepted: bool,
}

impl Serialize for SpatialProjectionEvidence {
    fn serialize<S: serde::Serializer>(&self, serializer: S) -> Result<S::Ok, S::Error> {
        EvidenceSerialization {
            excitation_id: &self.excitation_id,
            signal_id: &self.signal_id,
            quantity: self.quantity,
            sample: &self.sample,
            analysis_projection_digest: self.analysis_projection_digest,
            integration_rule: self.integration_rule,
            resolved_entity_count: self.resolved_entity_count,
            resolved_node_count: self.resolved_node_count,
            reference_point_m: self.reference_point_m,
            balance: self.balance,
            requirements: &self.requirements,
            accepted: self.accepted,
        }
        .serialize(serializer)
    }
}

impl SpatialProjectionEvidence {
    /// Borrow the actual source-evidence wire with a finite scalar-integral shape. Shared codec
    /// profiling counts numeric positions at maximum width regardless of these placeholder values.
    /// This projection is not physical evidence and must never be published as a result. Caller
    /// owns validation of authored IDs, quantity, sample, integration rule and requirements.
    pub(crate) fn scalar_projection_serialization<'a>(
        excitation_id: &'a str,
        signal_id: &'a str,
        quantity: PhysicalQuantityKind,
        sample: &'a ProjectionSample,
        integration_rule: SpatialIntegrationRule,
        requirements: &'a ProjectionRequirement,
    ) -> impl Serialize + 'a {
        EvidenceSerialization {
            excitation_id,
            signal_id,
            quantity,
            sample,
            analysis_projection_digest: Digest::from_bytes([0; 32]),
            integration_rule,
            resolved_entity_count: 1,
            resolved_node_count: 1,
            reference_point_m: [0.0; 3],
            balance: SpatialProjectionBalance::ScalarIntegral {
                requested_integral: ProjectionScalar {
                    real: 0.0,
                    imaginary: 0.0,
                },
                projected_integral: ProjectionScalar {
                    real: 0.0,
                    imaginary: 0.0,
                },
                normalization_scale: 0.0,
                relative_error: 0.0,
            },
            requirements,
            accepted: true,
        }
    }
}

#[cfg(test)]
mod tests;
