//! Native complex fields from an authenticated acoustic result; no second solve.
use super::*;
use crate::{
    execution::{
        FieldRecoveryCost, FieldRecoveryPlan, FieldRecoveryRequest, FieldRecoveryRequirement,
        FieldRecoverySource, FieldTopologyIdentity,
    },
    results::field::{
        RecoveredField, RecoveredFieldArtifact, RecoveredFieldDefinition,
        RecoveredFieldEntityCounts, RecoveredFieldEntityDomain,
        RecoveredFieldNumericRepresentation, RecoveredFieldSample,
    },
};
use runmat_analysis_core::{
    AnalysisField, AnalysisFieldValues, FieldLocation, ObservationQuantity, ObservationValueShape,
    RecoveryPolicy,
};

#[cfg(test)]
mod tests;

impl AcousticHarmonicSolverResult {
    /// Catalog and partition encoding profiles from native shape, without copying values.
    /// This is sizing authority only, not a published field or numerical receipt.
    pub fn native_field_encoding_profiles(
        &self,
    ) -> AnalysisExecutionResult<(
        runmat_canonical_codec::SerializedCanonicalProfile,
        runmat_canonical_codec::SerializedCanonicalProfile,
    )> {
        use crate::results::field::{
            RecoveredFieldArtifactManifest, RecoveredFieldPartition,
            RECOVERED_FIELD_PARTITION_ENTITY_COUNT,
        };
        let request = self.native_field_request_from_result()?;
        let sample = RecoveredFieldSample::FrequencyHertz(self.frequency().hertz);
        let definitions = [
            native_acoustic_field_definition("acoustic.pressure", &request.plan.requirements[0])?,
            native_acoustic_field_definition(
                "acoustic.particle_velocity",
                &request.plan.requirements[1],
            )?,
        ];
        let counts = [
            self.ordered_node_ids().len(),
            self.ordered_element_ids().len(),
        ];
        let catalog = RecoveredFieldArtifactManifest::profile_sampled_shape(
            &request,
            &definitions,
            &counts,
            sample,
        )
        .map_err(invalid)?;
        let mut complete = catalog;
        for (definition, count) in definitions.iter().zip(counts) {
            let mut remaining = count;
            while remaining != 0 {
                let entities = remaining.min(RECOVERED_FIELD_PARTITION_ENTITY_COUNT);
                let partition =
                    RecoveredFieldPartition::profile_shape(definition, sample, entities)
                        .map_err(invalid)?;
                complete.encoded_bytes_upper_bound = complete
                    .encoded_bytes_upper_bound
                    .checked_add(partition.encoded_bytes_upper_bound)
                    .ok_or_else(|| invalid("native field encoding profile overflow"))?;
                complete.materialization_resident_bytes_estimate = complete
                    .materialization_resident_bytes_estimate
                    .checked_add(partition.materialization_resident_bytes_estimate)
                    .ok_or_else(|| invalid("native field materialization profile overflow"))?;
                remaining -= entities;
            }
        }
        Ok((catalog, complete))
    }

    /// Exact publication request for native nodal pressure and element-average particle velocity.
    /// The producer, source result and storage topology are authenticated before planning.
    pub fn native_field_request(
        &self,
        workload: &AcousticHarmonicWorkload,
        mesh: &SolverMeshArtifact,
    ) -> AnalysisExecutionResult<FieldRecoveryRequest> {
        self.validate_for(workload, mesh)?;
        self.native_field_request_from_result()
    }

    /// Detached recovery authority, authenticated by the completed result's retained lineage.
    pub fn native_field_request_from_result(
        &self,
    ) -> AnalysisExecutionResult<FieldRecoveryRequest> {
        self.validate()?;
        FieldRecoveryRequest::new(
            self.producer(),
            FieldRecoverySource::SolverResult(self.source_result_identity()?),
            FieldTopologyIdentity::VolumeMesh {
                mesh_digest: self.mesh_digest(),
            },
            native_acoustic_field_plan(
                self.ordered_node_ids().len(),
                self.ordered_element_ids().len(),
            )?,
        )
    }

    /// Copy already recovered native values into the shared, partitionable field contract.
    /// Velocity is the kernel's quadrature-volume average per element, not a nodal value or
    /// pointwise integration-point sample. Intensity and energy remain distinct diagnostics.
    pub fn recover_native_fields(
        &self,
        workload: &AcousticHarmonicWorkload,
        mesh: &SolverMeshArtifact,
        request: &FieldRecoveryRequest,
    ) -> AnalysisExecutionResult<RecoveredFieldArtifact> {
        self.validate_for(workload, mesh)?;
        self.recover_native_fields_from_result(request)
    }

    pub fn recover_native_fields_from_result(
        &self,
        request: &FieldRecoveryRequest,
    ) -> AnalysisExecutionResult<RecoveredFieldArtifact> {
        request.validate()?;
        let expected = self.native_field_request_from_result()?;
        if request != &expected {
            return Err(AnalysisExecutionError::Identity(
                "acoustic field request differs from its exact producer, result, topology or plan",
            ));
        }
        let sample = RecoveredFieldSample::FrequencyHertz(self.frequency().hertz);
        let fields = vec![
            native_field(
                "acoustic.pressure",
                &request.plan.requirements[0],
                sample,
                vec![self.pressure_pa().len(), 2],
                self.pressure_pa()
                    .iter()
                    .flat_map(|p| [p.real, p.imaginary])
                    .collect(),
            )?,
            native_field(
                "acoustic.particle_velocity",
                &request.plan.requirements[1],
                sample,
                vec![self.velocity_m_per_s().len(), 3, 2],
                self.velocity_m_per_s()
                    .iter()
                    .flat_map(|v| (0..3).flat_map(move |axis| [v.real[axis], v.imaginary[axis]]))
                    .collect(),
            )?,
        ];
        request
            .plan
            .validate_definitions(fields.iter().map(|field| field.definition().clone()))?;
        RecoveredFieldArtifact::new(
            request.identity(),
            request.source,
            request.topology.discretization_digest(),
            native_acoustic_entity_counts(self),
            fields,
        )
        .map_err(invalid)
    }
}

/// Shared native publication inventory, also used to validate cold acoustic state payloads.
pub(in crate::execution) fn native_acoustic_field_plan(
    nodes: usize,
    elements: usize,
) -> AnalysisExecutionResult<FieldRecoveryPlan> {
    let plan = FieldRecoveryPlan {
        requirements: vec![
            native_requirement(
                ObservationQuantity::AcousticPressure,
                FieldLocation::Node,
                RecoveredFieldEntityDomain::Nodes,
                ObservationValueShape::Scalar,
                nodes,
            )?,
            native_requirement(
                ObservationQuantity::AcousticParticleVelocity,
                FieldLocation::Element,
                RecoveredFieldEntityDomain::VolumeElements,
                ObservationValueShape::Vector3,
                elements,
            )?,
        ],
    };
    plan.validate()?;
    Ok(plan)
}

fn native_requirement(
    quantity: ObservationQuantity,
    location: FieldLocation,
    entity_domain: RecoveredFieldEntityDomain,
    value_shape: ObservationValueShape,
    count: usize,
) -> AnalysisExecutionResult<FieldRecoveryRequirement> {
    let count = u64::try_from(count).map_err(invalid)?;
    Ok(FieldRecoveryRequirement {
        quantity,
        source_location: location,
        output_location: location,
        recovery: RecoveryPolicy::Native,
        value_shape,
        entity_domain,
        cost: FieldRecoveryCost {
            source_entity_count: count,
            output_entity_count: count,
            topology_incidence_count: 0,
            arithmetic_operation_count: count,
        },
    })
}

pub(in crate::execution) fn native_acoustic_entity_counts(
    result: &AcousticHarmonicSolverResult,
) -> RecoveredFieldEntityCounts {
    RecoveredFieldEntityCounts {
        oriented_faces: 0,
        nodes: result.ordered_node_ids().len(),
        volume_elements: result.ordered_element_ids().len(),
        volume_integration_points: 0,
        oriented_edges: 0,
        beam_elements: 0,
        shell_elements: 0,
        connectors: 0,
        constraint_groups: 0,
    }
}

pub(in crate::execution) fn native_acoustic_field_definition(
    field_id: &str,
    requirement: &FieldRecoveryRequirement,
) -> AnalysisExecutionResult<RecoveredFieldDefinition> {
    RecoveredFieldDefinition::new(
        field_id,
        requirement.quantity,
        requirement.output_location,
        requirement.entity_domain,
        requirement.value_shape.clone(),
        RecoveredFieldNumericRepresentation::ComplexInterleaved,
    )
    .map_err(invalid)
}

fn native_field(
    field_id: &str,
    requirement: &FieldRecoveryRequirement,
    sample: RecoveredFieldSample,
    shape: Vec<usize>,
    values: Vec<f64>,
) -> AnalysisExecutionResult<RecoveredField> {
    let definition = native_acoustic_field_definition(field_id, requirement)?;
    RecoveredField::new(
        definition,
        sample,
        AnalysisField {
            field_id: field_id.into(),
            shape,
            values: AnalysisFieldValues::HostF64(values),
        },
    )
    .map_err(invalid)
}
