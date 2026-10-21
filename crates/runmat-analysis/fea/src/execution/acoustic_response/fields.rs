//! Native real nodal pressure per retained sample, through the shared recovered-field contract.
use super::*;
use crate::{
    execution::{
        FieldRecoveryCost, FieldRecoveryPlan, FieldRecoveryRequest, FieldRecoveryRequirement,
        FieldRecoverySource, FieldTopologyIdentity,
    },
    results::field::{
        RecoveredField, RecoveredFieldArtifact, RecoveredFieldDefinition,
        RecoveredFieldEntityCounts, RecoveredFieldEntityDomain,
        RecoveredFieldNumericRepresentation,
    },
};
use runmat_analysis_core::{
    AnalysisField, AnalysisFieldValues, FieldLocation, ObservationQuantity, ObservationValueShape,
    RecoveryPolicy,
};

pub const ACOUSTIC_PRESSURE_FIELD_ID: &str = "acoustic.pressure";

fn pressure_plan(nodes: usize) -> AnalysisExecutionResult<FieldRecoveryPlan> {
    let count = u64::try_from(nodes).map_err(invalid)?;
    let plan = FieldRecoveryPlan {
        requirements: vec![FieldRecoveryRequirement {
            support: Default::default(),
            quantity: ObservationQuantity::AcousticPressure,
            source_location: FieldLocation::Node,
            output_location: FieldLocation::Node,
            recovery: RecoveryPolicy::Native,
            value_shape: ObservationValueShape::Scalar,
            entity_domain: RecoveredFieldEntityDomain::Nodes,
            cost: FieldRecoveryCost {
                source_entity_count: count,
                output_entity_count: count,
                topology_incidence_count: 0,
                arithmetic_operation_count: count,
            },
        }],
    };
    plan.validate()?;
    Ok(plan)
}

fn pressure_definition() -> AnalysisExecutionResult<RecoveredFieldDefinition> {
    RecoveredFieldDefinition::new(
        ACOUSTIC_PRESSURE_FIELD_ID,
        ObservationQuantity::AcousticPressure,
        FieldLocation::Node,
        RecoveredFieldEntityDomain::Nodes,
        ObservationValueShape::Scalar,
        RecoveredFieldNumericRepresentation::Real,
    )
    .map_err(invalid)
}

impl AcousticResponseSolverResult {
    /// Exact publication request for every retained pressure sample, bound to this result.
    pub fn native_field_request_from_result(&self) -> AnalysisExecutionResult<FieldRecoveryRequest> {
        self.validate()?;
        FieldRecoveryRequest::new(
            self.producer(),
            FieldRecoverySource::SolverResult(self.source_result_identity()?),
            FieldTopologyIdentity::VolumeMesh {
                mesh_digest: self.mesh_digest(),
            },
            pressure_plan(self.ordered_node_ids().len())?,
        )
    }

    /// Copy the retained pressure samples into the shared, partitionable field contract.
    pub fn recover_native_fields_from_result(
        &self,
        request: &FieldRecoveryRequest,
    ) -> AnalysisExecutionResult<RecoveredFieldArtifact> {
        request.validate()?;
        if request != &self.native_field_request_from_result()? {
            return Err(AnalysisExecutionError::Identity(
                "acoustic response field request differs from its exact producer, result or topology",
            ));
        }
        let definition = pressure_definition()?;
        let nodes = self.ordered_node_ids().len();
        let fields = self
            .samples()
            .iter()
            .zip(self.pressure_pa())
            .map(|(sample, values)| {
                RecoveredField::new(
                    definition.clone(),
                    *sample,
                    AnalysisField {
                        field_id: ACOUSTIC_PRESSURE_FIELD_ID.into(),
                        shape: vec![nodes],
                        values: AnalysisFieldValues::HostF64(values.clone()),
                    },
                )
                .map_err(invalid)
            })
            .collect::<AnalysisExecutionResult<Vec<_>>>()?;
        request
            .plan
            .validate_definitions(std::iter::once(definition))?;
        RecoveredFieldArtifact::new(
            request.identity(),
            request.source,
            request.topology.discretization_digest(),
            RecoveredFieldEntityCounts {
                oriented_faces: 0,
                nodes,
                volume_elements: self.ordered_element_ids().len(),
                volume_integration_points: 0,
                oriented_edges: 0,
                beam_elements: 0,
                shell_elements: 0,
                connectors: 0,
                constraint_groups: 0,
            },
            fields,
        )
        .map_err(invalid)
    }

    /// Run-level scalar series published beside the pressure samples: natural frequencies for
    /// cavity modes (`modal.frequency_hz`, the structural modal public shape), or time points and
    /// acoustic energy for a transient.
    pub fn summary_fields(&self) -> Vec<AnalysisField> {
        match self.diagnostics() {
            AcousticResponseDiagnostics::Modal { frequencies_hz, .. } => vec![
                AnalysisField::host_f64(
                    crate::contracts::FEA_FIELD_MODAL_FREQUENCY_HZ,
                    vec![frequencies_hz.len()],
                    frequencies_hz.clone(),
                ),
                AnalysisField::host_f64(
                    crate::contracts::FEA_FIELD_MODAL_EIGENVALUE,
                    vec![frequencies_hz.len()],
                    frequencies_hz
                        .iter()
                        .map(|f| (std::f64::consts::TAU * f).powi(2))
                        .collect(),
                ),
            ],
            AcousticResponseDiagnostics::Transient {
                time_points_s,
                acoustic_energy_j,
                boundary_absorbed_energy_j,
                ..
            } => {
                // Energy evidence covers every accepted time point, not only retained fields.
                let times = time_points_s.clone();
                vec![
                    AnalysisField::host_f64("acoustic.time_s", vec![times.len()], times),
                    AnalysisField::host_f64(
                        "acoustic.energy_j",
                        vec![acoustic_energy_j.len()],
                        acoustic_energy_j.clone(),
                    ),
                    AnalysisField::host_f64(
                        "acoustic.boundary_absorbed_energy_j",
                        vec![boundary_absorbed_energy_j.len()],
                        boundary_absorbed_energy_j.clone(),
                    ),
                ]
            }
        }
    }

    /// The complete state payload: all mode shapes with their frequencies, or the final
    /// pressure field of the transient.
    pub fn state_fields(&self) -> Vec<AnalysisField> {
        let nodes = self.ordered_node_ids().len();
        match self.diagnostics() {
            AcousticResponseDiagnostics::Modal { frequencies_hz, .. } => vec![
                AnalysisField::host_f64(
                    crate::contracts::FEA_FIELD_MODAL_FREQUENCY_HZ,
                    vec![frequencies_hz.len()],
                    frequencies_hz.clone(),
                ),
                AnalysisField::host_f64(
                    "acoustic.mode_pressure",
                    vec![self.pressure_pa().len(), nodes],
                    self.pressure_pa().iter().flatten().copied().collect(),
                ),
            ],
            AcousticResponseDiagnostics::Transient { .. } => vec![AnalysisField::host_f64(
                ACOUSTIC_PRESSURE_FIELD_ID,
                vec![nodes],
                self.pressure_pa()
                    .last()
                    .cloned()
                    .unwrap_or_else(|| vec![0.0; nodes]),
            )],
        }
    }
}
