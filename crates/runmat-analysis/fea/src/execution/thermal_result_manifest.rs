//! Metadata-only thermal result catalog over shared recovered-field storage partitions.
use super::state::CanonicalThermalStatePayload;
use super::{thermal_result::ThermalResultMetadata, *};
use crate::results::field::{
    canonical_field_partitions, reconstruct_recovered_field, RecoveredField,
    RecoveredFieldArtifactEntry, RecoveredFieldDefinition, RecoveredFieldPartition,
    RecoveredFieldPartitionDescriptor, RecoveredFieldSample,
    RECOVERED_FIELD_PARTITION_ENTITY_COUNT,
};
use crate::{FeaRunResult, FeaThermalRunResult, ThermalInitialStateEvidence};
use runmat_canonical_codec::{decode_contract, encode_contract, CanonicalLimits};
use runmat_execution::Digest;
use serde::{Deserialize, Serialize};

const PREFIX: &[u8] = b"runmat-thermal-solver-result-manifest/v1\0";
const LIMITS: CanonicalLimits = CanonicalLimits::new(64 * 1024 * 1024, 4_000_000, 65_536, 64);

#[cfg(test)]
#[path = "thermal_result_manifest_tests.rs"]
mod tests;

mod profile;
pub use profile::ThermalResultEncodingProfile;

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct ThermalSolverResultManifest {
    pub schema_version: u16,
    pub source_workload: AnalysisWorkloadIdentity,
    pub numerical_intent: ThermalNumericalIntentIdentity,
    identity: SolverResultIdentity,
    pub run: FeaRunResult,
    pub producer: StepIdentity,
    pub topology_digest: Digest,
    pub ordered_node_ids: Vec<u64>,
    pub ordered_element_ids: Vec<u64>,
    pub time_points_s: Vec<f64>,
    pub residual_norms: Vec<f64>,
    pub residual_target: f64,
    pub energy_balances: Vec<crate::contracts::ThermalEnergyBalance>,
    pub reference_temperature_k: f64,
    pub initial_state: ThermalInitialStateEvidence,
    pub fields: Vec<[RecoveredFieldArtifactEntry; 4]>,
}

fn invalid(error: impl std::fmt::Display) -> AnalysisExecutionError {
    AnalysisExecutionError::Invalid(format!("thermal result manifest: {error}"))
}

fn definition(index: usize, ordinal: usize) -> AnalysisExecutionResult<RecoveredFieldDefinition> {
    let definitions = crate::physics::thermal::history_fields::definitions().map_err(invalid)?;
    let template = &definitions[index];
    let ids = [
        crate::contracts::fea_thermal_temperature_field_id,
        crate::contracts::fea_thermal_temperature_gradient_field_id,
        crate::contracts::fea_thermal_heat_flux_field_id,
        crate::contracts::fea_thermal_energy_storage_rate_field_id,
    ];
    RecoveredFieldDefinition::new_derived(
        ids[index](ordinal),
        template.quantity(),
        template.location(),
        template.entity_domain(),
        template.value_shape().clone(),
        template.numeric_representation(),
        template.derivation(),
    )
    .map_err(invalid)
}

impl ThermalSolverResultManifest {
    pub fn from_result(result: &FeaThermalRunResult) -> AnalysisExecutionResult<Self> {
        result.validate_complete()?;
        let histories = [
            &result.temperature_snapshots,
            &result.temperature_gradient_snapshots,
            &result.heat_flux_snapshots,
            &result.energy_storage_rate_snapshots,
        ];
        let mut fields = Vec::with_capacity(result.time_points_s.len());
        for (ordinal, &time) in result.time_points_s.iter().enumerate() {
            let entry = |index: usize| -> AnalysisExecutionResult<_> {
                let definition = definition(index, ordinal)?;
                let sample = RecoveredFieldSample::TimeSeconds(time);
                let field = RecoveredField::new(
                    definition.clone(),
                    sample,
                    histories[index][ordinal].clone(),
                )
                .map_err(invalid)?;
                let partitions = canonical_field_partitions(&field)
                    .map_err(invalid)?
                    .into_iter()
                    .map(|partition| RecoveredFieldPartitionDescriptor {
                        partition: partition.identity,
                        first_source_index: partition.source_indices[0],
                        entity_count: partition.source_indices.len(),
                    })
                    .collect();
                Ok(RecoveredFieldArtifactEntry {
                    field: field.identity(),
                    definition,
                    sample,
                    partitions,
                })
            };
            fields.push([entry(0)?, entry(1)?, entry(2)?, entry(3)?]);
        }
        let run = &result.run;
        let manifest = Self {
            schema_version: 1,
            source_workload: result.source_workload,
            numerical_intent: result.numerical_intent,
            identity: result.source_result_identity()?,
            run: FeaRunResult {
                identity: run.identity.clone(),
                requested_backend: run.requested_backend,
                execution_backend: run.execution_backend,
                solver_backend: run.solver_backend,
                solver_device_apply_k_ratio: run.solver_device_apply_k_ratio,
                solver_method: run.solver_method,
                preconditioner: run.preconditioner,
                solver_host_sync_count: run.solver_host_sync_count,
                accelerator_evidence: run.accelerator_evidence.clone(),
                diagnostics: run.diagnostics.clone(),
                spatial_projection_evidence: run.spatial_projection_evidence.clone(),
                structural_static_balance: None,
                structural_constraint_result: None,
                structural_connector_result: None,
                observations: Vec::new(),
                observation_histories: Vec::new(),
                series_reductions: Vec::new(),
                fields: Vec::new(),
            },
            producer: result.producer,
            topology_digest: result.topology_digest,
            ordered_node_ids: result.ordered_node_ids.clone(),
            ordered_element_ids: result.ordered_element_ids.clone(),
            time_points_s: result.time_points_s.clone(),
            residual_norms: result.residual_norms.clone(),
            residual_target: result.residual_target,
            energy_balances: result.energy_balances.clone(),
            reference_temperature_k: result.reference_temperature_k,
            initial_state: result.initial_state.clone(),
            fields,
        };
        manifest.validate()?;
        Ok(manifest)
    }

    fn metadata(&self) -> ThermalResultMetadata<'_> {
        ThermalResultMetadata {
            source_workload: self.source_workload,
            numerical_intent: self.numerical_intent,
            run: &self.run,
            time_points_s: &self.time_points_s,
            residual_norms: &self.residual_norms,
            residual_target: self.residual_target,
            energy_balances: &self.energy_balances,
            reference_temperature_k: self.reference_temperature_k,
            topology_digest: self.topology_digest,
            producer: self.producer,
            ordered_node_ids: &self.ordered_node_ids,
            ordered_element_ids: &self.ordered_element_ids,
            initial_state: &self.initial_state,
        }
    }

    pub fn validate(&self) -> AnalysisExecutionResult<()> {
        self.metadata().validate()?;
        if self.schema_version != 1
            || self.fields.len() != self.time_points_s.len()
            || !self.run.fields.is_empty()
            || !self.run.observations.is_empty()
            || !self.run.observation_histories.is_empty()
            || !self.run.series_reductions.is_empty()
            || self.run.structural_static_balance.is_some()
            || self.run.structural_constraint_result.is_some()
            || self.run.structural_connector_result.is_some()
        {
            return Err(invalid(
                "invalid schema, history inventory, or embedded numerical payload",
            ));
        }
        let mut identities = std::collections::BTreeSet::new();
        for (ordinal, entries) in self.fields.iter().enumerate() {
            for (index, entry) in entries.iter().enumerate() {
                if entry.definition != definition(index, ordinal)?
                    || entry.sample
                        != RecoveredFieldSample::TimeSeconds(self.time_points_s[ordinal])
                {
                    return Err(invalid(
                        "field definition or sample differs from thermal history",
                    ));
                }
                let count = if index == 0 {
                    self.ordered_node_ids.len()
                } else {
                    self.ordered_element_ids.len()
                };
                let mut next = 0usize;
                for (position, partition) in entry.partitions.iter().enumerate() {
                    if partition.first_source_index != next
                        || partition.entity_count == 0
                        || partition.entity_count > RECOVERED_FIELD_PARTITION_ENTITY_COUNT
                        || (position + 1 < entry.partitions.len()
                            && partition.entity_count != RECOVERED_FIELD_PARTITION_ENTITY_COUNT)
                        || !identities.insert(partition.partition)
                    {
                        return Err(invalid("invalid canonical field partition coverage"));
                    }
                    next = next
                        .checked_add(partition.entity_count)
                        .ok_or_else(|| invalid("partition count overflow"))?;
                }
                if next != count {
                    return Err(invalid("incomplete field partition coverage"));
                }
            }
        }
        let fields = self
            .fields
            .iter()
            .map(|entries| entries.each_ref().map(|entry| entry.field))
            .collect::<Vec<_>>();
        if self.metadata().identity(&fields)? != self.identity {
            return Err(invalid("numerical identity does not converge"));
        }
        Ok(())
    }

    pub const fn identity(&self) -> SolverResultIdentity {
        self.identity
    }
    pub fn terminal_temperature(&self) -> AnalysisExecutionResult<&RecoveredFieldArtifactEntry> {
        self.fields
            .last()
            .map(|fields| &fields[0])
            .ok_or_else(|| invalid("missing terminal temperature"))
    }
    pub fn canonical_bytes(&self) -> AnalysisExecutionResult<Vec<u8>> {
        self.validate()?;
        Ok(encode_contract(
            PREFIX,
            "thermal-solver-result-manifest",
            self,
            LIMITS,
        )?)
    }
    pub fn from_canonical_bytes(bytes: &[u8]) -> AnalysisExecutionResult<Self> {
        let manifest: Self =
            decode_contract(PREFIX, "thermal-solver-result-manifest", bytes, LIMITS)?;
        manifest.validate()?;
        Ok(manifest)
    }

    /// Validate publication bindings before requesting numerical partitions.
    pub fn validate_publication_request(
        &self,
        request: &SemanticStatePublicationRequest,
    ) -> AnalysisExecutionResult<()> {
        self.validate()?;
        request.validate()?;
        if request.source != SemanticStatePublicationSource::ThermalPrimaryField
            || request.producer != self.producer
            || request.topology_digest != self.topology_digest
            || request.source_result != self.identity
        {
            return Err(invalid(
                "publication request differs from source, producer, or topology",
            ));
        }
        Ok(())
    }

    pub fn publish_terminal(
        &self,
        request: &SemanticStatePublicationRequest,
        partitions: Vec<RecoveredFieldPartition>,
    ) -> AnalysisExecutionResult<CanonicalSemanticState> {
        self.validate_publication_request(request)?;
        let entry = self.terminal_temperature()?;
        if partitions.len() != entry.partitions.len()
            || partitions.iter().zip(&entry.partitions).any(|(p, d)| {
                p.identity != d.partition
                    || p.source_indices.first() != Some(&d.first_source_index)
                    || p.source_indices.len() != d.entity_count
            })
        {
            return Err(invalid(
                "terminal partition inventory differs from manifest",
            ));
        }
        let temperature = reconstruct_recovered_field(
            entry.field,
            entry.definition.clone(),
            entry.sample,
            partitions,
            self.ordered_node_ids.len(),
        )
        .map_err(invalid)?
        .into_storage();
        CanonicalSemanticState::new(CanonicalSemanticStateInput {
            producer: request.producer,
            declaration: request.declaration.clone(),
            topology_digest: request.topology_digest,
            source_result: request.source_result,
            payload: CanonicalSemanticStatePayload::Thermal {
                state: Box::new(CanonicalThermalStatePayload {
                    temperature,
                    ordered_node_ids: self.ordered_node_ids.clone(),
                    reference_temperature_k: self.reference_temperature_k,
                    sample_time_s: *self.time_points_s.last().unwrap(),
                    maximum_relative_linear_residual: self
                        .residual_norms
                        .iter()
                        .copied()
                        .fold(0.0_f64, f64::max),
                }),
            },
        })
    }
}
