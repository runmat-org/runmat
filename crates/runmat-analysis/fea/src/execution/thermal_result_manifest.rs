//! Metadata-only thermal result catalog over shared recovered-field storage partitions.
use super::state::CanonicalThermalStatePayload;
use super::{thermal_result::ThermalResultMetadata, *};
use crate::results::field::{
    reconstruct_recovered_field, RecoveredFieldArtifactEntry, RecoveredFieldDefinition,
    RecoveredFieldPartition, RecoveredFieldView, RECOVERED_FIELD_PARTITION_ENTITY_COUNT,
};
use crate::{FeaRunResult, FeaThermalRunResult, ThermalInitialStateEvidence};
use runmat_canonical_codec::{decode_contract, encode_contract, CanonicalLimits};
use runmat_execution::Digest;
use serde::{Deserialize, Serialize};

const PREFIX: &[u8] = b"runmat-thermal-solver-result-manifest/v1\0";
const LIMITS: CanonicalLimits = crate::contracts::canonical_digest::wire_bounded(CanonicalLimits::new(0, 0, 65_536, 64));

#[cfg(test)]
#[path = "thermal_result_manifest_tests.rs"]
mod tests;

mod admission_profile;
mod profile;
mod validation_profile;
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
    pub nonlinear_evidence: Vec<crate::contracts::ThermalNonlinearSampleEvidence>,
    pub residual_target: f64,
    pub energy_balances: Vec<crate::contracts::ThermalEnergyBalance>,
    #[serde(with = "runmat_types::physical::canonical_scalar")]
    pub reference_temperature_k: runmat_types::physical::Temperature,
    pub initial_state: ThermalInitialStateEvidence,
    /// Mesh nodes of finite-conductance thermal interfaces (empty without one).
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub interface_node_ids: Vec<u64>,
    /// Per sample: temperature, gradient, flux, storage rate, then the second-side interface
    /// temperature when `interface_node_ids` is not empty.
    pub fields: Vec<Vec<RecoveredFieldArtifactEntry>>,
}

fn invalid(error: impl std::fmt::Display) -> AnalysisExecutionError {
    AnalysisExecutionError::Invalid(format!("thermal result manifest: {error}"))
}

fn definition(index: usize, ordinal: usize) -> AnalysisExecutionResult<RecoveredFieldDefinition> {
    crate::physics::thermal::history_fields::snapshot_definition(index, ordinal).map_err(invalid)
}

impl ThermalSolverResultManifest {
    pub fn from_result(result: &FeaThermalRunResult) -> AnalysisExecutionResult<Self> {
        result.validate_complete()?;
        let histories = [
            &result.temperature_snapshots,
            &result.temperature_gradient_snapshots,
            &result.heat_flux_snapshots,
            &result.energy_storage_rate_snapshots,
            &result.interface_temperature_snapshots,
        ];
        let count = if result.interface_node_ids.is_empty() {
            4
        } else {
            5
        };
        let mut fields = Vec::with_capacity(result.time_points_s.len());
        for (ordinal, &time) in result.time_points_s.iter().enumerate() {
            let entry = |index: usize| -> AnalysisExecutionResult<_> {
                let definition = definition(index, ordinal)?;
                let sample = crate::physics::thermal::history_fields::sample(
                    result.time_points_s.len(),
                    time,
                );
                let field =
                    RecoveredFieldView::new(&definition, sample, &histories[index][ordinal])
                        .map_err(invalid)?;
                let partitions = field.canonical_partition_descriptors().map_err(invalid)?;
                Ok(RecoveredFieldArtifactEntry {
                    field: field.identity(),
                    definition,
                    sample,
                    partitions,
                })
            };
            fields.push(
                (0..count)
                    .map(entry)
                    .collect::<AnalysisExecutionResult<Vec<_>>>()?,
            );
        }
        let identity =
            result
                .metadata()
                .identity(super::thermal_result::identity::FieldIdentities {
                    samples: &fields,
                    interface_node_ids: &result.interface_node_ids,
                    identities: super::thermal_result::identity::manifest_entry_identities,
                })?;
        let run = &result.run;
        let manifest = Self {
            schema_version: 1,
            source_workload: result.source_workload,
            numerical_intent: result.numerical_intent,
            identity,
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
                structural_static_convergence: None,
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
            nonlinear_evidence: result.nonlinear_evidence.clone(),
            residual_target: result.residual_target,
            energy_balances: result.energy_balances.clone(),
            reference_temperature_k: runmat_types::physical::Temperature::new(
                result.reference_temperature_k.canonical_value(),
            ),
            initial_state: result.initial_state.clone(),
            interface_node_ids: result.interface_node_ids.clone(),
            fields,
        };
        manifest.validate()?;
        Ok(manifest)
    }

    pub(super) fn metadata(&self) -> ThermalResultMetadata<'_> {
        ThermalResultMetadata {
            source_workload: self.source_workload,
            numerical_intent: self.numerical_intent,
            run: &self.run,
            time_points_s: &self.time_points_s,
            residual_norms: &self.residual_norms,
            nonlinear_evidence: &self.nonlinear_evidence,
            residual_target: self.residual_target,
            energy_balances: &self.energy_balances,
            reference_temperature_k: runmat_types::physical::Temperature::new(
                self.reference_temperature_k.canonical_value(),
            ),
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
        let partition_count = self
            .fields
            .iter()
            .flatten()
            .try_fold(0usize, |count, entry| {
                count
                    .checked_add(entry.partitions.len())
                    .ok_or_else(|| invalid("partition count overflow"))
            })?;
        let mut identities = Vec::new();
        identities
            .try_reserve_exact(partition_count)
            .map_err(|_| invalid("partition identity scratch allocation failed"))?;
        let field_count = if self.interface_node_ids.is_empty() {
            4
        } else {
            5
        };
        for (ordinal, entries) in self.fields.iter().enumerate() {
            if entries.len() != field_count {
                return Err(invalid("thermal sample has the wrong field inventory"));
            }
            for (index, entry) in entries.iter().enumerate() {
                if entry.definition != definition(index, ordinal)?
                    || entry.sample
                        != crate::physics::thermal::history_fields::sample(
                            self.time_points_s.len(),
                            self.time_points_s[ordinal],
                        )
                {
                    return Err(invalid(
                        "field definition or sample differs from thermal history",
                    ));
                }
                let count = match index {
                    0 => self.ordered_node_ids.len(),
                    crate::physics::thermal::history_fields::INTERFACE => {
                        self.interface_node_ids.len()
                    }
                    _ => self.ordered_element_ids.len(),
                };
                let mut next = 0usize;
                for (position, partition) in entry.partitions.iter().enumerate() {
                    if partition.first_source_index != next
                        || partition.entity_count == 0
                        || partition.entity_count > RECOVERED_FIELD_PARTITION_ENTITY_COUNT
                        || (position + 1 < entry.partitions.len()
                            && partition.entity_count != RECOVERED_FIELD_PARTITION_ENTITY_COUNT)
                    {
                        return Err(invalid("invalid canonical field partition coverage"));
                    }
                    identities.push(partition.partition);
                    next = next
                        .checked_add(partition.entity_count)
                        .ok_or_else(|| invalid("partition count overflow"))?;
                }
                if next != count {
                    return Err(invalid("incomplete field partition coverage"));
                }
            }
        }
        identities.sort_unstable();
        if identities.windows(2).any(|pair| pair[0] == pair[1]) {
            return Err(invalid("invalid canonical field partition coverage"));
        }
        drop(identities);
        if self
            .metadata()
            .identity(super::thermal_result::identity::FieldIdentities {
                samples: &self.fields,
                interface_node_ids: &self.interface_node_ids,
                identities: super::thermal_result::identity::manifest_entry_identities,
            })?
            != self.identity
        {
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

    /// The terminal temperature and, with a finite-conductance interface, the terminal
    /// second-side interface temperature: the fields a published thermal state carries.
    pub fn terminal_state_fields(
        &self,
    ) -> AnalysisExecutionResult<Vec<&RecoveredFieldArtifactEntry>> {
        let last = self
            .fields
            .last()
            .ok_or_else(|| invalid("missing terminal temperature"))?;
        let mut fields = vec![&last[0]];
        if let Some(interface) = last.get(crate::physics::thermal::history_fields::INTERFACE) {
            fields.push(interface);
        }
        Ok(fields)
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

    /// `partitions` covers `terminal_state_fields` in order.
    pub fn publish_terminal(
        &self,
        request: &SemanticStatePublicationRequest,
        partitions: Vec<RecoveredFieldPartition>,
    ) -> AnalysisExecutionResult<CanonicalSemanticState> {
        self.validate_publication_request(request)?;
        let entries = self.terminal_state_fields()?;
        let expected = entries
            .iter()
            .map(|entry| entry.partitions.len())
            .sum::<usize>();
        if partitions.len() != expected
            || partitions
                .iter()
                .zip(entries.iter().flat_map(|entry| entry.partitions.iter()))
                .any(|(p, d)| {
                    p.identity != d.partition
                        || p.source_indices.first() != Some(&d.first_source_index)
                        || p.source_indices.len() != d.entity_count
                })
        {
            return Err(invalid(
                "terminal partition inventory differs from manifest",
            ));
        }
        let mut partitions = partitions.into_iter();
        let mut fields = Vec::with_capacity(entries.len());
        for (index, entry) in entries.iter().enumerate() {
            let taken = partitions.by_ref().take(entry.partitions.len()).collect();
            fields.push(
                reconstruct_recovered_field(
                    entry.field,
                    entry.definition.clone(),
                    entry.sample,
                    taken,
                    if index == 0 {
                        self.ordered_node_ids.len()
                    } else {
                        self.interface_node_ids.len()
                    },
                )
                .map_err(invalid)?
                .into_storage(),
            );
        }
        let mut fields = fields.into_iter();
        let temperature = fields
            .next()
            .ok_or_else(|| invalid("missing terminal temperature"))?;
        let interface_temperature = fields.next();
        CanonicalSemanticState::new(CanonicalSemanticStateInput {
            producer: request.producer,
            declaration: request.declaration.clone(),
            topology_digest: request.topology_digest,
            source_result: request.source_result,
            payload: CanonicalSemanticStatePayload::Thermal {
                state: Box::new(CanonicalThermalStatePayload {
                    temperature,
                    ordered_node_ids: self.ordered_node_ids.clone(),
                    reference_temperature_k: runmat_types::physical::Temperature::new(
                        self.reference_temperature_k.canonical_value(),
                    ),
                    sample_time_s: runmat_types::physical::Time::new(
                        *self.time_points_s.last().unwrap(),
                    ),
                    maximum_relative_linear_residual: self
                        .residual_norms
                        .iter()
                        .copied()
                        .fold(0.0_f64, f64::max),
                    interface_node_ids: self.interface_node_ids.clone(),
                    interface_temperature,
                }),
            },
        })
    }
}
