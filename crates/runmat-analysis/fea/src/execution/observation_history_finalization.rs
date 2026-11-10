//! Validated finalization of source-bound histories without re-solving or recovering fields.
use runmat_analysis_core::{ObservationId, SeriesReduction};
use runmat_canonical_codec::{decode_contract, encode_contract, CanonicalLimits};
use runmat_execution::Digest;
use runmat_meshing_core::SolverMeshArtifact;
use serde::{Deserialize, Serialize};

use super::{AnalysisExecutionError, AnalysisExecutionResult, StepIdentity, ThermalWorkload};
use crate::results::observation::{
    reduce_observation_history, resolved_observation_accepts_sample, BoundedObservationHistory,
    ObservationExecutionPolicy, ObservationFieldCatalog, ObservationFieldCatalogSource,
    ObservationHistory, ObservationHistoryCheckpoint, ObservationHistoryCheckpointIdentity,
    ObservationHistoryStream, ResolvedObservationPlan, SeriesReductionResult,
};

const PREFIX: &[u8] = b"runmat-observation-history-finalization/v1\0";
const LIMITS: CanonicalLimits = crate::contracts::canonical_digest::wire_bounded(CanonicalLimits::new(0, 0, 65_536, 64));

#[derive(Clone, Copy)]
pub enum ObservationHistorySourceContext<'a> {
    MaxwellHarmonic {
        workload: &'a super::MaxwellHarmonicWorkload,
        mesh: &'a SolverMeshArtifact,
    },
    /// Native node/element fields on the canonical solver mesh of any family. The producing
    /// step and identity are carried by the volume-mesh catalog itself.
    VolumeMesh { mesh: &'a SolverMeshArtifact },
    Magnetostatic {
        workload: &'a super::MagnetostaticWorkload,
        mesh: &'a SolverMeshArtifact,
    },
    Structural {
        source: super::StructuralObservationSource<'a>,
        mesh: Option<&'a SolverMeshArtifact>,
    },
    Thermal {
        workload: &'a ThermalWorkload,
        mesh: &'a SolverMeshArtifact,
    },
}

impl ObservationHistorySourceContext<'_> {
    pub const fn observation_execution(self) -> ObservationExecutionPolicy {
        match self {
            Self::Magnetostatic { .. } | Self::MaxwellHarmonic { .. } | Self::VolumeMesh { .. } => {
                ObservationExecutionPolicy::serial()
            }
            Self::Structural { source, .. } => source.observation_execution(),
            Self::Thermal { .. } => ObservationExecutionPolicy::serial(),
        }
    }

    pub fn validate(self) -> AnalysisExecutionResult<()> {
        match self {
            Self::Magnetostatic { workload, .. } => workload.validate(),
            Self::MaxwellHarmonic { workload, .. } => workload.validate(),
            Self::VolumeMesh { mesh } => {
                runmat_meshing_core::CanonicalMeshingContract::validate_canonical(mesh)
                    .map_err(|error| AnalysisExecutionError::Invalid(error.to_string()))
            }
            Self::Structural { source, .. } => source.validate(),
            Self::Thermal { workload, .. } => workload.validate(),
        }
    }
}

#[derive(Clone, Copy, Debug, Eq, PartialEq, Serialize, Deserialize)]
#[serde(transparent)]
pub struct ObservationHistoryFinalizationRequestIdentity(Digest);
impl ObservationHistoryFinalizationRequestIdentity {
    pub const fn digest(self) -> Digest {
        self.0
    }
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields, try_from = "RequestWire")]
pub struct ObservationHistoryFinalizationRequest {
    pub schema_version: u16,
    identity: ObservationHistoryFinalizationRequestIdentity,
    pub plan: ResolvedObservationPlan,
    pub producer: StepIdentity,
    pub observation_id: ObservationId,
    pub terminal_checkpoint: ObservationHistoryCheckpointIdentity,
    pub source_catalog: Digest,
    pub series_reduction: Option<SeriesReduction>,
}

#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct RequestWire {
    schema_version: u16,
    identity: ObservationHistoryFinalizationRequestIdentity,
    plan: ResolvedObservationPlan,
    producer: StepIdentity,
    observation_id: ObservationId,
    terminal_checkpoint: ObservationHistoryCheckpointIdentity,
    source_catalog: Digest,
    series_reduction: Option<SeriesReduction>,
}

impl TryFrom<RequestWire> for ObservationHistoryFinalizationRequest {
    type Error = AnalysisExecutionError;
    fn try_from(wire: RequestWire) -> Result<Self, Self::Error> {
        let request = Self {
            schema_version: wire.schema_version,
            identity: wire.identity,
            plan: wire.plan,
            producer: wire.producer,
            observation_id: wire.observation_id,
            terminal_checkpoint: wire.terminal_checkpoint,
            source_catalog: wire.source_catalog,
            series_reduction: wire.series_reduction,
        };
        request.validate()?;
        Ok(request)
    }
}

pub struct FinalizedObservationHistory {
    pub history: ObservationHistory,
    pub series_reduction: Option<SeriesReductionResult>,
}

fn invalid(error: impl std::fmt::Display) -> AnalysisExecutionError {
    AnalysisExecutionError::Invalid(format!("observation history finalization: {error}"))
}

impl ObservationHistoryFinalizationRequest {
    pub fn new(
        plan: ResolvedObservationPlan,
        producer: StepIdentity,
        observation_id: ObservationId,
        checkpoint: &ObservationHistoryCheckpoint,
        catalog: &ObservationFieldCatalog,
        context: ObservationHistorySourceContext<'_>,
    ) -> AnalysisExecutionResult<Self> {
        let series_reduction = authored_series(&plan, &observation_id)?;
        let mut request = Self {
            schema_version: 1,
            identity: ObservationHistoryFinalizationRequestIdentity(Digest::sha256([])),
            plan,
            producer,
            observation_id,
            terminal_checkpoint: checkpoint.manifest().identity,
            source_catalog: catalog.identity(),
            series_reduction,
        };
        request.identity = request.derive_identity()?;
        request.validate_inputs(checkpoint, catalog, context)?;
        Ok(request)
    }

    pub const fn identity(&self) -> ObservationHistoryFinalizationRequestIdentity {
        self.identity
    }

    pub fn validate(&self) -> AnalysisExecutionResult<()> {
        if self.schema_version != 1
            || self.identity != self.derive_identity()?
            || self.series_reduction != authored_series(&self.plan, &self.observation_id)?
        {
            return Err(invalid(
                "request schema, identity, or authored series reduction differs",
            ));
        }
        Ok(())
    }

    pub fn validate_inputs(
        &self,
        checkpoint: &ObservationHistoryCheckpoint,
        catalog: &ObservationFieldCatalog,
        context: ObservationHistorySourceContext<'_>,
    ) -> AnalysisExecutionResult<()> {
        self.validate()?;
        checkpoint.validate().map_err(invalid)?;
        catalog.validate()?;
        if checkpoint.manifest().identity != self.terminal_checkpoint
            || catalog.identity() != self.source_catalog
        {
            return Err(invalid("terminal checkpoint or source catalog substituted"));
        }
        // The current shared checkpoint hashes evicted samples but does not bind their source
        // catalog. Never treat an opaque prefix hash as proof of source provenance. Supporting
        // eviction requires a source-bound checkpoint lineage contract, not relaxed validation.
        if checkpoint.manifest().observed_sample_count != checkpoint.samples().len() as u64 {
            return Err(invalid(
                "evicted history samples require source-bound checkpoint lineage",
            ));
        }
        let observation = self
            .plan
            .observations
            .iter()
            .find(|observation| observation.spec.observation_id == self.observation_id)
            .ok_or_else(|| invalid("observation absent"))?;
        let mut expected = Vec::new();
        match (catalog.source(), context) {
            (
                ObservationFieldCatalogSource::Structural {
                    recovery_request,
                    manifest,
                },
                ObservationHistorySourceContext::Structural { source, mesh },
            ) => {
                source.validate()?;
                if source.procedure_step()?.step_id != self.plan.step_id {
                    return Err(invalid("structural source workload selects another step"));
                }
                catalog.validate_structural_plan(&self.plan, source.model(), mesh)?;
                if recovery_request.producer != self.producer
                    || manifest.topology_digest != self.plan.topology_digest
                {
                    return Err(invalid("structural producer or topology differs"));
                }
                for entry in &manifest.fields {
                    if entry.definition.identity() == observation.source_field_definition_identity
                        && resolved_observation_accepts_sample(
                            &self.plan,
                            observation,
                            entry.sample,
                            None,
                        )
                        .map_err(invalid)?
                    {
                        expected.push(entry);
                    }
                }
            }
            (
                ObservationFieldCatalogSource::VolumeMeshRecovery {
                    recovery_request,
                    manifest,
                    ..
                },
                ObservationHistorySourceContext::VolumeMesh { mesh },
            ) => {
                if recovery_request.producer != self.producer {
                    return Err(invalid("volume-mesh history producer differs"));
                }
                catalog.validate_volume_mesh_plan(&self.plan, mesh)?;
                for entry in &manifest.fields {
                    if entry.definition.identity() == observation.source_field_definition_identity
                        && resolved_observation_accepts_sample(
                            &self.plan,
                            observation,
                            entry.sample,
                            None,
                        )
                        .map_err(invalid)?
                    {
                        expected.push(entry);
                    }
                }
            }
            (
                ObservationFieldCatalogSource::EntryHistory { samples },
                ObservationHistorySourceContext::Structural {
                    source: super::StructuralObservationSource::Frequency(workload),
                    mesh,
                },
            ) => {
                if workload.procedure_step()?.step_id != self.plan.step_id {
                    return Err(invalid("frequency source workload selects another step"));
                }
                catalog.validate_frequency_history_plan(&self.plan, workload, mesh)?;
                self.accepted_entry_samples(samples, observation, &mut expected)?;
            }
            (
                ObservationFieldCatalogSource::EntryHistory { samples },
                ObservationHistorySourceContext::VolumeMesh { mesh },
            ) => {
                let step = match samples.first().map(ObservationFieldCatalog::source) {
                    Some(ObservationFieldCatalogSource::VolumeMeshRecovery { step, .. }) => step,
                    _ => return Err(invalid("volume-mesh entry history has no authored step")),
                };
                catalog.validate_volume_mesh_history_plan(&self.plan, self.producer, step, mesh)?;
                self.accepted_entry_samples(samples, observation, &mut expected)?;
            }
            (
                ObservationFieldCatalogSource::Thermal { manifest },
                ObservationHistorySourceContext::Thermal { workload, mesh },
            ) => {
                if manifest.producer != self.producer {
                    return Err(invalid("thermal producer differs"));
                }
                let step = workload
                    .model
                    .sequence
                    .steps
                    .iter()
                    .find(|step| step.step_id == workload.step_id)
                    .ok_or_else(|| invalid("thermal authored step absent"))?;
                if self.plan.step_id != step.step_id || self.plan.independent_domain != step.independent_domain
                    || !step.observations.iter().any(|request| matches!(request,
                        runmat_analysis_core::ObservationRequest::History(spec) if spec == &observation.spec)) {
                    return Err(invalid("history plan differs from authored thermal declaration"));
                }
                catalog.validate_thermal_plan(&self.plan, workload, mesh)?;
                for entries in &manifest.fields {
                    for entry in entries {
                        if entry.definition.identity()
                            == observation.source_field_definition_identity
                            && resolved_observation_accepts_sample(
                                &self.plan,
                                observation,
                                entry.sample,
                                None,
                            )
                            .map_err(invalid)?
                        {
                            expected.push(entry);
                        }
                    }
                }
            }
            _ => {
                return Err(invalid(
                    "source context differs from authoritative catalog kind",
                ))
            }
        }
        if expected.len() != checkpoint.samples().len() || expected.is_empty() {
            return Err(invalid(
                "history does not cover the exact accepted source sampling schedule",
            ));
        }
        for (entry, sample) in expected.iter().zip(checkpoint.samples()) {
            let evidence = &sample.result.partition_evidence;
            if sample.coordinate != entry.sample
                || sample.result.sample != entry.sample
                || evidence.source_field != entry.field
                || evidence.sample != entry.sample
                || evidence.source_partitions.is_empty()
                || evidence.source_partitions.iter().any(|identity| {
                    !entry
                        .partitions
                        .iter()
                        .any(|descriptor| descriptor.partition == *identity)
                })
            {
                return Err(invalid(
                    "history sample field, coordinate, or partition evidence differs from catalog",
                ));
            }
        }
        let stream = ObservationHistoryStream::new(
            self.plan.identity,
            self.plan.topology_digest,
            self.plan.independent_domain.clone(),
            observation,
            observation.sample_bounds.maximum_produced_samples,
            observation.sample_bounds.maximum_retained_samples,
        )
        .map_err(invalid)?;
        if checkpoint.manifest().stream != stream {
            return Err(invalid(
                "checkpoint stream differs from authored observation plan",
            ));
        }
        Ok(())
    }

    pub fn finalize(
        &self,
        checkpoint: ObservationHistoryCheckpoint,
        catalog: &ObservationFieldCatalog,
        context: ObservationHistorySourceContext<'_>,
    ) -> AnalysisExecutionResult<FinalizedObservationHistory> {
        self.validate_inputs(&checkpoint, catalog, context)?;
        let history =
            BoundedObservationHistory::resume(&self.plan, &self.observation_id, checkpoint)
                .map_err(invalid)?
                .finish()
                .map_err(invalid)?;
        let series_reduction = self
            .series_reduction
            .map(|_| {
                reduce_observation_history(
                    &self.plan,
                    &history,
                    ObservationExecutionPolicy::serial(),
                )
                .map_err(invalid)
            })
            .transpose()?;
        Ok(FinalizedObservationHistory {
            history,
            series_reduction,
        })
    }

    fn derive_identity(
        &self,
    ) -> AnalysisExecutionResult<ObservationHistoryFinalizationRequestIdentity> {
        Ok(ObservationHistoryFinalizationRequestIdentity(
            Digest::sha256(encode_contract(
                PREFIX,
                "history-finalization-request-identity",
                &(
                    self.schema_version,
                    self.plan.identity,
                    self.producer,
                    &self.observation_id,
                    self.terminal_checkpoint,
                    self.source_catalog,
                    self.series_reduction,
                ),
                LIMITS,
            )?),
        ))
    }
    /// Every accepted source coordinate of an entry history, in entry order: one producer and
    /// one topology across the entries.
    fn accepted_entry_samples<'a>(
        &self,
        samples: &'a [ObservationFieldCatalog],
        observation: &crate::results::observation::ResolvedObservation,
        expected: &mut Vec<&'a crate::results::field::RecoveredFieldArtifactEntry>,
    ) -> AnalysisExecutionResult<()> {
        for sample in samples {
            let (recovery_request, manifest) = sample.entry_recovery()?;
            // Structural recoveries share the plan's discretization digest; a volume-mesh
            // entry's plan was resolved on the mesh by `validate_volume_mesh_history_plan`.
            let structural = matches!(
                sample.source(),
                ObservationFieldCatalogSource::Structural { .. }
            );
            if recovery_request.producer != self.producer
                || (structural && manifest.topology_digest != self.plan.topology_digest)
            {
                return Err(invalid("entry history producer or topology differs"));
            }
            for entry in &manifest.fields {
                if entry.definition.identity() == observation.source_field_definition_identity
                    && resolved_observation_accepts_sample(&self.plan, observation, entry.sample, None)
                        .map_err(invalid)?
                {
                    expected.push(entry);
                }
            }
        }
        Ok(())
    }
    pub fn canonical_bytes(&self) -> AnalysisExecutionResult<Vec<u8>> {
        self.validate()?;
        Ok(encode_contract(
            PREFIX,
            "history-finalization-request",
            self,
            LIMITS,
        )?)
    }
    pub fn from_canonical_bytes(bytes: &[u8]) -> AnalysisExecutionResult<Self> {
        Ok(decode_contract(
            PREFIX,
            "history-finalization-request",
            bytes,
            LIMITS,
        )?)
    }
}

fn authored_series(
    plan: &ResolvedObservationPlan,
    id: &ObservationId,
) -> AnalysisExecutionResult<Option<SeriesReduction>> {
    let observation = plan
        .observations
        .iter()
        .find(|observation| &observation.spec.observation_id == id)
        .ok_or_else(|| invalid("requested history absent from resolved plan"))?;
    Ok(observation
        .spec
        .reduction
        .as_ref()
        .map(|reduction| reduction.series)
        .filter(|series| *series != SeriesReduction::None))
}

#[cfg(test)]
mod tests;
