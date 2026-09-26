//! Observation provenance over already published fields, independent of field production.
use runmat_canonical_codec::{decode_contract, encode_contract, CanonicalLimits};
use runmat_execution::Digest;
use runmat_meshing_core::{SolverMeshArtifact, StableDigest};
use serde::{Deserialize, Serialize};

use crate::execution::{
    AnalysisExecutionError, AnalysisExecutionResult, AnalysisWorkloadIdentity,
    FieldRecoveryRequest, FieldRecoveryRequestIdentity, FieldRecoverySource, SolverResultIdentity,
    StepIdentity, ThermalNumericalIntentIdentity, ThermalSolverResultManifest, ThermalWorkload,
};
use crate::physics::structural::topology::CanonicalStructuralTopologyIdentity;
use crate::results::field::{
    RecoveredFieldArtifactEntry, RecoveredFieldArtifactIdentity, RecoveredFieldArtifactManifest,
    RecoveredFieldIdentity, RecoveredFieldPartitionSet, RecoveredFieldSelection,
};
use crate::topology::AnalysisProjection;

const PREFIX: &[u8] = b"runmat-observation-field-selection/v1\0";
const LIMITS: CanonicalLimits = CanonicalLimits::new(64 * 1024 * 1024, 1_000_000, 65_536, 64);

#[cfg(test)]
mod tests;

#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case", tag = "kind", deny_unknown_fields)]
pub enum ObservationFieldSource {
    StructuralRecovery {
        recovery_request: FieldRecoveryRequestIdentity,
        artifact: RecoveredFieldArtifactIdentity,
        numerical_source: FieldRecoverySource,
        topology: CanonicalStructuralTopologyIdentity,
    },
    ThermalResult {
        workload: AnalysisWorkloadIdentity,
        result: SolverResultIdentity,
        numerical_intent: ThermalNumericalIntentIdentity,
        mesh_digest: StableDigest,
        projection_digest: Digest,
        sample_ordinal: usize,
    },
}

impl ObservationFieldSource {
    pub const fn solver_result(self) -> Option<SolverResultIdentity> {
        match self {
            Self::StructuralRecovery {
                numerical_source, ..
            } => numerical_source.solver_result(),
            Self::ThermalResult { result, .. } => Some(result),
        }
    }

    pub fn observation_topology_digest(self) -> Digest {
        match self {
            Self::StructuralRecovery { topology, .. } => topology.discretization_digest(),
            Self::ThermalResult {
                projection_digest, ..
            } => projection_digest,
        }
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(transparent)]
pub struct ObservationFieldSelectionIdentity(Digest);

impl ObservationFieldSelectionIdentity {
    pub const fn digest(self) -> Digest {
        self.0
    }
}

/// Compact selected catalog. Decode checks internal integrity; admission must additionally call
/// `validate_structural` or `validate_thermal` against the authoritative producing catalog.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields, try_from = "SelectionWire")]
pub struct ObservationFieldSelection {
    schema_version: u16,
    identity: ObservationFieldSelectionIdentity,
    producer: StepIdentity,
    source: ObservationFieldSource,
    fields: Vec<RecoveredFieldArtifactEntry>,
}

#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct SelectionWire {
    schema_version: u16,
    identity: ObservationFieldSelectionIdentity,
    producer: StepIdentity,
    source: ObservationFieldSource,
    fields: Vec<RecoveredFieldArtifactEntry>,
}

impl TryFrom<SelectionWire> for ObservationFieldSelection {
    type Error = AnalysisExecutionError;
    fn try_from(wire: SelectionWire) -> Result<Self, Self::Error> {
        let selection = Self {
            schema_version: wire.schema_version,
            identity: wire.identity,
            producer: wire.producer,
            source: wire.source,
            fields: wire.fields,
        };
        selection.validate()?;
        Ok(selection)
    }
}

fn invalid(error: impl std::fmt::Display) -> AnalysisExecutionError {
    AnalysisExecutionError::Invalid(format!("observation field selection: {error}"))
}

impl ObservationFieldSelection {
    fn new(
        producer: StepIdentity,
        source: ObservationFieldSource,
        fields: Vec<RecoveredFieldArtifactEntry>,
    ) -> AnalysisExecutionResult<Self> {
        let mut selection = Self {
            schema_version: 1,
            identity: ObservationFieldSelectionIdentity(Digest::sha256([])),
            producer,
            source,
            fields,
        };
        selection.identity = selection.derive_identity()?;
        selection.validate()?;
        Ok(selection)
    }

    pub fn structural(
        request: &FieldRecoveryRequest,
        manifest: &RecoveredFieldArtifactManifest,
        selected: Vec<RecoveredFieldIdentity>,
    ) -> AnalysisExecutionResult<Self> {
        request.validate()?;
        let selection =
            RecoveredFieldSelection::new(manifest.clone(), selected).map_err(invalid)?;
        if request.identity() != manifest.recovery_request
            || request.source != manifest.source
            || request.topology.discretization_digest() != manifest.topology_digest
        {
            return Err(invalid(
                "structural catalog differs from its recovery request",
            ));
        }
        let fields = manifest
            .fields
            .iter()
            .filter(|entry| selection.fields.contains(&entry.field))
            .cloned()
            .collect();
        Self::new(
            request.producer,
            ObservationFieldSource::StructuralRecovery {
                recovery_request: request.identity(),
                artifact: manifest.identity,
                numerical_source: request.source,
                topology: request.topology,
            },
            fields,
        )
    }

    pub fn thermal(
        workload: &ThermalWorkload,
        manifest: &ThermalSolverResultManifest,
        mesh: &SolverMeshArtifact,
        sample_ordinal: usize,
        selected: Vec<RecoveredFieldIdentity>,
    ) -> AnalysisExecutionResult<Self> {
        workload.validate()?;
        manifest.validate()?;
        let projection = AnalysisProjection::derive(mesh).map_err(invalid)?;
        let schedule = workload.solve_schedule().map_err(invalid)?;
        let step = workload
            .model
            .sequence
            .steps
            .iter()
            .find(|step| step.step_id == workload.step_id)
            .ok_or_else(|| invalid("thermal producing step is absent"))?;
        let sequence =
            crate::execution::SequenceIdentity::from_solver_contract(&workload.model.sequence)?;
        let producer = StepIdentity::from_solver_contract(sequence, step)?;
        if manifest.source_workload != workload.identity()?
            || manifest.numerical_intent != workload.numerical_intent_identity()?
            || manifest.producer != producer
            || manifest.topology_digest != Digest::from_bytes(*projection.mesh_digest.bytes())
            || manifest.time_points_s != schedule.points_s()
            || !manifest
                .ordered_node_ids
                .iter()
                .copied()
                .eq(projection.nodes.iter().map(|node| node.node_id))
            || !manifest
                .ordered_element_ids
                .iter()
                .copied()
                .eq(projection.elements.iter().map(|element| element.element_id))
        {
            return Err(invalid(
                "thermal catalog differs from workload, mesh, producer, or schedule",
            ));
        }
        let catalog = manifest
            .fields
            .get(sample_ordinal)
            .ok_or_else(|| invalid("sample ordinal is absent"))?;
        let fields = catalog
            .iter()
            .filter(|entry| selected.contains(&entry.field))
            .cloned()
            .collect::<Vec<_>>();
        if fields.iter().map(|entry| entry.field).collect::<Vec<_>>() != selected {
            return Err(invalid(
                "selected fields are absent, duplicated, reordered, or from another sample",
            ));
        }
        Self::new(
            producer,
            ObservationFieldSource::ThermalResult {
                workload: manifest.source_workload,
                result: manifest.identity(),
                numerical_intent: manifest.numerical_intent,
                mesh_digest: projection.mesh_digest,
                projection_digest: projection.projection_digest,
                sample_ordinal,
            },
            fields,
        )
    }

    pub fn validate_structural(
        &self,
        request: &FieldRecoveryRequest,
        manifest: &RecoveredFieldArtifactManifest,
    ) -> AnalysisExecutionResult<()> {
        let expected = Self::structural(
            request,
            manifest,
            self.fields.iter().map(|entry| entry.field).collect(),
        )?;
        if self != &expected {
            return Err(invalid("selection differs from structural authority"));
        }
        Ok(())
    }

    pub fn validate_thermal(
        &self,
        workload: &ThermalWorkload,
        manifest: &ThermalSolverResultManifest,
        mesh: &SolverMeshArtifact,
    ) -> AnalysisExecutionResult<()> {
        let ObservationFieldSource::ThermalResult { sample_ordinal, .. } = self.source else {
            return Err(invalid("selection is not thermal"));
        };
        let expected = Self::thermal(
            workload,
            manifest,
            mesh,
            sample_ordinal,
            self.fields.iter().map(|entry| entry.field).collect(),
        )?;
        if self != &expected {
            return Err(invalid("selection differs from thermal authority"));
        }
        Ok(())
    }

    pub const fn identity(&self) -> ObservationFieldSelectionIdentity {
        self.identity
    }
    pub const fn producer(&self) -> StepIdentity {
        self.producer
    }
    pub const fn source(&self) -> ObservationFieldSource {
        self.source
    }
    pub fn fields(&self) -> &[RecoveredFieldArtifactEntry] {
        &self.fields
    }

    pub fn validate(&self) -> AnalysisExecutionResult<()> {
        if self.schema_version != 1
            || self.fields.is_empty()
            || self.identity != self.derive_identity()?
        {
            return Err(invalid("invalid schema, empty selection, or identity"));
        }
        let mut seen = std::collections::BTreeSet::new();
        for entry in &self.fields {
            entry.definition.validate().map_err(invalid)?;
            if !seen.insert(entry.field) || entry.partitions.is_empty() {
                return Err(invalid("duplicate field or absent partitions"));
            }
            let mut end = 0usize;
            for descriptor in &entry.partitions {
                if descriptor.first_source_index != end || descriptor.entity_count == 0 {
                    return Err(invalid("invalid partition coverage"));
                }
                end = end
                    .checked_add(descriptor.entity_count)
                    .ok_or_else(|| invalid("coverage overflow"))?;
            }
        }
        Ok(())
    }

    pub fn validate_partitions(
        &self,
        partitions: &RecoveredFieldPartitionSet,
    ) -> AnalysisExecutionResult<()> {
        self.validate()?;
        partitions.validate().map_err(invalid)?;
        let entry = self
            .fields
            .iter()
            .find(|entry| entry.field == partitions.source_field)
            .ok_or_else(|| invalid("partition field is outside selection"))?;
        if entry.definition != partitions.definition || entry.sample != partitions.sample {
            return Err(invalid(
                "partition definition or sample differs from selection",
            ));
        }
        for partition in &partitions.partitions {
            if !entry.partitions.iter().any(|descriptor| {
                descriptor.partition == partition.identity
                    && Some(descriptor.first_source_index)
                        == partition.source_indices.first().copied()
                    && descriptor.entity_count == partition.source_indices.len()
            }) {
                return Err(invalid(
                    "partition is outside authoritative descriptor inventory",
                ));
            }
        }
        Ok(())
    }

    fn derive_identity(&self) -> AnalysisExecutionResult<ObservationFieldSelectionIdentity> {
        Ok(ObservationFieldSelectionIdentity(Digest::sha256(
            encode_contract(
                PREFIX,
                "observation-field-selection-identity",
                &(
                    self.schema_version,
                    self.producer,
                    self.source,
                    &self.fields,
                ),
                LIMITS,
            )?,
        )))
    }
    pub fn canonical_bytes(&self) -> AnalysisExecutionResult<Vec<u8>> {
        self.validate()?;
        Ok(encode_contract(
            PREFIX,
            "observation-field-selection",
            self,
            LIMITS,
        )?)
    }
    pub fn from_canonical_bytes(bytes: &[u8]) -> AnalysisExecutionResult<Self> {
        Ok(decode_contract(
            PREFIX,
            "observation-field-selection",
            bytes,
            LIMITS,
        )?)
    }
}
