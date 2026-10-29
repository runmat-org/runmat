use std::collections::BTreeMap;

use runmat_execution::Digest;

use crate::topology::AnalysisProjection;

use super::contract::{FIELD_LIMITS, FIELD_PREFIX};
use super::{
    RecoveredField, RecoveredFieldArtifact, RecoveredFieldDefinitionIdentity,
    RecoveredFieldEntityCounts, RecoveredFieldError, RecoveredFieldSample,
};

#[derive(Clone, Debug)]
pub struct RecoveredFieldCatalog {
    topology_digest: Digest,
    fields: Vec<RecoveredField>,
    by_sample: BTreeMap<Digest, usize>,
}

impl RecoveredFieldCatalog {
    pub fn from_artifact(
        projection: &AnalysisProjection,
        artifact: RecoveredFieldArtifact,
    ) -> Result<Self, RecoveredFieldError> {
        artifact.validate()?;
        if artifact.topology_digest != projection.projection_digest
            || artifact.entity_counts
                != (RecoveredFieldEntityCounts {
                    oriented_faces: 0,
                    nodes: projection.nodes.len(),
                    volume_elements: projection.elements.len(),
                    volume_integration_points: projection.elements.len() * 4,
                    oriented_edges: 0,
                    beam_elements: 0,
                    shell_elements: 0,
                    connectors: 0,
                    constraint_groups: 0,
                })
        {
            return Err(RecoveredFieldError::ArtifactTopologyMismatch);
        }
        Self::new(projection, artifact.into_fields())
    }

    pub fn new(
        projection: &AnalysisProjection,
        fields: Vec<RecoveredField>,
    ) -> Result<Self, RecoveredFieldError> {
        Self::new_for_topology(
            projection.projection_digest,
            RecoveredFieldEntityCounts {
                oriented_faces: 0,
                nodes: projection.nodes.len(),
                volume_elements: projection.elements.len(),
                volume_integration_points: projection.elements.len() * 4,
                oriented_edges: 0,
                beam_elements: 0,
                shell_elements: 0,
                connectors: 0,
                constraint_groups: 0,
            },
            fields,
        )
    }

    pub fn new_for_topology(
        topology_digest: Digest,
        entity_counts: RecoveredFieldEntityCounts,
        fields: Vec<RecoveredField>,
    ) -> Result<Self, RecoveredFieldError> {
        let mut by_sample = BTreeMap::new();
        for (index, field) in fields.iter().enumerate() {
            let definition = field.definition();
            let sample_key = sample_key(definition.identity(), field.sample())?;
            if by_sample.insert(sample_key, index).is_some() {
                return Err(RecoveredFieldError::CatalogDuplicateSample);
            }
            let entity_count = definition.stored_entities(entity_counts);
            if entity_count == 0 {
                return Err(RecoveredFieldError::ArtifactTopologyMismatch);
            }
            let expected_shape = definition.storage_shape(entity_count)?;
            if field.storage().shape != expected_shape {
                return Err(RecoveredFieldError::StorageShapeMismatch);
            }
        }
        Ok(Self {
            topology_digest,
            fields,
            by_sample,
        })
    }

    pub const fn topology_digest(&self) -> Digest {
        self.topology_digest
    }

    pub fn fields(&self) -> &[RecoveredField] {
        &self.fields
    }

    pub fn into_fields(self) -> Vec<RecoveredField> {
        self.fields
    }

    pub fn get(
        &self,
        definition: RecoveredFieldDefinitionIdentity,
        sample: RecoveredFieldSample,
    ) -> Option<&RecoveredField> {
        let key = sample_key(definition, sample).ok()?;
        self.by_sample
            .get(&key)
            .and_then(|index| self.fields.get(*index))
    }
}

fn sample_key(
    definition: RecoveredFieldDefinitionIdentity,
    sample: RecoveredFieldSample,
) -> Result<Digest, RecoveredFieldError> {
    sample.validate()?;
    crate::contracts::canonical_digest::canonical_contract_digest(
        FIELD_PREFIX,
        "recovered-field-sample",
        &(definition, sample),
        FIELD_LIMITS,
    )
    .map_err(|error| RecoveredFieldError::Encoding(error.to_string()))
}
