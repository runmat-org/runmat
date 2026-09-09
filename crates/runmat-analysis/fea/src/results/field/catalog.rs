use std::collections::BTreeMap;

use runmat_analysis_core::FieldLocation;
use runmat_canonical_codec::{encode_contract, CanonicalLimits};
use runmat_execution::Digest;

use crate::topology::AnalysisProjection;

use super::{
    RecoveredField, RecoveredFieldArtifact, RecoveredFieldDefinitionIdentity, RecoveredFieldError,
    RecoveredFieldSample,
};

const FIELD_PREFIX: &[u8] = b"runmat-fea-recovered-field/v1\0";
const FIELD_LIMITS: CanonicalLimits = CanonicalLimits::new(64 * 1024 * 1024, 1_000_000, 4096, 32);

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
            || artifact.node_count != projection.nodes.len()
            || artifact.element_count != projection.elements.len()
        {
            return Err(RecoveredFieldError::ArtifactTopologyMismatch);
        }
        Self::new(projection, artifact.into_fields())
    }

    pub fn new(
        projection: &AnalysisProjection,
        fields: Vec<RecoveredField>,
    ) -> Result<Self, RecoveredFieldError> {
        let mut by_sample = BTreeMap::new();
        for (index, field) in fields.iter().enumerate() {
            let definition = field.definition();
            let sample_key = sample_key(definition.identity(), field.sample())?;
            if by_sample.insert(sample_key, index).is_some() {
                return Err(RecoveredFieldError::CatalogDuplicateSample);
            }
            let entity_count = match definition.location() {
                FieldLocation::Node => projection.nodes.len(),
                FieldLocation::Element => projection.elements.len(),
                _ => return Err(RecoveredFieldError::CatalogUnsupportedLocation),
            };
            let components = definition
                .value_shape()
                .component_count()
                .ok_or(RecoveredFieldError::QuantityShapeMismatch)?;
            let expected_shape = if components == 1 {
                vec![entity_count]
            } else {
                vec![entity_count, components]
            };
            if field.storage().shape != expected_shape {
                return Err(RecoveredFieldError::StorageShapeMismatch);
            }
        }
        Ok(Self {
            topology_digest: projection.projection_digest,
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
    encode_contract(
        FIELD_PREFIX,
        "recovered-field-sample",
        &(definition, sample),
        FIELD_LIMITS,
    )
    .map(Digest::sha256)
    .map_err(|error| RecoveredFieldError::Encoding(error.to_string()))
}
