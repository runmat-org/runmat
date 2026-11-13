//! Logical result identity and exact native entity/storage validation.
use super::*;
impl ObservationResult {
    pub(in crate::results::observation) fn derive_identity(
        &self,
    ) -> Result<ObservationResultIdentity, ObservationError> {
        let bytes = encode_contract(
            OBSERVATION_PREFIX,
            "observation-result-identity",
            &(
                self.schema_version,
                self.plan_identity,
                self.topology_digest,
                &self.observation_id,
                self.sample,
                self.quantity,
                self.location,
                &self.value_shape,
                self.complex_representation,
                &self.entities,
                &self.field,
                self.partial_access,
            ),
            OBSERVATION_LIMITS,
        )?;
        Ok(ObservationResultIdentity(Digest::sha256(bytes)))
    }

    pub fn validate(&self) -> Result<(), ObservationError> {
        let values = self
            .field
            .as_host_f64()
            .ok_or(ObservationError::Unsupported(
                "observation publication requires host-resident f64 values",
            ))?;
        let expected = self
            .field
            .shape
            .iter()
            .try_fold(1usize, |count, extent| count.checked_mul(*extent));
        let entity_shape_is_valid = self.field.shape
            == observation_result_storage_shape(
                (!self.entities.is_empty()).then_some(self.entities.len()),
                &self.value_shape,
                self.complex_representation,
            )?;
        let mut entity_ids = std::collections::BTreeSet::new();
        let entities_are_valid = self.entities.iter().all(|entity| {
            let (location, identity) = match entity {
                ObservationEntity::Node(identity) => (FieldLocation::Node, identity),
                ObservationEntity::Element(identity) => (FieldLocation::Element, identity),
                ObservationEntity::IntegrationPoint(identity) => {
                    (FieldLocation::IntegrationPoint, identity)
                }
                ObservationEntity::OrientedEdge(identity) => (FieldLocation::Edge, identity),
                ObservationEntity::OrientedFace(identity) => (FieldLocation::Face, identity),
                ObservationEntity::Connector(identity) => (FieldLocation::Connector, identity),
                ObservationEntity::Point {
                    identity,
                    position_m,
                    source_element,
                    ..
                } if position_m.iter().all(|value| value.is_finite())
                    && *source_element != StableDigest::ZERO =>
                {
                    (FieldLocation::Point, identity)
                }
                ObservationEntity::Point { .. } => return false,
            };
            (location == self.location
                || (location == FieldLocation::Element && self.location == FieldLocation::Cell))
                && entity_ids.insert(*identity)
        });
        let evidence_shape_is_valid = match self.partition_evidence.combination {
            ObservationPartitionCombination::CanonicalConcatenation => !self.entities.is_empty(),
            ObservationPartitionCombination::CanonicalCompensatedReduction { collective_count } => {
                self.entities.is_empty()
                    && collective_count as usize
                        == self
                            .partition_evidence
                            .contributions
                            .len()
                            .saturating_sub(1)
            }
            ObservationPartitionCombination::CanonicalExtremaReduction { collective_count } => {
                !self.entities.is_empty()
                    && collective_count as usize
                        == self
                            .partition_evidence
                            .contributions
                            .len()
                            .saturating_sub(1)
            }
        };
        self.partition_evidence.validate()?;
        if self.schema_version != 1
            || self.field.field_id != self.observation_id.0
            || expected != Some(values.len())
            || values.is_empty()
            || values.iter().any(|value| !value.is_finite())
            || !entity_shape_is_valid
            || !entities_are_valid
            || !evidence_shape_is_valid
            || self.partition_evidence.sample != self.sample
            || self.logical_identity != self.derive_identity()?
        {
            return Err(ObservationError::Invalid(
                "observation result metadata, values, or identity is inconsistent".into(),
            ));
        }
        Ok(())
    }

    pub fn canonical_bytes(&self) -> Result<Vec<u8>, ObservationError> {
        self.validate()?;
        Ok(encode_contract(
            OBSERVATION_PREFIX,
            "observation-result",
            self,
            OBSERVATION_LIMITS,
        )?)
    }

    pub fn from_canonical_bytes(bytes: &[u8]) -> Result<Self, ObservationError> {
        let result: Self = decode_contract(
            OBSERVATION_PREFIX,
            "observation-result",
            bytes,
            OBSERVATION_LIMITS,
        )?;
        result.validate()?;
        Ok(result)
    }
}
