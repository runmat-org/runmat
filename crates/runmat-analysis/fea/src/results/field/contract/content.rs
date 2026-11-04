//! Shared recovered-field content identity, validation, and canonical encoding.
use super::*;

mod profile;

// The identity remains the existing three-member tuple for both retained and projected storage.
fn identity_content<'a, S: Serialize + ?Sized>(
    definition: &'a RecoveredFieldDefinition,
    sample: RecoveredFieldSample,
    storage: &'a S,
) -> impl Serialize + 'a {
    (definition, sample, storage)
}

impl RecoveredField {
    /// The definition is the one authority for a recovered field's quantity, location, value
    /// shape and representation: the stored values take its layout. Only the published state
    /// (the ordinal: which mode or step) comes from the storage.
    pub fn new(
        definition: RecoveredFieldDefinition,
        sample: RecoveredFieldSample,
        mut storage: AnalysisField,
    ) -> Result<Self, RecoveredFieldError> {
        storage.layout = definition.layout().with_ordinal(storage.layout.ordinal);
        let field = Self {
            identity: RecoveredFieldIdentity(Digest::sha256([])),
            definition,
            sample,
            storage,
        };
        let mut field = field;
        field.identity = field.derive_identity()?;
        Ok(field)
    }

    pub const fn identity(&self) -> RecoveredFieldIdentity {
        self.identity
    }

    pub fn definition(&self) -> &RecoveredFieldDefinition {
        &self.definition
    }

    pub const fn sample(&self) -> RecoveredFieldSample {
        self.sample
    }

    pub fn storage(&self) -> &AnalysisField {
        &self.storage
    }

    pub fn into_storage(self) -> AnalysisField {
        self.storage
    }

    pub fn validate(&self) -> Result<(), RecoveredFieldError> {
        if self.storage.layout
            != self
                .definition
                .layout()
                .with_ordinal(self.storage.layout.ordinal)
        {
            return Err(RecoveredFieldError::StorageLayoutMismatch);
        }
        if self.identity != self.derive_identity()? {
            return Err(RecoveredFieldError::IdentityMismatch);
        }
        Ok(())
    }

    /// Validate and identify borrowed field content without cloning its numeric storage.
    /// Uses exactly the same canonical contract and limits as an owned recovered field.
    pub fn identity_for(
        definition: &RecoveredFieldDefinition,
        sample: RecoveredFieldSample,
        storage: &AnalysisField,
    ) -> Result<RecoveredFieldIdentity, RecoveredFieldError> {
        definition.validate()?;
        sample.validate()?;
        definition.validate_storage_shape(storage)?;
        // Borrowed content is identified as the owned field would be: in its definition's layout.
        if storage.layout != definition.layout().with_ordinal(storage.layout.ordinal) {
            return Err(RecoveredFieldError::StorageLayoutMismatch);
        }
        encode_contract(
            FIELD_PREFIX,
            "recovered-field-identity",
            &identity_content(definition, sample, storage),
            FIELD_LIMITS,
        )
        .map(Digest::sha256)
        .map(RecoveredFieldIdentity)
        .map_err(|error| RecoveredFieldError::Encoding(error.to_string()))
    }

    pub fn canonical_bytes(&self) -> Result<Vec<u8>, RecoveredFieldError> {
        self.validate()?;
        encode_contract(FIELD_PREFIX, "recovered-field", self, FIELD_LIMITS)
            .map_err(|error| RecoveredFieldError::Encoding(error.to_string()))
    }

    pub fn from_canonical_bytes(bytes: &[u8]) -> Result<Self, RecoveredFieldError> {
        decode_contract(FIELD_PREFIX, "recovered-field", bytes, FIELD_LIMITS)
            .map_err(|error| RecoveredFieldError::Encoding(error.to_string()))
    }

    fn derive_identity(&self) -> Result<RecoveredFieldIdentity, RecoveredFieldError> {
        Self::identity_for(&self.definition, self.sample, &self.storage)
    }
}

impl RecoveredFieldDefinition {
    /// Shared storage cardinality boundary for complete fields and compact partitions.
    pub(crate) fn validate_storage_shape(
        &self,
        storage: &AnalysisField,
    ) -> Result<(), RecoveredFieldError> {
        if storage.field_id != self.field_id {
            return Err(RecoveredFieldError::StorageIdentifierMismatch);
        }
        let expected = storage
            .shape
            .iter()
            .try_fold(1usize, |count, extent| count.checked_mul(*extent))
            .ok_or(RecoveredFieldError::StorageShapeMismatch)?;
        let actual = match &storage.values {
            AnalysisFieldValues::HostF64(values) => values.len(),
            AnalysisFieldValues::DeviceRef(reference) => reference.element_count,
        };
        let entity_count = storage
            .shape
            .first()
            .copied()
            .ok_or(RecoveredFieldError::StorageShapeMismatch)?;
        if expected != actual || storage.shape != self.storage_shape(entity_count)? {
            return Err(RecoveredFieldError::StorageShapeMismatch);
        }
        let layout = self.physical_layout(entity_count)?;
        if let AnalysisFieldValues::HostF64(values) = &storage.values {
            runmat_types::physical::QuantityField::try_new(layout, values.as_slice())
                .map_err(|_| RecoveredFieldError::InvalidPhysicalStorage)?;
        }
        Ok(())
    }
}
