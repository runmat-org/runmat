//! Shared quantity projection of the authoritative recovered-field header.
//! Complex real/imaginary lanes have the same physical quantity; tensor/component
//! shape and entity association remain part of the existing definition.
use super::{RecoveredField, RecoveredFieldDefinition, RecoveredFieldError};
use runmat_types::physical::{DynamicDimension, QuantityField, QuantityLayout};

#[derive(Clone, Copy, Debug)]
pub struct RecoveredQuantityLayout<'a> {
    definition: &'a RecoveredFieldDefinition,
    components: usize,
    descriptor: DynamicDimension,
}
impl QuantityLayout for RecoveredQuantityLayout<'_> {
    fn component_count(&self) -> usize {
        self.components
    }
    fn component_descriptor(&self, index: usize) -> Option<DynamicDimension> {
        (index < self.components).then_some(self.descriptor)
    }
}
impl RecoveredQuantityLayout<'_> {
    pub fn definition(&self) -> &RecoveredFieldDefinition {
        self.definition
    }
}
impl RecoveredFieldDefinition {
    pub fn physical_layout(
        &self,
        entity_count: usize,
    ) -> Result<RecoveredQuantityLayout<'_>, RecoveredFieldError> {
        let descriptor = self
            .quantity()
            .quantity_descriptor()
            .ok_or(RecoveredFieldError::MissingQuantityDescriptor)?;
        let components = self
            .storage_shape(entity_count)?
            .into_iter()
            .try_fold(1usize, |count, extent| count.checked_mul(extent))
            .ok_or(RecoveredFieldError::StorageShapeMismatch)?;
        Ok(RecoveredQuantityLayout {
            definition: self,
            components,
            descriptor,
        })
    }
}
impl RecoveredField {
    /// Borrow values together with their checked quantity, representation and
    /// association. Numerical reductions may extract only inside this typed view.
    pub fn physical_host_field(
        &self,
    ) -> Result<QuantityField<RecoveredQuantityLayout<'_>, &[f64]>, RecoveredFieldError> {
        let values = self.storage().as_host_f64().ok_or_else(|| {
            RecoveredFieldError::InvalidArtifact(
                "physical host projection requires host-resident storage".into(),
            )
        })?;
        let count = *self
            .storage()
            .shape
            .first()
            .ok_or(RecoveredFieldError::StorageShapeMismatch)?;
        QuantityField::try_new(self.definition().physical_layout(count)?, values)
            .map_err(|_| RecoveredFieldError::InvalidPhysicalStorage)
    }
}
