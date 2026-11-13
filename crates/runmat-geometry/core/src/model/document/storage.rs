//! Exact requested backing inventory for an owned document and its clone.
//! Inline document fields belong to their containing carrier; allocator overhead
//! and process RSS are not reported as typed buffer storage.
use super::*;
impl GeometryDocument {
    pub fn cloned_buffer_storage_bytes(&self) -> Result<u64, GeometryContractError> {
        self.buffer_storage_bytes(true)
    }
    pub fn owned_buffer_storage_bytes(&self) -> Result<u64, GeometryContractError> {
        self.buffer_storage_bytes(false)
    }
    fn buffer_storage_bytes(&self, cloning: bool) -> Result<u64, GeometryContractError> {
        self.validate()?;
        let string_bytes = |value: &String| {
            if cloning {
                value.len()
            } else {
                value.capacity()
            }
        };
        let count = if cloning {
            self.display_tessellations.len()
        } else {
            self.display_tessellations.capacity()
        };
        let mut bytes = count
            .checked_mul(std::mem::size_of::<DisplayTessellationRef>())
            .ok_or_else(overflow)?;
        let mut add = |value: usize| -> Result<(), GeometryContractError> {
            bytes = bytes.checked_add(value).ok_or_else(overflow)?;
            Ok(())
        };
        add(string_bytes(&self.source.importer_version))?;
        if let Some(version) = &self.source.kernel_version {
            add(string_bytes(version))?;
        }
        add(string_bytes(&self.healing.algorithm_version))?;
        add(string_bytes(&self.primary_artifact().media_type))?;
        if let GeometryModel::ExactBRep { model } = &self.model {
            add(string_bytes(&model.kernel_abi))?;
        }
        for display in &self.display_tessellations {
            add(string_bytes(&display.profile_id))?;
            add(string_bytes(&display.artifact.media_type))?;
        }
        u64::try_from(bytes).map_err(|_| overflow())
    }
}
fn overflow() -> GeometryContractError {
    GeometryContractError::invalid("geometry document storage", "buffer inventory overflow")
}
