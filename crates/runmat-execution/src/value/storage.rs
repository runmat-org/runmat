//! Physical reference string ownership, distinct from logical encoded length.
use super::*;
impl ValueRef {
    pub fn owned_buffer_bytes(&self) -> Result<u64, ContractError> {
        let Self {
            schema_version: _,
            id: _,
            logical_digest: _,
            encoded_length: _,
            media_type,
            value_schema,
            encryption_context: _,
            kind: _,
            authorization_scope,
            resident_fence,
        } = self;
        // Accelerator fence text is retained beside the reference; bound it
        // by its identifier lengths plus one feature slot per declared feature.
        let device_bytes = resident_fence
            .as_ref()
            .and_then(|fence| fence.accelerator.as_ref())
            .map_or(0, |accelerator| {
                let lease = &accelerator.lease;
                lease.class.as_str().len()
                    + lease.provider.id.as_str().len()
                    + lease.provider.version.as_str().len()
                    + lease.features.len()
                        * std::mem::size_of::<crate::resource::AcceleratorFeature>()
            });
        [
            media_type.capacity(),
            value_schema.capacity(),
            authorization_scope.capacity(),
            device_bytes,
        ]
        .into_iter()
        .try_fold(0usize, usize::checked_add)
        .and_then(|bytes| u64::try_from(bytes).ok())
        .ok_or_else(|| {
            ContractError::invalid(
                "value reference storage",
                "owned string capacity inventory overflow",
            )
        })
    }
}
