//! Actual retained immutable object capacities. Decode/closure verification
//! scratch is deliberately absent once the verified input has been assembled.
use super::*;
impl PreparedMeshingStageObjects {
    pub(crate) fn retained_storage_bytes(
        &self,
        control: &dyn runmat_geometry_core::GeometryEvaluationControl,
    ) -> MeshingExecutionResult<u64> {
        let Self {
            result_identity,
            manifest,
            root,
            objects,
        } = self;
        let MeshingStageResultIdentity {
            schema_version: _,
            stage: _,
            result_kind: _,
            producer_identity_digest: _,
            logical_content_digest: _,
            logical_entity_count: _,
            invariant_summary_digest: _,
        } = result_identity;
        let mut bytes = std::mem::size_of::<Self>() as u64;
        let mut add = |value: u64| {
            control
                .checkpoint()
                .map_err(|error| MeshingExecutionError::Invalid(error.to_string()))?;
            bytes = bytes.checked_add(value).ok_or_else(|| {
                ArtifactError::Limit("retained stage object storage overflow".into())
            })?;
            Ok::<_, MeshingExecutionError>(())
        };
        add(manifest.owned_buffer_bytes()?)?;
        add(root.owned_buffer_bytes()?)?;
        add(crate::object_support::import_slots::<LogicalObject>(
            objects.capacity(),
        )?)?;
        for object in objects {
            let LogicalObject { descriptor, bytes } = object;
            add(descriptor.owned_buffer_bytes()?)?;
            add(bytes.capacity() as u64)?;
        }
        Ok(bytes)
    }
}
