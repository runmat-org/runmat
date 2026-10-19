//! Verify each immutable input in its own allocation scope, then transfer only
//! retained object/reference capacities. Each complete closure is still checked.
use super::*;
use crate::codec_control::CodecControl;
use runmat_geometry_core::GeometryEvaluationControl;

pub(super) fn import<S: CacheImport>(
    reader: &InputReader<'_, S>,
    root: &runmat_execution::value::ValueRef,
    access: &crate::ArtifactAccess,
    limits: runmat_execution_artifact::object::ObjectInventoryLimits,
) -> crate::MeshingExecutionResult<crate::PreparedMeshingResultPublication> {
    let scope = GeometryAllocationScope::new(reader.scope);
    let nested = InputReader::new(reader.source, &scope);
    let codec = CodecControl::new(&scope);
    let result = (|| {
        let mut admit = |bytes| {
            codec
                .consume_allocation_bytes(bytes)
                .map_err(|error| crate::MeshingExecutionError::Invalid(error.to_string()))
        };
        let prepared = crate::publication::import_result_publication_with_admission(
            &nested, root, access, limits, &codec, &mut admit,
        )?;
        // The caller already admitted the boxed input's inline value.
        let retained = prepared
            .retained_storage_bytes(&codec)?
            .checked_sub(std::mem::size_of::<crate::PreparedMeshingResultPublication>() as u64)
            .ok_or(crate::MeshingExecutionError::Identity(
                "publication inline inventory is absent",
            ))?;
        let mut ownership = scope
            .transfer_reservation(retained)
            .map_err(|error| reader.control_error(error))?;
        ownership.retain_bytes(retained);
        Ok(prepared)
    })();
    result.map_err(
        |error| match codec.take_failure().or_else(|| nested.take_failure()) {
            Some(original) => reader.control_error(original).into(),
            None => error,
        },
    )
}
