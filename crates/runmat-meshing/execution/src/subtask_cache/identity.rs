//! Complete numerical scope plus the original local curve boundary, streamed into a digest.
use super::*;
use sha2::{Digest as _, Sha256};
use std::io::Write;

pub(super) fn scope(
    implementation: StableDigest,
    geometry: &runmat_execution::Digest,
    request: &runmat_meshing_core::MeshingRequest,
    metric: &runmat_meshing_core::MetricFieldRequest,
    control: &dyn GeometryEvaluationControl,
) -> Result<StableDigest, ExactFacePartitionError> {
    let mut writer = HashWriter::new(b"runmat-surface-face-scope/v1\0", control);
    writer
        .serialize(&(implementation, geometry, request, metric))
        .map_err(store_error)?;
    Ok(writer.finish())
}

pub(super) fn face(
    scope: StableDigest,
    boundary: &ExactFaceBoundary,
    coordinates: &ExactBoundaryNodeCoordinates<'_>,
    control: &dyn GeometryEvaluationControl,
) -> Result<StableDigest, ExactFacePartitionError> {
    let mut writer = HashWriter::new(b"runmat-surface-face-input/v1\0", control);
    writer.hash.update(scope.bytes());
    writer.serialize(boundary).map_err(store_error)?;
    for segment in boundary
        .outer_loop
        .segments
        .iter()
        .chain(boundary.inner_loops.iter().flat_map(|wire| &wire.segments))
    {
        for id in segment.node_ids {
            let point = coordinates
                .point(id, control)
                .map_err(control_error)?
                .ok_or_else(|| {
                    invalid("cached-face boundary lacks its original curve coordinate")
                })?;
            for value in point {
                writer.hash.update(value.to_bits().to_le_bytes());
            }
        }
    }
    Ok(writer.finish())
}

struct HashWriter<'a> {
    hash: Sha256,
    control: &'a dyn GeometryEvaluationControl,
    failure: Option<GeometryEvaluationError>,
}

impl<'a> HashWriter<'a> {
    fn new(domain: &[u8], control: &'a dyn GeometryEvaluationControl) -> Self {
        let mut hash = Sha256::new();
        hash.update(domain);
        Self {
            hash,
            control,
            failure: None,
        }
    }
    fn serialize(&mut self, value: &impl serde::Serialize) -> Result<(), MeshingSubtaskCacheError> {
        let encoded = serde_json::to_writer(&mut *self, value);
        if let Some(error) = self.failure.take() {
            return Err(MeshingSubtaskCacheError::Control(error));
        }
        encoded.map_err(|error| {
            MeshingSubtaskCacheError::Storage(format!("subtask input identity: {error}"))
        })
    }
    fn finish(self) -> StableDigest {
        StableDigest::from_bytes(self.hash.finalize().into())
    }
}

impl Write for HashWriter<'_> {
    fn write(&mut self, bytes: &[u8]) -> std::io::Result<usize> {
        for chunk in bytes.chunks(16 * 1024) {
            if let Err(error) = self
                .control
                .checkpoint()
                .and_then(|_| self.control.consume_search_work(1))
            {
                self.failure = Some(error);
                return Err(std::io::Error::other("face input identity interrupted"));
            }
            self.hash.update(chunk);
        }
        Ok(bytes.len())
    }
    fn flush(&mut self) -> std::io::Result<()> {
        Ok(())
    }
}

pub(crate) fn metric(
    implementation: StableDigest,
    geometry: &runmat_execution::Digest,
    request: &runmat_meshing_core::MeshingRequest,
    control: &dyn GeometryEvaluationControl,
) -> Result<StableDigest, MeshingSubtaskCacheError> {
    let mut writer = HashWriter::new(b"runmat-geometry-metric-input/v1\0", control);
    writer.serialize(&(
        implementation,
        geometry,
        &request.metric,
        request.quality.curve,
        request.quality.surface,
    ))?;
    Ok(writer.finish())
}
