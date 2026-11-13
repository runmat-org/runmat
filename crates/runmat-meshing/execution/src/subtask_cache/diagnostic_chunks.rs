//! Volume diagnostics retain their original transport namespace and no acceptance authority.
use super::record_chunks::{self, DIAGNOSTIC};
use super::*;

pub(crate) fn retain(
    store: &dyn MeshingSubtaskCacheStore,
    bytes: &[u8],
    materialization: u64,
    control: &dyn GeometryEvaluationControl,
) -> Result<Option<StableDigest>, MeshingSubtaskCacheError> {
    record_chunks::retain(DIAGNOSTIC, store, bytes, materialization, control)
}

pub struct VolumeDiagnosticRecord<'a> {
    record: MeshingSubtaskCacheEntry<'a>,
}
impl VolumeDiagnosticRecord<'_> {
    pub fn bytes(&self) -> &[u8] {
        &self.record.bytes
    }
}

pub fn read_volume_diagnostic<'a>(
    store: &dyn MeshingSubtaskCacheStore,
    key: StableDigest,
    control: &'a dyn GeometryEvaluationControl,
) -> Result<Option<VolumeDiagnosticRecord<'a>>, MeshingSubtaskCacheError> {
    let codec = crate::codec_control::CodecControl::new(control);
    record_chunks::read(DIAGNOSTIC, store, key, control, |bytes| {
        runmat_canonical_codec::canonical_chunked_payload_materialization_resident_bytes_estimate(
            [bytes],
            &codec,
        )
        .map_err(|error| codec.error(error))
    })
    .map(|record| record.map(|record| VolumeDiagnosticRecord { record }))
}

#[cfg(test)]
mod tests;
