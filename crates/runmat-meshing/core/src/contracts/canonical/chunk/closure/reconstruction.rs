//! Admit and size the complete reconstruction before allocating record payloads.
use super::*;
use crate::EncodedMeshingChunkRef;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct MeshingStreamStorageProfile {
    /// Output stream/record slots plus payload capacity; excludes allocator overhead.
    pub retained_bytes: u64,
    /// Record lengths, validated chunk views, manifest uniqueness and identity codec;
    /// released before returning the streams.
    pub scratch_bytes: u64,
}
impl MeshingStreamStorageProfile {
    pub fn requested_bytes(self) -> Result<u64, MeshingContractError> {
        add(self.retained_bytes, self.scratch_bytes)
    }
}

/// The admission callback runs before allocating either the layout or the output.
/// It must keep retained storage charged while returned record buffers remain live.
/// Raw callers can use the cancellation-only entry point; governed callers supply
/// their existing resource authority. No serialized claim is trusted as validation.
pub fn decode_stage_manifest_streams_borrowed_with_admission(
    manifest: &MeshingStageManifest,
    identity: &MeshingStageResultIdentity,
    chunks: &[EncodedMeshingChunkRef<'_>],
    cancellation: &dyn MeshingCancellationSignal,
    mut admit: impl FnMut(MeshingStreamStorageProfile) -> Result<(), MeshingContractError>,
) -> Result<Vec<MeshingChunkStream>, MeshingContractError> {
    cancellation_checkpoint(cancellation)?;
    let profile = layout::storage_profile(manifest, identity, chunks, true)?;
    admit(profile)?;
    let layout = layout::prepare(manifest, identity, chunks, cancellation)?;
    let lengths = &layout.lengths;
    let mut output = Vec::<MeshingChunkStream>::with_capacity(layout.streams);
    let mut record_index = 0;
    let mut record = Vec::new();
    for (chunk_index, decoded) in layout.chunks.iter().enumerate() {
        let key = (decoded.media_type, decoded.schema_version);
        if output
            .last()
            .map(|stream| (stream.media_type, stream.schema_version))
            != Some(key)
        {
            let count = chunks[chunk_index..]
                .iter()
                .take_while(|next| {
                    (next.descriptor.media_type, next.descriptor.schema_version) == key
                })
                .try_fold(0_u64, |total, next| {
                    add(total, next.descriptor.logical_entity_count)
                })?;
            if count == 0 {
                return Err(invalid("stream has no records"));
            }
            output.push(MeshingChunkStream {
                media_type: key.0,
                schema_version: key.1,
                records: Vec::with_capacity(count as usize),
            });
        }
        for fragment in decoded.fragments.iter() {
            if record.capacity() == 0 {
                record = Vec::with_capacity(lengths[record_index]);
            }
            for block in fragment.bytes.chunks(1024 * 1024) {
                cancellation_checkpoint(cancellation)?;
                record.extend_from_slice(block);
            }
            if fragment.completes_record {
                output
                    .last_mut()
                    .expect("validated stream")
                    .records
                    .push(std::mem::take(&mut record));
                record_index += 1;
            }
        }
    }
    validate_streams(&output)?;
    Ok(output)
}
fn invalid(reason: &'static str) -> MeshingContractError {
    MeshingContractError::invalid("meshing stream reconstruction", reason)
}
fn add(a: u64, b: u64) -> Result<u64, MeshingContractError> {
    a.checked_add(b)
        .ok_or_else(|| invalid("reconstruction storage overflow"))
}
#[cfg(test)]
mod tests;
