//! Preserve stage-specific validation identities without copying whole publications.
use std::borrow::Cow;

use runmat_execution::Digest;
use runmat_meshing_core::{MeshingChunkStream, MeshingFailure, StableDigest};

use crate::{MeshingStageCheckpoint, MeshingStageControl};

#[cfg(test)]
mod tests;

pub(crate) fn record(
    domain: &[u8],
    encoded: &[u8],
    control: &mut MeshingStageControl<'_>,
    checkpoint: &MeshingStageCheckpoint,
) -> Result<StableDigest, Box<MeshingFailure>> {
    hash([domain, encoded], control, checkpoint)
}

pub(crate) fn streams(
    domain: &[u8],
    streams: &[MeshingChunkStream],
    control: &mut MeshingStageControl<'_>,
    checkpoint: &MeshingStageCheckpoint,
) -> Result<StableDigest, Box<MeshingFailure>> {
    let parts = std::iter::once(Cow::Borrowed(domain)).chain(streams.iter().flat_map(|stream| {
        std::iter::once(Cow::Borrowed(stream.media_type.media_type().as_bytes()))
            .chain(std::iter::once(Cow::Owned(
                stream.schema_version.to_be_bytes().to_vec(),
            )))
            .chain(stream.records.iter().flat_map(|record| {
                [
                    Cow::Owned((record.len() as u64).to_be_bytes().to_vec()),
                    Cow::Borrowed(record.as_slice()),
                ]
            }))
    }));
    hash(parts, control, checkpoint)
}

fn hash(
    parts: impl IntoIterator<Item = impl AsRef<[u8]>>,
    control: &mut MeshingStageControl<'_>,
    checkpoint: &MeshingStageCheckpoint,
) -> Result<StableDigest, Box<MeshingFailure>> {
    Digest::sha256_parts_with_checkpoint(parts, || control.checkpoint(checkpoint.clone()))
        .map(|digest| StableDigest::from_bytes(*digest.bytes()))
}
