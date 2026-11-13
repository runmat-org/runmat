use runmat_analysis_fea::execution::AnalysisObjectManifest;
use runmat_execution::Digest;
#[cfg(test)]
mod tests;
pub(super) fn read_payload_partition(
    manifest: &AnalysisObjectManifest,
    logical_identity: Digest,
    read: &mut impl FnMut(Digest) -> Result<Option<Vec<u8>>, String>,
) -> Result<Vec<u8>, String> {
    let partition = manifest
        .partitions
        .iter()
        .find(|partition| partition.logical_identity == logical_identity)
        .ok_or_else(|| {
            "recovered field partition is absent from its object manifest".to_string()
        })?;
    let first = partition.first_chunk_ordinal as usize;
    let end = first
        .checked_add(partition.chunk_count as usize)
        .ok_or_else(|| "recovered field chunk interval overflow".to_string())?;
    let chunks = manifest
        .chunks
        .get(first..end)
        .ok_or_else(|| "recovered field chunk interval exceeds its manifest".to_string())?;
    let capacity = usize::try_from(partition.encoded_length)
        .map_err(|_| "recovered field partition length is not representable".to_string())?;
    let mut payload = Vec::with_capacity(capacity);
    for chunk in chunks {
        let bytes = read(chunk.digest)?.ok_or_else(|| {
            format!(
                "recovered field partition is missing chunk {}",
                chunk.ordinal
            )
        })?;
        super::validate_chunk(chunk, &bytes)?;
        payload.extend_from_slice(&bytes);
    }
    if payload.len() as u64 != partition.encoded_length
        || Digest::sha256(&payload) != partition.payload_digest
    {
        return Err("recovered field partition differs from its object manifest".into());
    }
    Ok(payload)
}
