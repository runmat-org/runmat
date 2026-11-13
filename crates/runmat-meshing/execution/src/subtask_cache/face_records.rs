//! Bounded face-record persistence. Direct entries and chunk manifests retain the
//! same numerical lookup key; transport never grants accepted-face authority.
use super::*;
use super::{record, record_chunks};
use runmat_geometry_core::GeometryAllocationReservation;
use serde::{Deserialize, Serialize};
const PREFIX: &[u8] = b"runmat-meshing-face-cache-pointer/v1\0";
const DOMAIN: &str = "meshing-face-cache-pointer/v1";
const LIMITS: runmat_canonical_codec::CanonicalLimits =
    runmat_canonical_codec::CanonicalLimits::new(4096, 512, 512, 16);
#[derive(Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
struct Pointer {
    schema_version: u16,
    numerical_key: StableDigest,
    manifest: StableDigest,
}

pub(super) fn retain(
    store: &dyn MeshingSubtaskCacheStore,
    key: StableDigest,
    bytes: &[u8],
    materialization: u64,
    control: &dyn GeometryEvaluationControl,
) -> Result<bool, MeshingSubtaskCacheError> {
    if bytes.len() as u64 <= store.maximum_entry_bytes() {
        return store.write(key, bytes, materialization, control);
    }
    let Some(manifest) =
        record_chunks::retain(record_chunks::FACE, store, bytes, materialization, control)?
    else {
        return Ok(false);
    };
    let pointer = Pointer {
        schema_version: 1,
        numerical_key: key,
        manifest,
    };
    let codec = crate::codec_control::CodecControl::new(control);
    let profile =
        runmat_canonical_codec::profile_canonical_encoding_with_control(&pointer, LIMITS, &codec)
            .map_err(|error| codec.error(error))?;
    let _storage =
        GeometryAllocationReservation::new(control, profile.resident_workspace_bytes_estimate)
            .map_err(MeshingSubtaskCacheError::Control)?;
    let mut bytes = runmat_canonical_codec::encode_contract_with_control(
        PREFIX, DOMAIN, &pointer, LIMITS, &codec,
    )
    .map_err(|error| codec.error(error))?;
    let workspace =
        runmat_canonical_codec::canonical_chunked_payload_materialization_resident_bytes_estimate(
            [bytes.as_slice()],
            &codec,
        )
        .map_err(|error| codec.error(error))?;
    record::frame(&mut bytes, key);
    // Publish the lookup pointer last. Interrupted/orphaned chunks remain a miss.
    store.write(key, &bytes, workspace, control)
}

pub(super) fn read<'a>(
    store: &dyn MeshingSubtaskCacheStore,
    key: StableDigest,
    control: &'a dyn GeometryEvaluationControl,
    inventory: impl FnOnce(&[u8]) -> Result<u64, MeshingSubtaskCacheError>,
) -> Result<Option<MeshingSubtaskCacheEntry<'a>>, MeshingSubtaskCacheError> {
    let Some(entry) = store.read(key, control)? else {
        return Ok(None);
    };
    if entry.bytes.len() as u64 > store.maximum_entry_bytes() {
        return Err(MeshingSubtaskCacheError::Storage(
            "face lookup exceeds its per-entry byte bound".into(),
        ));
    }
    let body = record::body(&entry.bytes, key)?;
    if !body.starts_with(PREFIX) {
        return Ok(Some(entry));
    }
    let codec = crate::codec_control::CodecControl::new(control);
    let required =
        runmat_canonical_codec::canonical_chunked_payload_materialization_resident_bytes_estimate(
            [body],
            &codec,
        )
        .map_err(|error| codec.error(error))?;
    if required > entry.materialization_resident_bytes {
        return Err(MeshingSubtaskCacheError::Storage(
            "face lookup understates its decoded storage".into(),
        ));
    }
    let pointer: Pointer =
        runmat_canonical_codec::decode_contract_with_control(PREFIX, DOMAIN, body, LIMITS, &codec)
            .map_err(|error| codec.error(error))?;
    if pointer.schema_version != 1 || pointer.numerical_key != key {
        return Err(MeshingSubtaskCacheError::Storage(
            "face lookup belongs to another numerical identity".into(),
        ));
    }
    drop(entry);
    record_chunks::read(
        record_chunks::FACE,
        store,
        pointer.manifest,
        control,
        |bytes| {
            let body = record::body(bytes, key)?;
            inventory(body)
        },
    )
}

#[cfg(test)]
mod tests;
