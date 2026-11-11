//! Numerical payload ownership transfer into the shared artifact inventory.
use super::*;
use runmat_analysis_fea::execution::{AnalysisPayloadPlan, AnalysisTransportProfile};
use runmat_execution_artifact::object::inventory_validation_scratch_bytes;
use std::fmt::Write;

const ROOT_PREFIX: &str = "analysis/manifests";
const CHUNK_PREFIX: &str = "analysis/chunks/";
const ORDINAL_CAPACITY: usize = 10; // All u32 ordinals, including the eight-digit padding.
const DIGEST_SUFFIX_BYTES: usize = 65; // Slash and raw SHA-256 hex (no algorithm prefix).
const HEX: &[u8; 16] = b"0123456789abcdef";

pub(super) fn matches_root_name(name: &str, digest: Digest) -> bool {
    name.strip_prefix(ROOT_PREFIX)
        .and_then(|suffix| suffix.strip_prefix('/'))
        .is_some_and(|suffix| matches_digest_suffix(suffix, digest))
}

pub(super) fn matches_chunk_name(name: &str, ordinal: u32, digest: Digest) -> bool {
    let Some((number, suffix)) = name
        .strip_prefix(CHUNK_PREFIX)
        .and_then(|value| value.split_once('/'))
    else {
        return false;
    };
    let width = ordinal
        .checked_ilog10()
        .map_or(1, |value| value as usize + 1)
        .max(8);
    number.len() == width
        && number.bytes().all(|byte| byte.is_ascii_digit())
        && number.parse::<u32>() == Ok(ordinal)
        && matches_digest_suffix(suffix, digest)
}

fn matches_digest_suffix(suffix: &str, digest: Digest) -> bool {
    suffix.len() == 64
        && suffix
            .as_bytes()
            .chunks_exact(2)
            .zip(digest.bytes())
            .all(|(pair, byte)| {
                pair[0] == HEX[(byte >> 4) as usize] && pair[1] == HEX[(byte & 15) as usize]
            })
}

/// Requested builder and artifact-conversion heap, excluding caller-owned source fields,
/// their canonical encoding, allocator overhead and process/native memory. Conservative
/// overlap includes the root encoding/materialization and both descriptor inventories.
pub fn projected_chunked_output_resource_facts(
    plan: &AnalysisPayloadPlan,
) -> AnalysisBridgeResult<PreparedOutputResourceFacts> {
    projected_conversion_facts(plan.transport_profile(), plan.resident_bytes()?)
}

/// Pre-solve shape projection, before partition identities exist. Planned entries must use the
/// exact-reserved inventory; the post-solve plan variant additionally counts excess capacity.
pub fn projected_transport_output_resource_facts(
    profile: &AnalysisTransportProfile,
) -> AnalysisBridgeResult<PreparedOutputResourceFacts> {
    let count = usize::try_from(profile.partition_count).map_err(|_| overflow())?;
    projected_conversion_facts(profile, profile.builder_resident_bytes(count)?)
}

fn projected_conversion_facts(
    profile: &AnalysisTransportProfile,
    builder_bytes: u64,
) -> AnalysisBridgeResult<PreparedOutputResourceFacts> {
    let count = profile
        .chunk_count_upper_bound
        .checked_add(1)
        .ok_or_else(overflow)?;
    let chunk_strings = CHUNK_PREFIX.len()
        + ORDINAL_CAPACITY
        + DIGEST_SUFFIX_BYTES
        + ANALYSIS_CHUNK_MEDIA_TYPE.len();
    let root_strings = ROOT_PREFIX.len() + DIGEST_SUFFIX_BYTES + ANALYSIS_MANIFEST_MEDIA_TYPE.len();
    let components = [
        builder_bytes,
        // Optional kernel handoff boxes the typed envelope, never the already owned buffers.
        std::mem::size_of::<PreparedAnalysisObjects>() as u64,
        count
            .checked_mul(std::mem::size_of::<LogicalObject>() as u64)
            .ok_or_else(overflow)?,
        profile
            .chunk_count_upper_bound
            .checked_mul(chunk_strings as u64)
            .ok_or_else(overflow)?,
        (2 * root_strings + CHUNK_PREFIX.len() + ORDINAL_CAPACITY) as u64,
        profile.root.encoded_bytes_upper_bound,
        profile.root.materialization_resident_bytes_estimate,
        inventory_validation_scratch_bytes(usize::try_from(count).map_err(|_| overflow())?)?,
    ];
    let resident = components.into_iter().try_fold(0_u64, |sum, value| {
        sum.checked_add(value).ok_or_else(overflow)
    })?;
    Ok(PreparedOutputResourceFacts {
        encoded_payload_bytes_upper_bound: ByteCount(profile.payload_bytes_upper_bound),
        transport_closure_bytes_upper_bound: ByteCount(profile.total_bytes_upper_bound),
        construction_resident_bytes_admission_budget: ByteCount(resident),
    })
}

pub(crate) fn prepare_chunked_analysis_objects(
    chunked: ChunkedAnalysisPayload,
    namespace: ObjectNamespace,
    limits: ObjectInventoryLimits,
) -> AnalysisBridgeResult<PreparedAnalysisObjects> {
    if chunked.chunks.len() != chunked.manifest.chunks.len() {
        return Err(AnalysisBridgeError::Identity(
            "analysis chunk inventory differs from its manifest",
        ));
    }
    let count = chunked.chunks.len().checked_add(1).ok_or_else(overflow)?;
    // A run-data payload is admitted against the run's inventory budget before its objects are
    // built: the refusal names the requirement and the admitted allowance.
    let manifest_bytes = chunked.manifest.canonical_bytes()?;
    let (mut total, mut largest) = (manifest_bytes.len() as u64, manifest_bytes.len() as u64);
    for chunk in &chunked.chunks {
        total = total.checked_add(chunk.len() as u64).ok_or_else(overflow)?;
        largest = largest.max(chunk.len() as u64);
    }
    limits.require_projection("analysis payload", count as u64, largest, total)?;
    let mut objects = Vec::new();
    objects.try_reserve_exact(count).map_err(|_| overflow())?;
    let manifest_object = logical_object(
        namespace,
        ROOT_PREFIX,
        ANALYSIS_MANIFEST_MEDIA_TYPE,
        manifest_bytes,
    )?;
    let descriptor = &manifest_object.descriptor;
    let root = ObjectDescriptor {
        namespace: descriptor.namespace,
        logical_name: owned_string(&descriptor.logical_name)?,
        digest: descriptor.digest,
        encoded_length: descriptor.encoded_length,
        media_type: owned_string(&descriptor.media_type)?,
    };
    objects.push(manifest_object);
    for (descriptor, bytes) in chunked.manifest.chunks.iter().zip(chunked.chunks) {
        if descriptor.digest != Digest::sha256(&bytes)
            || descriptor.encoded_length != bytes.len() as u64
        {
            return Err(AnalysisBridgeError::Identity(
                "analysis chunk differs from its manifest descriptor",
            ));
        }
        let mut prefix = reserved_string(CHUNK_PREFIX.len() + ORDINAL_CAPACITY)?;
        prefix.push_str(CHUNK_PREFIX);
        write!(prefix, "{:08}", descriptor.ordinal).map_err(|_| overflow())?;
        objects.push(logical_object(
            namespace,
            &prefix,
            ANALYSIS_CHUNK_MEDIA_TYPE,
            bytes,
        )?);
    }
    validate_inventory(&objects, limits)?;
    Ok(PreparedAnalysisObjects {
        manifest: chunked.manifest,
        root,
        objects,
    })
}

pub(super) fn logical_object(
    namespace: ObjectNamespace,
    prefix: &str,
    media_type: &str,
    bytes: Vec<u8>,
) -> AnalysisBridgeResult<LogicalObject> {
    let digest = Digest::sha256(&bytes);
    let mut name = reserved_string(
        prefix
            .len()
            .checked_add(DIGEST_SUFFIX_BYTES)
            .ok_or_else(overflow)?,
    )?;
    name.push_str(prefix);
    name.push('/');
    for byte in digest.bytes() {
        name.push(HEX[(byte >> 4) as usize] as char);
        name.push(HEX[(byte & 15) as usize] as char);
    }
    Ok(LogicalObject::new(
        namespace,
        name,
        owned_string(media_type)?,
        bytes,
    )?)
}

fn reserved_string(capacity: usize) -> AnalysisBridgeResult<String> {
    let mut value = String::new();
    value.try_reserve_exact(capacity).map_err(|_| overflow())?;
    Ok(value)
}

fn owned_string(value: &str) -> AnalysisBridgeResult<String> {
    let mut owned = reserved_string(value.len())?;
    owned.push_str(value);
    Ok(owned)
}

fn overflow() -> AnalysisBridgeError {
    AnalysisBridgeError::Invalid(
        "numerical object conversion allocation exceeds available capacity".into(),
    )
}

/// Ordered imported occurrences retain separate owned buffers even when digests repeat.
pub(super) fn import_transport_metadata_bytes(count: usize) -> AnalysisBridgeResult<u64> {
    let strings = CHUNK_PREFIX.len()
        + ORDINAL_CAPACITY
        + DIGEST_SUFFIX_BYTES
        + ANALYSIS_CHUNK_MEDIA_TYPE.len();
    let slots = count
        .checked_mul(std::mem::size_of::<LogicalObject>() + strings)
        .ok_or_else(overflow)?;
    // Root descriptor clone and root naming/media; count includes its object slot.
    let root = std::mem::size_of::<ObjectDescriptor>()
        + 2 * (ROOT_PREFIX.len() + DIGEST_SUFFIX_BYTES + ANALYSIS_MANIFEST_MEDIA_TYPE.len());
    u64::try_from(slots.checked_add(root).ok_or_else(overflow)?).map_err(|_| overflow())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn logical_names_preserve_raw_digest_wire_identity() {
        let object = logical_object(
            ObjectNamespace::ResultValue,
            "analysis/chunks/00000003",
            ANALYSIS_CHUNK_MEDIA_TYPE,
            b"abc".to_vec(),
        )
        .unwrap();
        assert_eq!(object.descriptor.logical_name,
            "analysis/chunks/00000003/ba7816bf8f01cfea414140de5dae2223b00361a396177a9cb410ff61f20015ad");
        object.validate().unwrap();
    }

    #[test]
    fn conversion_rejects_extra_and_missing_chunk_buffers() {
        let chunked = chunk_numerical_payload(
            AnalysisNumericalObjectKind::SolverResult,
            Digest::sha256(b"test"),
            b"abc",
            2,
        )
        .unwrap();
        let mut missing = chunked.clone();
        missing.chunks.pop();
        let mut extra = chunked;
        extra.chunks.push(vec![0]);
        for invalid in [missing, extra] {
            assert!(prepare_chunked_analysis_objects(
                invalid,
                ObjectNamespace::ResultValue,
                ObjectInventoryLimits::default()
            )
            .is_err());
        }
    }

    #[test]
    fn payload_beyond_the_admitted_inventory_is_refused_with_requirement_and_budget() {
        let payload = vec![7_u8; 4096];
        let chunked = chunk_numerical_payload(
            AnalysisNumericalObjectKind::ObservationEvaluationRequest,
            Digest::sha256(b"request"),
            &payload,
            1024,
        )
        .unwrap();
        let fits = prepare_chunked_analysis_objects(
            chunked.clone(),
            ObjectNamespace::InputValue,
            ObjectInventoryLimits::admitted(1 << 20),
        )
        .unwrap();
        let required: u64 = fits
            .objects
            .iter()
            .map(|o| o.descriptor.encoded_length)
            .sum();
        prepare_chunked_analysis_objects(
            chunked.clone(),
            ObjectNamespace::InputValue,
            ObjectInventoryLimits::admitted(required),
        )
        .unwrap();
        let error = prepare_chunked_analysis_objects(
            chunked,
            ObjectNamespace::InputValue,
            ObjectInventoryLimits::admitted(required - 1),
        )
        .unwrap_err();
        match error {
            AnalysisBridgeError::InventoryCapacity(capacity) => {
                assert_eq!(
                    (capacity.required, capacity.maximum),
                    (required, required - 1)
                );
                assert!(capacity.to_string().contains(&required.to_string()));
            }
            other => panic!("expected an inventory capacity refusal, got {other:?}"),
        }
    }
}
