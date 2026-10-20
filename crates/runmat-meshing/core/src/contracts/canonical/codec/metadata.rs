//! Domain validation scratch is supplied by the typed owner; codec profiling
//! counts the actual validated CBOR structure, never a wire-size multiplier.
use super::*;
pub(super) fn decode<T: CanonicalMeshingContract>(
    bytes: &[u8],
    control: &dyn runmat_canonical_codec::CanonicalCodecControl,
    admit: impl FnMut(u64) -> Result<(), MeshingContractError>,
    validation_scratch: impl FnOnce(&T) -> Result<u64, MeshingContractError>,
) -> Result<T, MeshingContractError> {
    decode_validated(
        bytes,
        control,
        admit,
        validation_scratch,
        T::validate_canonical,
    )
}
/// Requested storage for directly decoding a canonical meshing value. Typed decoding builds
/// no JSON tree. A compact solver topology decodes its columns at exact length (the typed
/// wire estimate) and then rebuilds its records, whose added storage is inventoried from
/// the wire. Other contracts retain three typed wire estimates for decoder growth slack.
pub(crate) fn topology_decode_admission(
    value: &[u8],
    control: &dyn runmat_canonical_codec::CanonicalCodecControl,
) -> Result<u64, runmat_canonical_codec::CanonicalCodecError> {
    let typed =
        runmat_canonical_codec::canonical_value_typed_resident_bytes_estimate(value, control)?;
    let overflow = || runmat_canonical_codec::CanonicalCodecError {
        kind: runmat_canonical_codec::CanonicalCodecErrorKind::ResourceLimit,
        field: "meshing decode admission",
        reason: "byte count overflow".into(),
    };
    match super::super::topology_wire::rebuilt_record_bytes(value, control)? {
        Some(records) => typed.checked_add(records).ok_or_else(overflow),
        None => typed.checked_mul(3).ok_or_else(overflow),
    }
}

pub(crate) fn decode_validated<T: CanonicalMeshingContract>(
    bytes: &[u8],
    control: &dyn runmat_canonical_codec::CanonicalCodecControl,
    mut admit: impl FnMut(u64) -> Result<(), MeshingContractError>,
    validation_scratch: impl FnOnce(&T) -> Result<u64, MeshingContractError>,
    validate: impl FnOnce(&T) -> Result<(), MeshingContractError>,
) -> Result<T, MeshingContractError> {
    let mut denied = None;
    let decoded = runmat_canonical_codec::decode_contract_with_admission(
        CODEC_PREFIX,
        T::DOMAIN,
        bytes,
        T::LIMITS.into(),
        control,
        |value| {
            let requested = topology_decode_admission(value, control)?;
            admit(requested).map_err(|error| {
                let detail = error.to_string();
                denied = Some(error);
                runmat_canonical_codec::CanonicalCodecError {
                    kind: runmat_canonical_codec::CanonicalCodecErrorKind::ResourceLimit,
                    field: "meshing metadata admission",
                    reason: detail,
                }
            })
        },
    );
    if let Some(error) = denied {
        return Err(error);
    }
    let decoded: T = decoded.map_err(map_codec_error)?;
    admit(validation_scratch(&decoded)?)?;
    validate(&decoded)?;
    Ok(decoded)
}
