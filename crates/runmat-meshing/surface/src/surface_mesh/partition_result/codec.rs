//! One canonical wire form for pure and cooperatively controlled partition artifacts.
use super::{
    validate_exact_face_partition_result, validate_exact_face_partition_result_with_cancellation,
};
use crate::surface_mesh::indexed_face::partition::{self as indexed, PartitionEncoding};
use crate::{ExactFacePartitionResult, ExactSurfaceMeshError, ExactSurfaceMeshErrorKind};
use runmat_canonical_codec::{CanonicalCodecError, CanonicalLimits};
use runmat_geometry_core::ExactBRepTopology;
use runmat_meshing_core::NeverCancelled;
use runmat_meshing_core::{MeshingCancellationSignal, MeshingCanonicalCodecControl};
use runmat_meshing_curve::SharedCurveMesh;
const CODEC_PREFIX: &[u8] = b"runmat-meshing-face-partition-result-canonical-cbor/v1\0";
const CONTRACT_DOMAIN: &str = "exact-face-partition-result/v1";
const CONTRACT_LIMITS: CanonicalLimits =
    CanonicalLimits::platform_content(1024 * 1024, 64);

mod face;
pub use face::decode_exact_face_reuse_record_with_cancellation;
pub(crate) use face::{encode_accepted_face, profile_accepted_face};

pub fn encode_exact_face_partition_result(
    result: &ExactFacePartitionResult,
    topology: &ExactBRepTopology,
    curves: &SharedCurveMesh,
) -> Result<Vec<u8>, ExactSurfaceMeshError> {
    validate_exact_face_partition_result(result, topology, curves)?;
    PartitionEncoding::new(result, &NeverCancelled)?.encode(CONTRACT_LIMITS, &NeverCancelled)
}

pub fn decode_exact_face_partition_result(
    bytes: &[u8],
    topology: &ExactBRepTopology,
    curves: &SharedCurveMesh,
) -> Result<ExactFacePartitionResult, ExactSurfaceMeshError> {
    decode_with_limits(bytes, topology, curves, CONTRACT_LIMITS)
}

fn decode_with_limits(
    bytes: &[u8],
    topology: &ExactBRepTopology,
    curves: &SharedCurveMesh,
    limits: CanonicalLimits,
) -> Result<ExactFacePartitionResult, ExactSurfaceMeshError> {
    let result = if bytes.starts_with(indexed::PREFIX) {
        indexed::decode(bytes, limits, &NeverCancelled)?
    } else {
        runmat_canonical_codec::decode_contract(CODEC_PREFIX, CONTRACT_DOMAIN, bytes, limits)
            .map_err(map_codec_error)?
    };
    validate_exact_face_partition_result(&result, topology, curves)?;
    Ok(result)
}

#[cfg(test)]
pub(crate) fn decode_exact_face_partition_result_with_byte_limit(
    bytes: &[u8],
    topology: &ExactBRepTopology,
    curves: &SharedCurveMesh,
    maximum_encoded_bytes: usize,
) -> Result<ExactFacePartitionResult, ExactSurfaceMeshError> {
    decode_with_limits(
        bytes,
        topology,
        curves,
        CanonicalLimits::content_bounded(maximum_encoded_bytes, 1024 * 1024, 64),
    )
}

fn map_codec_error(error: CanonicalCodecError) -> ExactSurfaceMeshError {
    ExactSurfaceMeshError::new(
        ExactSurfaceMeshErrorKind::InvalidEncoding,
        format!("{}: {}", error.field, error.reason),
    )
}

pub fn encode_exact_face_partition_result_with_cancellation(
    result: &ExactFacePartitionResult,
    topology: &ExactBRepTopology,
    curves: &SharedCurveMesh,
    cancellation: &dyn MeshingCancellationSignal,
) -> Result<Vec<u8>, ExactSurfaceMeshError> {
    validate_exact_face_partition_result_with_cancellation(result, topology, curves, cancellation)?;
    PartitionEncoding::new(result, cancellation)?.encode(CONTRACT_LIMITS, cancellation)
}

pub fn decode_exact_face_partition_result_with_cancellation(
    bytes: &[u8],
    topology: &ExactBRepTopology,
    curves: &SharedCurveMesh,
    cancellation: &dyn MeshingCancellationSignal,
) -> Result<ExactFacePartitionResult, ExactSurfaceMeshError> {
    let codec_control = MeshingCanonicalCodecControl::new(cancellation);
    let result = if bytes.starts_with(indexed::PREFIX) {
        indexed::decode(bytes, CONTRACT_LIMITS, cancellation)?
    } else {
        runmat_canonical_codec::decode_contract_with_control(
            CODEC_PREFIX,
            CONTRACT_DOMAIN,
            bytes,
            CONTRACT_LIMITS,
            &codec_control,
        )
        .map_err(|error| controlled_codec_error(error, &codec_control))?
    };
    validate_exact_face_partition_result_with_cancellation(
        &result,
        topology,
        curves,
        cancellation,
    )?;
    Ok(result)
}

fn controlled_codec_error(
    error: CanonicalCodecError,
    control: &MeshingCanonicalCodecControl<'_>,
) -> ExactSurfaceMeshError {
    if control.observed_cancellation() {
        ExactSurfaceMeshError::new(ExactSurfaceMeshErrorKind::Cancelled, error.to_string())
    } else {
        map_codec_error(error)
    }
}

/// Forge an untrusted canonical payload so cold admission is tested independently
/// of the producer's semantic checks. This helper is absent from production.
#[cfg(test)]
pub(crate) fn encode_unadmitted_for_test(result: &ExactFacePartitionResult) -> Vec<u8> {
    PartitionEncoding::new(result, &NeverCancelled)
        .unwrap()
        .encode(CONTRACT_LIMITS, &NeverCancelled)
        .unwrap()
}

/// Profile bounded encoder workspace without copying the mesh or granting semantic authority.
/// Encoding still validates the complete partition against its prerequisites.
pub fn profile_exact_face_partition_result_encoding(
    result: &ExactFacePartitionResult,
    cancellation: &dyn MeshingCancellationSignal,
) -> Result<runmat_canonical_codec::CanonicalEncodingProfile, ExactSurfaceMeshError> {
    PartitionEncoding::new(result, cancellation)?.profile(CONTRACT_LIMITS, cancellation)
}

/// Charges typed reconstruction before any owned decoder collections are created.
/// The caller supplies a scope that outlives the returned original meshes.
pub fn decode_exact_face_partition_result_admitted(
    bytes: &[u8],
    topology: &ExactBRepTopology,
    curves: &SharedCurveMesh,
    cancellation: &dyn MeshingCancellationSignal,
    allocations: &dyn runmat_geometry_core::GeometryEvaluationControl,
) -> Result<ExactFacePartitionResult, ExactSurfaceMeshError> {
    let mut allocation_failure = None;
    let result = if bytes.starts_with(indexed::PREFIX) {
        indexed::decode_admitted(bytes, CONTRACT_LIMITS, cancellation, |requested| {
            allocations
                .consume_allocation_bytes(requested)
                .map_err(|failure| {
                    let reason = failure.to_string();
                    allocation_failure = Some(failure);
                    runmat_canonical_codec::CanonicalCodecError {
                        kind: runmat_canonical_codec::CanonicalCodecErrorKind::ResourceLimit,
                        field: "surface typed reconstruction",
                        reason,
                    }
                })
        })
    } else {
        let control = MeshingCanonicalCodecControl::new(cancellation);
        let requested = runmat_canonical_codec::canonical_chunked_payload_materialization_resident_bytes_estimate([bytes], &control)
            .map_err(|error| controlled_codec_error(error, &control))?;
        allocations
            .consume_allocation_bytes(requested)
            .map_err(|failure| {
                allocation_failure = Some(failure.clone());
                ExactSurfaceMeshError::new(
                    ExactSurfaceMeshErrorKind::ResourceLimit,
                    failure.to_string(),
                )
            })
            .and_then(|()| {
                decode_exact_face_partition_result_with_cancellation(
                    bytes,
                    topology,
                    curves,
                    cancellation,
                )
            })
    };
    if let Some(failure) = allocation_failure {
        return Err(ExactSurfaceMeshError::new(
            if failure.kind == runmat_geometry_core::GeometryEvaluationErrorKind::Cancelled {
                ExactSurfaceMeshErrorKind::Cancelled
            } else {
                ExactSurfaceMeshErrorKind::ResourceLimit
            },
            failure.to_string(),
        ));
    }
    let result = result?;
    validate_exact_face_partition_result_with_cancellation(
        &result,
        topology,
        curves,
        cancellation,
    )?;
    Ok(result)
}
