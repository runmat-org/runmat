//! Borrowed single-face transport shares the production partition representation.
use super::*;
use crate::ExactFaceMesh;

pub(crate) fn encode_accepted_face(
    mesh: &ExactFaceMesh,
    maximum_encoded_bytes: usize,
    cancellation: &dyn MeshingCancellationSignal,
) -> Result<Vec<u8>, ExactSurfaceMeshError> {
    PartitionEncoding::single(mesh, cancellation)?
        .encode(limits(maximum_encoded_bytes), cancellation)
}
pub(crate) fn profile_accepted_face(
    mesh: &ExactFaceMesh,
    maximum_encoded_bytes: usize,
    cancellation: &dyn MeshingCancellationSignal,
) -> Result<Option<runmat_canonical_codec::CanonicalEncodingProfile>, ExactSurfaceMeshError> {
    let encoded = PartitionEncoding::single(mesh, cancellation)?;
    encoded.bounded_combined_profile(limits(maximum_encoded_bytes), cancellation)
}
fn limits(maximum_encoded_bytes: usize) -> CanonicalLimits {
    CanonicalLimits {
        maximum_encoded_bytes: maximum_encoded_bytes.min(CONTRACT_LIMITS.maximum_encoded_bytes),
        ..CONTRACT_LIMITS
    }
}

/// Reconstruct the original face and independently validate current boundary provenance.
pub fn decode_exact_face_reuse_record_with_cancellation(
    bytes: &[u8],
    topology: &runmat_geometry_core::ExactBRepTopology,
    curves: &runmat_meshing_curve::SharedCurveMesh,
    boundary: &crate::ExactFaceBoundary,
    cancellation: &dyn MeshingCancellationSignal,
) -> Result<ExactFaceMesh, ExactSurfaceMeshError> {
    let control = MeshingCanonicalCodecControl::new(cancellation);
    let decoded: crate::ExactFacePartitionResult = if bytes.starts_with(indexed::PREFIX) {
        indexed::decode(bytes, CONTRACT_LIMITS, cancellation)?
    } else {
        runmat_canonical_codec::decode_contract_with_control(
            CODEC_PREFIX,
            CONTRACT_DOMAIN,
            bytes,
            CONTRACT_LIMITS,
            &control,
        )
        .map_err(|error| controlled_codec_error(error, &control))?
    };
    let crate::ExactFacePartitionOutcome::Converged { mut faces } = decoded.outcome else {
        return Err(super::super::invalid("retained face is unconverged"));
    };
    if decoded.schema_version != crate::EXACT_FACE_PARTITION_RESULT_SCHEMA_VERSION
        || faces.len() != 1
    {
        return Err(super::super::invalid(
            "face reuse schema or inventory differs",
        ));
    }
    let mesh = faces.pop().expect("one original face checked");
    if mesh.source_face_id != boundary.source_face_id
        || decoded.partition != indexed::single_partition(&mesh)
    {
        return Err(super::super::invalid(
            "face reuse record differs from its one original face",
        ));
    }
    crate::face_mesh::validate_exact_face_mesh_contract_with_cancellation(
        &mesh,
        topology,
        Some(cancellation),
    )
    .map_err(|error| {
        ExactSurfaceMeshError::new(
            if error.kind == crate::ExactFaceJoinErrorKind::Cancelled {
                ExactSurfaceMeshErrorKind::Cancelled
            } else {
                ExactSurfaceMeshErrorKind::InvalidInput
            },
            error.to_string(),
        )
    })?;
    super::super::boundary::validate_face_curve_boundary(
        &mesh,
        boundary,
        topology,
        curves,
        Some(cancellation),
    )?;
    Ok(mesh)
}
