use super::indexed_face::surface::{self as indexed, SurfaceEncoding};
use runmat_canonical_codec::CanonicalLimits;
use runmat_geometry_core::ExactBRepTopology;
use runmat_meshing_core::MeshingCancellationSignal;
use runmat_meshing_curve::SharedCurveMesh;

use super::{
    validate_exact_surface_mesh, validate_exact_surface_pass_result,
    validate_published_exact_surface_mesh, ExactFaceMeshBatch, ExactFacePartitionResult,
    ExactSurfaceJoinOptions, ExactSurfaceMesh, ExactSurfaceMeshError, ExactSurfaceMeshErrorKind,
    ExactSurfacePassOutcome, ExactSurfacePassResult,
};

const CODEC_PREFIX: &[u8] = b"runmat-meshing-exact-surface-canonical-cbor/v1\0";
const CONTRACT_DOMAIN: &str = "exact-surface-mesh/v1";
const CONTRACT_LIMITS: CanonicalLimits = CanonicalLimits::platform_content(1024 * 1024, 64);

pub fn encode_exact_surface_mesh(
    mesh: &ExactSurfaceMesh,
    topology: &ExactBRepTopology,
    batches: &[ExactFaceMeshBatch],
    options: ExactSurfaceJoinOptions,
) -> Result<Vec<u8>, ExactSurfaceMeshError> {
    validate_exact_surface_mesh(mesh, topology, batches, options)?;
    encode_indexed(mesh, &runmat_meshing_core::NeverCancelled)
}

pub fn decode_exact_surface_mesh(
    bytes: &[u8],
    topology: &ExactBRepTopology,
    batches: &[ExactFaceMeshBatch],
    options: ExactSurfaceJoinOptions,
) -> Result<ExactSurfaceMesh, ExactSurfaceMeshError> {
    decode_with_limits(bytes, topology, batches, options, CONTRACT_LIMITS)
}

/// Decodes the final exact-surface publication without requiring its transient face partitions.
/// The final artifact is independently checked against exact topology before downstream use.
pub fn decode_published_exact_surface_mesh(
    bytes: &[u8],
    topology: &ExactBRepTopology,
    options: ExactSurfaceJoinOptions,
) -> Result<ExactSurfaceMesh, ExactSurfaceMeshError> {
    let mesh =
        decode_indexed_or_legacy(bytes, CONTRACT_LIMITS, &runmat_meshing_core::NeverCancelled)?;
    validate_published_exact_surface_mesh(&mesh, topology, options)?;
    Ok(mesh)
}

pub fn encode_exact_surface_mesh_from_pass(
    pass: &ExactSurfacePassResult,
    topology: &ExactBRepTopology,
    curves: &SharedCurveMesh,
    partitions: &[ExactFacePartitionResult],
    options: ExactSurfaceJoinOptions,
) -> Result<Vec<u8>, ExactSurfaceMeshError> {
    validate_exact_surface_pass_result(pass, topology, curves, partitions, options)?;
    let ExactSurfacePassOutcome::Converged { surface } = &pass.outcome else {
        return Err(ExactSurfaceMeshError::new(
            ExactSurfaceMeshErrorKind::InvalidInput,
            "a curve-restart pass cannot publish a final exact surface",
        ));
    };
    encode_indexed(surface, &runmat_meshing_core::NeverCancelled)
}

pub fn decode_exact_surface_mesh_from_pass(
    bytes: &[u8],
    pass: &ExactSurfacePassResult,
    topology: &ExactBRepTopology,
    curves: &SharedCurveMesh,
    partitions: &[ExactFacePartitionResult],
    options: ExactSurfaceJoinOptions,
) -> Result<ExactSurfaceMesh, ExactSurfaceMeshError> {
    validate_exact_surface_pass_result(pass, topology, curves, partitions, options)?;
    if !matches!(pass.outcome, ExactSurfacePassOutcome::Converged { .. }) {
        return Err(ExactSurfaceMeshError::new(
            ExactSurfaceMeshErrorKind::InvalidInput,
            "a curve-restart pass cannot admit a final exact surface",
        ));
    }
    let decoded =
        decode_indexed_or_legacy(bytes, CONTRACT_LIMITS, &runmat_meshing_core::NeverCancelled)?;
    let ExactSurfacePassOutcome::Converged { surface } = &pass.outcome else {
        unreachable!("converged pass checked above");
    };
    if &decoded != surface {
        return Err(ExactSurfaceMeshError::new(
            ExactSurfaceMeshErrorKind::InvalidInput,
            "decoded surface differs from the independently validated pass",
        ));
    }
    Ok(decoded)
}

fn decode_with_limits(
    bytes: &[u8],
    topology: &ExactBRepTopology,
    batches: &[ExactFaceMeshBatch],
    options: ExactSurfaceJoinOptions,
    limits: CanonicalLimits,
) -> Result<ExactSurfaceMesh, ExactSurfaceMeshError> {
    let mesh = decode_indexed_or_legacy(bytes, limits, &runmat_meshing_core::NeverCancelled)?;
    validate_exact_surface_mesh(&mesh, topology, batches, options)?;
    Ok(mesh)
}

#[cfg(test)]
pub(crate) fn decode_exact_surface_mesh_with_byte_limit(
    bytes: &[u8],
    topology: &ExactBRepTopology,
    batches: &[ExactFaceMeshBatch],
    options: ExactSurfaceJoinOptions,
    maximum_encoded_bytes: usize,
) -> Result<ExactSurfaceMesh, ExactSurfaceMeshError> {
    decode_with_limits(
        bytes,
        topology,
        batches,
        options,
        CanonicalLimits::content_bounded(maximum_encoded_bytes, 1024 * 1024, 64),
    )
}

/// Publishes a complete converged surface with cooperative prerequisite replay and encoding.
pub fn encode_exact_surface_mesh_from_pass_with_cancellation(
    pass: &ExactSurfacePassResult,
    topology: &ExactBRepTopology,
    curves: &SharedCurveMesh,
    partitions: &[ExactFacePartitionResult],
    options: ExactSurfaceJoinOptions,
    control: &dyn MeshingCancellationSignal,
) -> Result<Vec<u8>, ExactSurfaceMeshError> {
    super::pass_result::validate_with_control(
        pass,
        topology,
        curves,
        partitions,
        options,
        Some(control),
    )?;
    let ExactSurfacePassOutcome::Converged { surface } = &pass.outcome else {
        return Err(ExactSurfaceMeshError::new(
            ExactSurfaceMeshErrorKind::InvalidInput,
            "a curve-restart pass cannot publish a final exact surface",
        ));
    };
    encode_indexed(surface, control)
}

/// Admits a self-contained cold surface against exact topology under execution control.
pub fn decode_published_exact_surface_mesh_with_cancellation(
    bytes: &[u8],
    topology: &ExactBRepTopology,
    options: ExactSurfaceJoinOptions,
    control: &dyn MeshingCancellationSignal,
) -> Result<ExactSurfaceMesh, ExactSurfaceMeshError> {
    let mesh = decode_indexed_or_legacy(bytes, CONTRACT_LIMITS, control)?;
    super::artifact::validate_with_control(&mesh, topology, options, Some(control))?;
    Ok(mesh)
}

// Only the numerical convergence owner exposes this through its opaque accepted
// result; deserializing an ordinary public mesh cannot construct that authority.
pub(in crate::surface_mesh) fn encode_accepted_surface(
    surface: &ExactSurfaceMesh,
    control: &dyn MeshingCancellationSignal,
) -> Result<Vec<u8>, ExactSurfaceMeshError> {
    encode_indexed(surface, control)
}

fn encode_indexed(
    mesh: &ExactSurfaceMesh,
    signal: &dyn MeshingCancellationSignal,
) -> Result<Vec<u8>, ExactSurfaceMeshError> {
    let prepared = SurfaceEncoding::new(mesh, signal)?;
    let profile = prepared.profile(signal)?;
    prepared.encode_profiled(&profile, signal)
}
fn decode_indexed_or_legacy(
    bytes: &[u8],
    limits: CanonicalLimits,
    signal: &dyn MeshingCancellationSignal,
) -> Result<ExactSurfaceMesh, ExactSurfaceMeshError> {
    if bytes.starts_with(indexed::PREFIX) {
        indexed::decode(bytes, limits, signal)
    } else {
        super::codec::decode(CODEC_PREFIX, CONTRACT_DOMAIN, bytes, limits, Some(signal))
    }
}

/// Charges original typed global state before allocation; the supplied scope must
/// outlive the reconstructed surface. Serialized claims grant no admission.
pub fn decode_published_exact_surface_mesh_admitted(
    bytes: &[u8],
    topology: &ExactBRepTopology,
    options: ExactSurfaceJoinOptions,
    signal: &dyn MeshingCancellationSignal,
    allocations: &dyn runmat_geometry_core::GeometryEvaluationControl,
) -> Result<ExactSurfaceMesh, ExactSurfaceMeshError> {
    let mesh = if bytes.starts_with(indexed::PREFIX) {
        indexed::decode_admitted(bytes, CONTRACT_LIMITS, signal, Some(allocations))?
    } else {
        let codec = runmat_meshing_core::MeshingCanonicalCodecControl::new(signal);
        let requested=runmat_canonical_codec::canonical_chunked_payload_materialization_resident_bytes_estimate([bytes],&codec).map_err(|error|ExactSurfaceMeshError::new(ExactSurfaceMeshErrorKind::InvalidEncoding,error.to_string()))?;
        allocations
            .consume_allocation_bytes(requested)
            .map_err(|error| {
                ExactSurfaceMeshError::new(
                    ExactSurfaceMeshErrorKind::ResourceLimit,
                    error.to_string(),
                )
            })?;
        decode_indexed_or_legacy(bytes, CONTRACT_LIMITS, signal)?
    };
    super::artifact::validate_admitted(&mesh, topology, options, Some(signal), Some(allocations))?;
    Ok(mesh)
}
