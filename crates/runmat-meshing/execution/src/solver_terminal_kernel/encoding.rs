//! Terminal codec work is admitted independently of its original typed input.
//! Hash already-produced canonical bytes rather than serializing them twice.
use super::*;
use crate::{codec_control::CodecControl, MeshingGeometryEvaluationControl};
use runmat_geometry_core::{GeometryAllocationReservation, GeometryEvaluationControl};

pub(super) fn encode<T: CanonicalMeshingContract>(
    value: &T,
    stage: MeshingStageKind,
    control: &MeshingGeometryEvaluationControl<'_>,
    validation_scratch: u64,
    produce: impl FnOnce(
        &CodecControl<'_>,
    ) -> Result<Vec<u8>, runmat_meshing_core::MeshingContractError>,
) -> Result<Vec<u8>, Box<MeshingFailure>> {
    let codec = CodecControl::new(control);
    let map_error = |error: runmat_meshing_core::MeshingContractError| match codec.take_failure() {
        Some(error) => crate::budget::failure(
            stage,
            crate::geometry_failure::category(error.kind),
            &error.reason,
            None,
        ),
        None => terminal_failure(stage, &error.to_string()),
    };
    let profile = runmat_canonical_codec::profile_canonical_encoding_with_control(
        value,
        T::LIMITS.into(),
        &codec,
    )
    .map_err(|error| {
        map_error(runmat_meshing_core::MeshingContractError::invalid(
            error.field,
            error.reason,
        ))
    })?;
    let mut storage = GeometryAllocationReservation::new(
        control,
        validation_scratch.max(profile.resident_workspace_bytes_estimate),
    )
    .map_err(|error| {
        crate::budget::failure(
            stage,
            crate::geometry_failure::category(error.kind),
            &error.reason,
            None,
        )
    })?;
    let bytes = produce(&codec).map_err(map_error)?;
    if bytes.len() as u64 > profile.encoded_bytes_upper_bound
        || bytes.capacity() as u64 > storage.bytes()
    {
        return Err(terminal_failure(
            stage,
            "terminal canonical output exceeds its admitted profile",
        ));
    }
    storage.retain_bytes(bytes.capacity() as u64);
    Ok(bytes)
}

pub(super) fn digest(
    bytes: &[u8],
    stage: MeshingStageKind,
    control: &MeshingGeometryEvaluationControl<'_>,
) -> Result<StableDigest, Box<MeshingFailure>> {
    runmat_execution::Digest::sha256_parts_with_checkpoint([bytes], || control.checkpoint())
        .map(|digest| StableDigest::from_bytes(*digest.bytes()))
        .map_err(|error| {
            crate::budget::failure(
                stage,
                crate::geometry_failure::category(error.kind),
                &error.reason,
                None,
            )
        })
}

/// Full original projection validation and canonical hashing use their actual
/// shared profile, separately from the retained typed projection.
pub(super) fn projection_work<'a>(
    projection: &SolverMeshProjection,
    stage: MeshingStageKind,
    control: &'a MeshingGeometryEvaluationControl<'_>,
    sealing: bool,
) -> Result<GeometryAllocationReservation<'a>, Box<MeshingFailure>> {
    let codec = CodecControl::new(control);
    let bytes = if sealing {
        projection.artifact_sealing_workspace_bytes(&codec)
    } else {
        runmat_canonical_codec::profile_canonical_encoding_with_control(
            projection,
            <SolverMeshProjection as CanonicalMeshingContract>::LIMITS.into(),
            &codec,
        )
        .map_err(|error| {
            runmat_meshing_core::MeshingContractError::invalid(error.field, error.reason)
        })
        .and_then(|profile| {
            Ok(profile.resident_workspace_bytes_estimate.max(
                projection
                    .topology
                    .validation_scratch_bytes(projection.resolved_request.element_order)?,
            ))
        })
    }
    .map_err(|error| source::map_error(stage, &codec, error))?;
    GeometryAllocationReservation::new(control, bytes).map_err(|error| {
        crate::budget::failure(
            stage,
            crate::geometry_failure::category(error.kind),
            &error.reason,
            None,
        )
    })
}
