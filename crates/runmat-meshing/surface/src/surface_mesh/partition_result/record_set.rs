//! An explicit stream header plus individually encoded original faces. The
//! consumer validates all faces and complete current partition coverage; a header
//! is transport metadata and cannot grant numerical acceptance.
use super::*;
use crate::{ExactFaceMesh, ExactFacePartitionCompletion};
use runmat_canonical_codec::{CanonicalEncodingProfile, CanonicalLimits};
use runmat_geometry_core::{GeometryAllocationScope, GeometryEvaluationControl};
use serde::{Deserialize, Serialize};
const PREFIX: &[u8] = b"runmat-meshing-face-record-set/v2\0";
const DOMAIN: &str = "exact-face-record-set/v2";
const LIMITS: CanonicalLimits = CanonicalLimits::platform_content(1024 * 1024, 64);
pub const EXACT_FACE_PARTITION_RECORD_SET_SCHEMA_VERSION: u16 = 2;
#[derive(Serialize)]
struct Header<'a> {
    format_version: u16,
    partition: &'a MeshingPartitionDescriptor,
    face_count: usize,
    splits: &'a [runmat_meshing_curve::SharedCurveSegmentSplit],
}
impl<'a> Header<'a> {
    fn new(completion: &'a ExactFacePartitionCompletion) -> Self {
        Self {
            format_version: 2,
            partition: &completion.partition,
            face_count: if completion.curve_splits.is_empty() {
                completion.accepted_face_count
            } else {
                0
            },
            splits: &completion.curve_splits,
        }
    }
}
#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct OwnedHeader {
    format_version: u16,
    partition: MeshingPartitionDescriptor,
    face_count: usize,
    #[serde(deserialize_with = "runmat_canonical_codec::deserialize_exact_vec")]
    splits: Vec<runmat_meshing_curve::SharedCurveSegmentSplit>,
}
pub fn profile_exact_face_partition_completion(
    completion: &ExactFacePartitionCompletion,
    signal: &dyn MeshingCancellationSignal,
) -> Result<CanonicalEncodingProfile, ExactSurfaceMeshError> {
    let control = runmat_meshing_core::MeshingCanonicalCodecControl::new(signal);
    runmat_canonical_codec::profile_canonical_encoding_with_control(
        &Header::new(completion),
        LIMITS,
        &control,
    )
    .map_err(|error| codec_error(error, &control))
}
pub fn encode_exact_face_partition_completion(
    completion: &ExactFacePartitionCompletion,
    signal: &dyn MeshingCancellationSignal,
) -> Result<Vec<u8>, ExactSurfaceMeshError> {
    let control = runmat_meshing_core::MeshingCanonicalCodecControl::new(signal);
    runmat_canonical_codec::encode_contract_with_control(
        PREFIX,
        DOMAIN,
        &Header::new(completion),
        LIMITS,
        &control,
    )
    .map_err(|error| codec_error(error, &control))
}
pub fn decode_exact_face_partition_records(
    records: &[Vec<u8>],
    topology: &ExactBRepTopology,
    curves: &SharedCurveMesh,
) -> Result<ExactFacePartitionResult, ExactSurfaceMeshError> {
    decode(
        records,
        topology,
        curves,
        &runmat_meshing_core::NeverCancelled,
        None,
    )
}
pub fn decode_exact_face_partition_records_with_cancellation(
    records: &[Vec<u8>],
    topology: &ExactBRepTopology,
    curves: &SharedCurveMesh,
    signal: &dyn MeshingCancellationSignal,
) -> Result<ExactFacePartitionResult, ExactSurfaceMeshError> {
    decode(records, topology, curves, signal, None)
}
pub fn decode_exact_face_partition_records_admitted(
    records: &[Vec<u8>],
    topology: &ExactBRepTopology,
    curves: &SharedCurveMesh,
    signal: &dyn MeshingCancellationSignal,
    allocations: &dyn GeometryEvaluationControl,
) -> Result<ExactFacePartitionResult, ExactSurfaceMeshError> {
    decode(records, topology, curves, signal, Some(allocations))
}
fn allocation_error(error: runmat_geometry_core::GeometryEvaluationError) -> ExactSurfaceMeshError {
    ExactSurfaceMeshError::new(
        if error.kind == runmat_geometry_core::GeometryEvaluationErrorKind::Cancelled {
            ExactSurfaceMeshErrorKind::Cancelled
        } else {
            ExactSurfaceMeshErrorKind::ResourceLimit
        },
        error.to_string(),
    )
}
fn decode(
    records: &[Vec<u8>],
    topology: &ExactBRepTopology,
    curves: &SharedCurveMesh,
    signal: &dyn MeshingCancellationSignal,
    allocations: Option<&dyn GeometryEvaluationControl>,
) -> Result<ExactFacePartitionResult, ExactSurfaceMeshError> {
    let Some(first) = records.first() else {
        return Err(invalid("surface partition records are empty"));
    };
    if !first.starts_with(PREFIX) {
        if records.len() != 1 {
            return Err(invalid("legacy partition must contain one record"));
        }
        return match allocations {
            Some(allocations) => super::codec::decode_exact_face_partition_result_admitted(
                first,
                topology,
                curves,
                signal,
                allocations,
            ),
            None => super::codec::decode_exact_face_partition_result_with_cancellation(
                first, topology, curves, signal,
            ),
        };
    }
    let control = runmat_meshing_core::MeshingCanonicalCodecControl::new(signal);
    let metadata = allocations.map(GeometryAllocationScope::new);
    let profile =
        runmat_canonical_codec::canonical_chunked_payload_materialization_resident_bytes_estimate(
            [first.as_slice()],
            &control,
        )
        .map_err(|error| codec_error(error, &control))?;
    if let Some(scope) = &metadata {
        scope
            .consume_allocation_bytes(profile)
            .map_err(allocation_error)?;
    }
    let header: OwnedHeader = runmat_canonical_codec::decode_contract_with_control(
        PREFIX, DOMAIN, first, LIMITS, &control,
    )
    .map_err(|error| codec_error(error, &control))?;
    if header.format_version != 2
        || header.face_count > topology.faces.len()
        || header.face_count != records.len() - 1
    {
        return Err(invalid(
            "surface partition record inventory differs from its header",
        ));
    }
    validate_face_partition_descriptor(&header.partition)?;
    let outcome = if !header.splits.is_empty() {
        if records.len() != 1 {
            return Err(invalid(
                "curve restart must not publish partial face records",
            ));
        }
        ExactFacePartitionOutcome::RequiresCurveSplits {
            splits: header.splits,
        }
    } else {
        if header.face_count == 0 {
            return Err(invalid("converged partition omits its face records"));
        }
        if let Some(allocations) = allocations {
            allocations
                .consume_allocation_bytes(
                    (header.face_count as u64)
                        .checked_mul(std::mem::size_of::<ExactFaceMesh>() as u64)
                        .ok_or_else(|| invalid("surface record owner overflow"))?,
                )
                .map_err(allocation_error)?;
        }
        let mut faces = Vec::with_capacity(header.face_count);
        for bytes in &records[1..] {
            checkpoint(Some(signal))?;
            let scope = allocations.map(GeometryAllocationScope::new);
            let decoded = match &scope {
                Some(scope) => super::codec::decode_exact_face_partition_result_admitted(
                    bytes, topology, curves, signal, scope,
                ),
                None => super::codec::decode_exact_face_partition_result_with_cancellation(
                    bytes, topology, curves, signal,
                ),
            }?;
            let ExactFacePartitionOutcome::Converged {
                faces: mut original,
            } = decoded.outcome
            else {
                return Err(invalid("face stream contains a nested curve restart"));
            };
            if original.len() != 1 {
                return Err(invalid("face stream record must contain one original face"));
            }
            let face = original.pop().expect("one checked original face");
            if decoded.partition
                != crate::surface_mesh::indexed_face::partition::single_partition(&face)
            {
                return Err(invalid("face stream record has a foreign local descriptor"));
            }
            if let Some(scope) = &scope {
                let retained = face
                    .retained_storage_bytes(scope)
                    .map_err(allocation_error)?;
                scope.retain_bytes(retained - std::mem::size_of::<ExactFaceMesh>() as u64);
            }
            faces.push(face);
        }
        ExactFacePartitionOutcome::Converged { faces }
    };
    let boundary = crate::build_exact_surface_boundary_with_cancellation(topology, curves, signal)
        .map_err(|error| {
            ExactSurfaceMeshError::new(
                match error.kind {
                    crate::ExactSurfaceBoundaryErrorKind::Cancelled => {
                        ExactSurfaceMeshErrorKind::Cancelled
                    }
                    crate::ExactSurfaceBoundaryErrorKind::ResourceLimit => {
                        ExactSurfaceMeshErrorKind::ResourceLimit
                    }
                    _ => ExactSurfaceMeshErrorKind::InvalidInput,
                },
                error.to_string(),
            )
        })?;
    let result = super::build_exact_face_partition_result_with_boundary(
        topology,
        curves,
        &boundary,
        header.partition,
        outcome,
        Some(signal),
    )?;
    if let Some(scope) = &metadata {
        scope.retain_bytes(profile);
    }
    Ok(result)
}
fn codec_error(
    error: runmat_canonical_codec::CanonicalCodecError,
    control: &runmat_meshing_core::MeshingCanonicalCodecControl<'_>,
) -> ExactSurfaceMeshError {
    ExactSurfaceMeshError::new(
        if control.observed_cancellation() {
            ExactSurfaceMeshErrorKind::Cancelled
        } else {
            ExactSurfaceMeshErrorKind::InvalidEncoding
        },
        error.to_string(),
    )
}
