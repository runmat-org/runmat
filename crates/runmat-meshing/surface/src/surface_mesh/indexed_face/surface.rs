//! Global surface transport shares the original identity and quality columns.
//! Consumers reconstruct original values and independently validate topology.
use super::{decoding::DecodedFace, encoding::FaceEncoding, rows::MeshRows, *};
use crate::{ExactSurfaceMesh, ExactSurfaceShellEvidence};
use runmat_canonical_codec::{CanonicalEncodingProfile, CanonicalLimits, ExactVec};
use runmat_meshing_core::MeshingCanonicalCodecControl;
use serde::{Deserialize, Serialize};
pub(crate) const PREFIX: &[u8] = b"runmat-meshing-indexed-surface/v2\0";
const DOMAIN: &str = "exact-surface-mesh/indexed-v2";
pub(crate) const LIMITS: CanonicalLimits =
    CanonicalLimits::platform_content(1024 * 1024, 64);
pub(crate) struct SurfaceEncoding<'a> {
    mesh: &'a ExactSurfaceMesh,
    rows: FaceEncoding<'a>,
}
impl<'a> SurfaceEncoding<'a> {
    pub fn preparation_storage_bytes(
        mesh: &ExactSurfaceMesh,
        control: &dyn MeshingCancellationSignal,
    ) -> Result<u64, ExactSurfaceMeshError> {
        FaceEncoding::preparation_for_rows(MeshRows::surface(mesh)?, control)
    }
    pub fn new(
        mesh: &'a ExactSurfaceMesh,
        control: &dyn MeshingCancellationSignal,
    ) -> Result<Self, ExactSurfaceMeshError> {
        Ok(Self {
            mesh,
            rows: FaceEncoding::prepare_rows(MeshRows::surface(mesh)?, control)?,
        })
    }
    pub fn profile(
        &self,
        signal: &dyn MeshingCancellationSignal,
    ) -> Result<CanonicalEncodingProfile, ExactSurfaceMeshError> {
        let control = MeshingCanonicalCodecControl::new(signal);
        runmat_canonical_codec::profile_canonical_encoding_with_control(self, LIMITS, &control)
            .map_err(|error| codec_error(error, &control))
    }
    pub fn encode_profiled(
        &self,
        profile: &CanonicalEncodingProfile,
        signal: &dyn MeshingCancellationSignal,
    ) -> Result<Vec<u8>, ExactSurfaceMeshError> {
        let control = MeshingCanonicalCodecControl::new(signal);
        let capacity = usize::try_from(profile.encoded_bytes_upper_bound)
            .map_err(|_| invalid("global surface profile overflow"))?;
        let bytes = runmat_canonical_codec::encode_contract_with_capacity(
            PREFIX, DOMAIN, self, LIMITS, &control, capacity,
        )
        .map_err(|error| codec_error(error, &control))?;
        if bytes.len() > capacity || bytes.capacity() > capacity {
            return Err(invalid("global surface encoder exceeds its profile"));
        }
        Ok(bytes)
    }
}
impl Serialize for SurfaceEncoding<'_> {
    fn serialize<S: serde::Serializer>(&self, serializer: S) -> Result<S::Ok, S::Error> {
        #[derive(Serialize)]
        struct Surface<'a, 'b> {
            format_version: u16,
            schema_version: u16,
            face_ids: &'a [PersistentEntityId],
            shells: &'a [ExactSurfaceShellEvidence],
            rows: &'a FaceEncoding<'b>,
        }
        Surface {
            format_version: 2,
            schema_version: self.mesh.schema_version,
            face_ids: &self.mesh.face_ids,
            shells: &self.mesh.shells,
            rows: &self.rows,
        }
        .serialize(serializer)
    }
}
#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct DecodedSurface {
    format_version: u16,
    schema_version: u16,
    face_ids: ExactVec<PersistentEntityId>,
    shells: ExactVec<ExactSurfaceShellEvidence>,
    rows: DecodedFace,
}

pub(crate) fn decode(
    bytes: &[u8],
    limits: CanonicalLimits,
    signal: &dyn MeshingCancellationSignal,
) -> Result<ExactSurfaceMesh, ExactSurfaceMeshError> {
    decode_admitted(bytes, limits, signal, None)
}
pub(crate) fn decode_admitted(
    bytes: &[u8],
    limits: CanonicalLimits,
    signal: &dyn MeshingCancellationSignal,
    allocations: Option<&dyn runmat_geometry_core::GeometryEvaluationControl>,
) -> Result<ExactSurfaceMesh, ExactSurfaceMeshError> {
    let control = MeshingCanonicalCodecControl::new(signal);
    let decoded: DecodedSurface = runmat_canonical_codec::decode_contract_with_admission(
        PREFIX,
        DOMAIN,
        bytes,
        limits,
        &control,
        |value| {
            if let Some(allocations) = allocations {
                let requested = super::inventory::requested_surface_bytes(value, &control)?;
                allocations
                    .consume_allocation_bytes(requested)
                    .map_err(|error| runmat_canonical_codec::CanonicalCodecError {
                        kind: runmat_canonical_codec::CanonicalCodecErrorKind::ResourceLimit,
                        field: "global surface typed admission",
                        reason: error.to_string(),
                    })?;
            }
            Ok(())
        },
    )
    .map_err(|error| codec_error(error, &control))?;
    if decoded.format_version != 2
        || decoded.schema_version != crate::EXACT_SURFACE_MESH_SCHEMA_VERSION
    {
        return Err(invalid("unsupported indexed global surface schema"));
    }
    decoded.rows.validate_global_frame(&decoded.face_ids)?;
    let rows = decoded.rows.into_mesh(signal, |_| Ok(()))?;
    Ok(ExactSurfaceMesh {
        schema_version: decoded.schema_version,
        face_ids: decoded.face_ids.into_vec(),
        nodes: rows.nodes,
        triangles: rows.triangles,
        boundary_segments: rows.boundary_segments,
        shells: decoded.shells.into_vec(),
        maximum_chordal_deviation_m: runmat_types::physical::Length::new(
            rows.maximum_chordal_deviation_m.canonical_value(),
        ),
        maximum_normal_deviation_rad: rows.maximum_normal_deviation_rad,
    })
}
fn codec_error(
    error: runmat_canonical_codec::CanonicalCodecError,
    control: &MeshingCanonicalCodecControl<'_>,
) -> ExactSurfaceMeshError {
    ExactSurfaceMeshError::new(
        if control.observed_cancellation() {
            ExactSurfaceMeshErrorKind::Cancelled
        } else if error.kind == runmat_canonical_codec::CanonicalCodecErrorKind::ResourceLimit {
            ExactSurfaceMeshErrorKind::ResourceLimit
        } else {
            ExactSurfaceMeshErrorKind::InvalidEncoding
        },
        error.to_string(),
    )
}
