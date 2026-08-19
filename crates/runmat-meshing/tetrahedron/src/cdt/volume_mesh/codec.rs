use runmat_canonical_codec::{CanonicalCodecError, CanonicalLimits};
use runmat_geometry_core::ExactBRepTopology;
use runmat_meshing_core::MeshingCancellationSignal;
use runmat_meshing_size::metric::MetricFieldRequest;
use runmat_meshing_surface::ExactSurfaceMesh;
use serde::{Deserialize, Serialize};

use super::{validate_delaunay_volume_mesh, DelaunayVolumeMesh, DelaunayVolumeMeshOptions};

pub const DELAUNAY_VOLUME_MESH_SCHEMA_VERSION: u16 = 1;

const CODEC_PREFIX: &[u8] = b"runmat-meshing-delaunay-volume-canonical-cbor/v1\0";
const CONTRACT_DOMAIN: &str = "delaunay-volume-mesh/v1";
const CONTRACT_LIMITS: CanonicalLimits =
    CanonicalLimits::new(1024 * 1024 * 1024, 50_000_000, 1024 * 1024, 64);

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum DelaunayVolumeMeshCodecErrorKind {
    InvalidEncoding,
    InvalidMesh,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct DelaunayVolumeMeshCodecError {
    pub kind: DelaunayVolumeMeshCodecErrorKind,
    pub reason: String,
}

impl std::fmt::Display for DelaunayVolumeMeshCodecError {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(
            formatter,
            "Delaunay volume artifact {:?}: {}",
            self.kind, self.reason
        )
    }
}

impl std::error::Error for DelaunayVolumeMeshCodecError {}

#[derive(Serialize)]
#[serde(deny_unknown_fields)]
struct DelaunayVolumeMeshEnvelope<'a> {
    schema_version: u16,
    mesh: &'a DelaunayVolumeMesh,
}

#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct DecodedDelaunayVolumeMeshEnvelope {
    schema_version: u16,
    mesh: DelaunayVolumeMesh,
}

pub fn encode_delaunay_volume_mesh(
    mesh: &DelaunayVolumeMesh,
    topology: &ExactBRepTopology,
    surface: &ExactSurfaceMesh,
    metric_request: &MetricFieldRequest,
    options: DelaunayVolumeMeshOptions,
    cancellation: &dyn MeshingCancellationSignal,
) -> Result<Vec<u8>, DelaunayVolumeMeshCodecError> {
    validate_delaunay_volume_mesh(
        topology,
        surface,
        metric_request,
        mesh,
        options,
        cancellation,
    )
    .map_err(invalid_mesh)?;
    runmat_canonical_codec::encode_contract(
        CODEC_PREFIX,
        CONTRACT_DOMAIN,
        &DelaunayVolumeMeshEnvelope {
            schema_version: DELAUNAY_VOLUME_MESH_SCHEMA_VERSION,
            mesh,
        },
        CONTRACT_LIMITS,
    )
    .map_err(invalid_encoding)
}

pub fn decode_delaunay_volume_mesh(
    bytes: &[u8],
    topology: &ExactBRepTopology,
    surface: &ExactSurfaceMesh,
    metric_request: &MetricFieldRequest,
    options: DelaunayVolumeMeshOptions,
    cancellation: &dyn MeshingCancellationSignal,
) -> Result<DelaunayVolumeMesh, DelaunayVolumeMeshCodecError> {
    decode_with_limits(
        bytes,
        topology,
        surface,
        metric_request,
        options,
        cancellation,
        CONTRACT_LIMITS,
    )
}

fn decode_with_limits(
    bytes: &[u8],
    topology: &ExactBRepTopology,
    surface: &ExactSurfaceMesh,
    metric_request: &MetricFieldRequest,
    options: DelaunayVolumeMeshOptions,
    cancellation: &dyn MeshingCancellationSignal,
    limits: CanonicalLimits,
) -> Result<DelaunayVolumeMesh, DelaunayVolumeMeshCodecError> {
    let artifact: DecodedDelaunayVolumeMeshEnvelope =
        runmat_canonical_codec::decode_contract(CODEC_PREFIX, CONTRACT_DOMAIN, bytes, limits)
            .map_err(invalid_encoding)?;
    if artifact.schema_version != DELAUNAY_VOLUME_MESH_SCHEMA_VERSION {
        return Err(DelaunayVolumeMeshCodecError {
            kind: DelaunayVolumeMeshCodecErrorKind::InvalidEncoding,
            reason: "unsupported Delaunay volume artifact schema".to_owned(),
        });
    }
    validate_delaunay_volume_mesh(
        topology,
        surface,
        metric_request,
        &artifact.mesh,
        options,
        cancellation,
    )
    .map_err(invalid_mesh)?;
    Ok(artifact.mesh)
}

#[cfg(test)]
pub(super) fn decode_delaunay_volume_mesh_with_byte_limit(
    bytes: &[u8],
    topology: &ExactBRepTopology,
    surface: &ExactSurfaceMesh,
    metric_request: &MetricFieldRequest,
    options: DelaunayVolumeMeshOptions,
    cancellation: &dyn MeshingCancellationSignal,
    maximum_encoded_bytes: usize,
) -> Result<DelaunayVolumeMesh, DelaunayVolumeMeshCodecError> {
    decode_with_limits(
        bytes,
        topology,
        surface,
        metric_request,
        options,
        cancellation,
        CanonicalLimits::new(maximum_encoded_bytes, 50_000_000, 1024 * 1024, 64),
    )
}

#[cfg(test)]
pub(super) fn encode_delaunay_volume_mesh_with_schema_version(
    mesh: &DelaunayVolumeMesh,
    schema_version: u16,
) -> Result<Vec<u8>, DelaunayVolumeMeshCodecError> {
    runmat_canonical_codec::encode_contract(
        CODEC_PREFIX,
        CONTRACT_DOMAIN,
        &DelaunayVolumeMeshEnvelope {
            schema_version,
            mesh,
        },
        CONTRACT_LIMITS,
    )
    .map_err(invalid_encoding)
}

fn invalid_encoding(error: CanonicalCodecError) -> DelaunayVolumeMeshCodecError {
    DelaunayVolumeMeshCodecError {
        kind: DelaunayVolumeMeshCodecErrorKind::InvalidEncoding,
        reason: format!("{}: {}", error.field, error.reason),
    }
}

fn invalid_mesh(error: super::DelaunayVolumeMeshError) -> DelaunayVolumeMeshCodecError {
    DelaunayVolumeMeshCodecError {
        kind: DelaunayVolumeMeshCodecErrorKind::InvalidMesh,
        reason: error.to_string(),
    }
}
