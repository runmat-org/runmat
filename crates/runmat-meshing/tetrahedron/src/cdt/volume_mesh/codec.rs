use runmat_canonical_codec::{CanonicalCodecError, CanonicalLimits};
use runmat_geometry_core::ExactBRepTopology;
use runmat_meshing_core::MeshingCancellationSignal;
use runmat_meshing_size::metric::MetricFieldRequest;
use runmat_meshing_surface::ExactSurfaceMesh;
use serde::{Deserialize, Serialize};

use super::{validate_delaunay_volume_mesh, DelaunayVolumeMesh, DelaunayVolumeMeshOptions};

pub(in crate::cdt) mod admission;
pub(super) mod control;
mod indexed;
pub use indexed::VolumeMeshEncoding;
pub const DELAUNAY_VOLUME_MESH_SCHEMA_VERSION: u16 = 5;

const CODEC_PREFIX: &[u8] = b"runmat-meshing-delaunay-volume-canonical-cbor/v4\0";
const CONTRACT_DOMAIN: &str = "delaunay-volume-mesh/v4";
const LEGACY_PREFIX: &[u8] = b"runmat-meshing-delaunay-volume-canonical-cbor/v3\0";
const LEGACY_DOMAIN: &str = "delaunay-volume-mesh/v3";
const CONTRACT_LIMITS: CanonicalLimits = CanonicalLimits::platform_content(1024 * 1024, 64);

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum DelaunayVolumeMeshCodecErrorKind {
    InvalidEncoding,
    InvalidMesh,
    Cancelled,
    ResourceLimit,
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

#[cfg(test)]
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

/// Construction-only authority. Immutable access preserves the complete final
/// CAD/surface/quality admission through publication; deserialization cannot
/// manufacture this token and callers cannot mutate its mesh.
pub struct ValidatedDelaunayVolumeMesh(DelaunayVolumeMesh);
impl std::ops::Deref for ValidatedDelaunayVolumeMesh {
    type Target = DelaunayVolumeMesh;
    fn deref(&self) -> &Self::Target {
        &self.0
    }
}
impl ValidatedDelaunayVolumeMesh {
    pub(super) fn new(mesh: DelaunayVolumeMesh) -> Self {
        Self(mesh)
    }
    pub fn into_mesh(self) -> DelaunayVolumeMesh {
        self.0
    }
    /// Requested flat dictionary and cooperative-sort workspace. Admit before preparation.
    pub fn encoding_preparation_storage_bytes(&self) -> Result<u64, DelaunayVolumeMeshCodecError> {
        VolumeMeshEncoding::preparation_bytes(&self.0)
    }
    pub fn prepare_encoding(
        &self,
        control: &dyn runmat_canonical_codec::CanonicalCodecControl,
    ) -> Result<VolumeMeshEncoding<'_>, DelaunayVolumeMeshCodecError> {
        VolumeMeshEncoding::new(&self.0, control)
    }
    pub fn encoding_profile(
        &self,
        control: &dyn runmat_canonical_codec::CanonicalCodecControl,
    ) -> Result<runmat_canonical_codec::CanonicalEncodingProfile, DelaunayVolumeMeshCodecError>
    {
        self.prepare_encoding(control)?.encoding_profile(control)
    }
    pub fn encode(
        &self,
        control: &dyn runmat_canonical_codec::CanonicalCodecControl,
    ) -> Result<Vec<u8>, DelaunayVolumeMeshCodecError> {
        self.prepare_encoding(control)?.encode(control)
    }
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
    let control = control::Control::new(cancellation);
    VolumeMeshEncoding::new(mesh, &control)?
        .encode(&control)
        .map_err(|error| control.classify(error))
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
        CONTRACT_LIMITS.into(),
    )
}

fn decode_with_limits(
    bytes: &[u8],
    topology: &ExactBRepTopology,
    surface: &ExactSurfaceMesh,
    metric_request: &MetricFieldRequest,
    options: DelaunayVolumeMeshOptions,
    cancellation: &dyn MeshingCancellationSignal,
    policy: admission::DecodePolicy<'_>,
) -> Result<DelaunayVolumeMesh, DelaunayVolumeMeshCodecError> {
    let limits = policy.limits;
    if bytes.len() > limits.maximum_encoded_bytes {
        return Err(DelaunayVolumeMeshCodecError {
            kind: DelaunayVolumeMeshCodecErrorKind::InvalidEncoding,
            reason: "volume artifact exceeds its encoded byte ceiling".into(),
        });
    }
    super::validate_option_consistency(options).map_err(invalid_mesh)?;
    let control = control::Control::new(cancellation);
    let mesh = if bytes.starts_with(LEGACY_PREFIX) {
        let artifact: DecodedDelaunayVolumeMeshEnvelope =
            runmat_canonical_codec::decode_contract_with_admission(
                LEGACY_PREFIX,
                LEGACY_DOMAIN,
                bytes,
                limits,
                &control,
                |_| policy.admit_legacy(bytes, &control),
            )
            .map_err(|error| control.classify(invalid_encoding(error)))?;
        if artifact.schema_version != 3 {
            return Err(unsupported_schema());
        }
        artifact.mesh
    } else {
        let expansion = match policy.allocations {
            // The governed route charges each actual expanded inventory to the
            // caller's original resource ceiling instead of a second loose bound.
            Some(_) => u64::MAX,
            None => runmat_canonical_codec::canonical_chunked_payload_materialization_resident_bytes_estimate([bytes], &control)
                .map_err(|error| control.classify(invalid_encoding(error)))?,
        };
        let artifact: indexed::DecodedEnvelope =
            runmat_canonical_codec::decode_contract_with_admission(
                CODEC_PREFIX,
                CONTRACT_DOMAIN,
                bytes,
                limits,
                &control,
                |value| policy.admit_indexed(value, &control),
            )
            .map_err(|error| control.classify(invalid_encoding(error)))?;
        if artifact.schema_version != DELAUNAY_VOLUME_MESH_SCHEMA_VERSION {
            return Err(unsupported_schema());
        }
        artifact
            .mesh
            .into_mesh(options, cancellation, expansion, policy.allocations)
            .map_err(invalid_mesh)?
    };
    validate_delaunay_volume_mesh(
        topology,
        surface,
        metric_request,
        &mesh,
        options,
        cancellation,
    )
    .map_err(invalid_mesh)?;
    Ok(mesh)
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
        CanonicalLimits::content_bounded(maximum_encoded_bytes, 1024 * 1024, 64).into(),
    )
}

#[cfg(test)]
pub(super) fn encode_delaunay_volume_mesh_with_schema_version(
    mesh: &DelaunayVolumeMesh,
    schema_version: u16,
) -> Result<Vec<u8>, DelaunayVolumeMeshCodecError> {
    runmat_canonical_codec::encode_contract(
        LEGACY_PREFIX,
        LEGACY_DOMAIN,
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
        kind: if error.kind == runmat_canonical_codec::CanonicalCodecErrorKind::ResourceLimit {
            DelaunayVolumeMeshCodecErrorKind::ResourceLimit
        } else {
            DelaunayVolumeMeshCodecErrorKind::InvalidEncoding
        },
        reason: format!("{}: {}", error.field, error.reason),
    }
}

fn invalid_mesh(error: super::DelaunayVolumeMeshError) -> DelaunayVolumeMeshCodecError {
    DelaunayVolumeMeshCodecError {
        kind: match error.kind {
            super::DelaunayVolumeMeshErrorKind::Cancelled => {
                DelaunayVolumeMeshCodecErrorKind::Cancelled
            }
            super::DelaunayVolumeMeshErrorKind::ResourceLimit => {
                DelaunayVolumeMeshCodecErrorKind::ResourceLimit
            }
            _ => DelaunayVolumeMeshCodecErrorKind::InvalidMesh,
        },
        reason: error.to_string(),
    }
}

fn unsupported_schema() -> DelaunayVolumeMeshCodecError {
    DelaunayVolumeMeshCodecError {
        kind: DelaunayVolumeMeshCodecErrorKind::InvalidEncoding,
        reason: "unsupported Delaunay volume artifact schema".into(),
    }
}
