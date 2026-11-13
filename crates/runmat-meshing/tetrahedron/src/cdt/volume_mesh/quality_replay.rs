//! Diagnostic input at the quality boundary, never an accepted mesh artifact.
//! Replaying always rechecks topology and recomputes quality before execution.
use super::*;
use runmat_canonical_codec::{CanonicalCodecControl, CanonicalLimits};
use runmat_meshing_core::StableDigest;

const PREFIX: &[u8] = b"runmat-volume-quality-diagnostic/v1\0";
const DOMAIN: &str = "volume-quality-diagnostic/v1";

#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct VolumeQualityBinding {
    pub request: StableDigest,
    pub geometry: StableDigest,
    /// Implementation that produced this diagnostic input, not a proof token.
    pub implementation: StableDigest,
}

#[derive(Serialize)]
pub struct VolumeQualitySnapshot<'a> {
    pub topology: &'a DelaunayVolumeTopology,
    pub provenance: &'a DelaunayVolumeProvenance,
    pub metric: &'a MetricFieldRequest,
    pub fixed_nodes: &'a [StableDigest],
    pub options: DelaunayVolumeMeshOptions,
}

/// Host-owned bounded diagnostic persistence. Numeric construction does no I/O.
pub trait VolumeQualityObserver: Send + Sync {
    fn observe(&self, snapshot: VolumeQualitySnapshot<'_>) -> Result<(), DelaunayVolumeMeshError>;
    /// Optional terminal diagnostic, never an accepted or resumable mesh token.
    fn observe_stalled(
        &self,
        _snapshot: VolumeQualitySnapshot<'_>,
    ) -> Result<(), DelaunayVolumeMeshError> {
        Ok(())
    }
}
pub(super) struct NoopQualityObserver;
impl VolumeQualityObserver for NoopQualityObserver {
    fn observe(&self, _: VolumeQualitySnapshot<'_>) -> Result<(), DelaunayVolumeMeshError> {
        Ok(())
    }
}

mod encoding;
use super::representation::DecodedVolumeRepresentation;
pub use encoding::{
    prepare_volume_quality_snapshot, volume_quality_snapshot_preparation_storage_bytes,
    PreparedVolumeQualitySnapshot,
};
#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct DecodedEnvelope {
    binding: VolumeQualityBinding,
    input: Input,
}
#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct Input {
    volume: DecodedVolumeRepresentation,
    metric: MetricFieldRequest,
    fixed_nodes: Vec<StableDigest>,
    options: DelaunayVolumeMeshOptions,
}

/// Deliberately distinct from the complete, surface-admitted mesh artifact.
pub struct VolumeQualityReplayResult {
    pub topology: DelaunayVolumeTopology,
    pub provenance: DelaunayVolumeProvenance,
    pub quality: DelaunayVolumeQuality,
    /// Applied refinement and optimization steps in their original order.
    pub quality_mutations: Vec<super::DelaunayVolumeQualityMutation>,
}

/// Profiles borrowed inputs before any diagnostic wire allocation. The caller
/// separately owns the original topology and charges encoding/decoding lifetimes.
pub fn profile_volume_quality_snapshot(
    snapshot: &VolumeQualitySnapshot<'_>,
    binding: VolumeQualityBinding,
    limits: CanonicalLimits,
    control: &dyn CanonicalCodecControl,
) -> Result<(runmat_canonical_codec::CanonicalEncodingProfile, u64), DelaunayVolumeMeshError> {
    prepare_volume_quality_snapshot(snapshot, binding, control)?.profile(limits, control)
}

pub fn encode_volume_quality_snapshot(
    snapshot: &VolumeQualitySnapshot<'_>,
    binding: VolumeQualityBinding,
    limits: CanonicalLimits,
    control: &dyn CanonicalCodecControl,
) -> Result<Vec<u8>, DelaunayVolumeMeshError> {
    prepare_volume_quality_snapshot(snapshot, binding, control)?.encode(limits, control)
}

/// Runs only the quality stage with the original controls. This returns a
/// diagnostic result; whole CAD/surface admission and publication remain required.
pub fn replay_volume_quality(
    bytes: &[u8],
    expected: VolumeQualityBinding,
    options: DelaunayVolumeMeshOptions,
    limits: CanonicalLimits,
    control: &dyn CanonicalCodecControl,
    cancellation: &dyn MeshingCancellationSignal,
) -> Result<VolumeQualityReplayResult, DelaunayVolumeMeshError> {
    replay_volume_quality_observed(
        bytes,
        expected,
        options,
        limits,
        control,
        cancellation,
        &NoopQualityObserver,
    )
}

/// Replays the same checked diagnostic while forwarding bounded host observation.
/// Observer failures never turn rejected geometry into accepted mesh authority.
pub fn replay_volume_quality_observed(
    bytes: &[u8],
    expected: VolumeQualityBinding,
    options: DelaunayVolumeMeshOptions,
    limits: CanonicalLimits,
    control: &dyn CanonicalCodecControl,
    cancellation: &dyn MeshingCancellationSignal,
    observer: &dyn VolumeQualityObserver,
) -> Result<VolumeQualityReplayResult, DelaunayVolumeMeshError> {
    if bytes.len() > limits.maximum_encoded_bytes {
        return Err(error(
            DelaunayVolumeMeshStage::Quality,
            DelaunayVolumeMeshErrorKind::ResourceLimit,
            "quality diagnostic exceeds its encoded byte ceiling",
        ));
    }
    let observed = super::codec::control::Control::from_codec(control);
    let control: &dyn CanonicalCodecControl = &observed;
    validate_binding(expected)?;
    validate_option_consistency(options)?;
    let expansion =
        runmat_canonical_codec::canonical_chunked_payload_materialization_resident_bytes_estimate(
            [bytes],
            control,
        )
        .map_err(|failure| codec_error(failure, control))?;
    let decoded: DecodedEnvelope = runmat_canonical_codec::decode_contract_with_control(
        PREFIX, DOMAIN, bytes, limits, control,
    )
    .map_err(|failure| codec_error(failure, control))?;
    if decoded.binding != expected || decoded.input.options != options {
        return Err(error(
            DelaunayVolumeMeshStage::Quality,
            DelaunayVolumeMeshErrorKind::InvalidOptions,
            "quality diagnostic binding or original controls differ",
        ));
    }
    let Input {
        volume,
        metric,
        fixed_nodes,
        ..
    } = decoded.input;
    let parts = volume.into_parts(options, cancellation, expansion, None)?;
    let topology = parts.topology;
    let provenance = parts.provenance;
    let faces = provenance
        .facets
        .iter()
        .map(|face| face.node_identities)
        .collect::<Vec<_>>();
    super::super::insertion::validate_conforming_volume_topology(
        &topology,
        &faces,
        options.refinement.step.insertion,
        cancellation,
    )
    .map_err(|failure| {
        error(
            DelaunayVolumeMeshStage::Quality,
            DelaunayVolumeMeshErrorKind::InvalidTopology,
            failure.to_string(),
        )
    })?;
    if fixed_nodes.len() > topology.nodes.len()
        || fixed_nodes.windows(2).any(|pair| pair[0] >= pair[1])
    {
        return Err(error(
            DelaunayVolumeMeshStage::Quality,
            DelaunayVolumeMeshErrorKind::InvalidTopology,
            "diagnostic fixed nodes are not a bounded canonical set",
        ));
    }
    for identity in &fixed_nodes {
        if topology
            .nodes
            .binary_search_by_key(identity, |node| node.identity)
            .is_err()
        {
            return Err(error(
                DelaunayVolumeMeshStage::Quality,
                DelaunayVolumeMeshErrorKind::InvalidTopology,
                "diagnostic fixed node is absent",
            ));
        }
    }
    let quality = evaluate_delaunay_volume_quality(
        &topology,
        &metric,
        &provenance,
        options.quality,
        cancellation,
    )
    .map_err(quality_error)?;
    let result = quality_convergence::converge_volume_quality(
        topology,
        quality,
        &metric,
        provenance,
        &fixed_nodes,
        options,
        quality_convergence::QualityExecution {
            cancellation,
            observer,
        },
    )?;
    Ok(VolumeQualityReplayResult {
        topology: result.topology,
        provenance: result.provenance,
        quality: result.quality,
        quality_mutations: result.quality_mutations,
    })
}

fn validate_binding(binding: VolumeQualityBinding) -> Result<(), DelaunayVolumeMeshError> {
    if [binding.request, binding.geometry, binding.implementation].contains(&StableDigest::ZERO) {
        return Err(error(
            DelaunayVolumeMeshStage::Quality,
            DelaunayVolumeMeshErrorKind::InvalidOptions,
            "quality diagnostic lacks source identity",
        ));
    }
    Ok(())
}
fn codec_error(
    failure: runmat_canonical_codec::CanonicalCodecError,
    control: &dyn CanonicalCodecControl,
) -> DelaunayVolumeMeshError {
    let kind = if control.is_cancelled() {
        DelaunayVolumeMeshErrorKind::Cancelled
    } else if failure.kind == runmat_canonical_codec::CanonicalCodecErrorKind::ResourceLimit {
        DelaunayVolumeMeshErrorKind::ResourceLimit
    } else {
        DelaunayVolumeMeshErrorKind::InvalidTopology
    };
    error(
        DelaunayVolumeMeshStage::Quality,
        kind,
        format!("quality diagnostic codec: {failure}"),
    )
}

#[cfg(test)]
mod tests;
