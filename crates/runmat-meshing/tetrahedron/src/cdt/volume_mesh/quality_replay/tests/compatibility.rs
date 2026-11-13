//! A fixed boundary can make a requested volume metric quality impossible.
//! The original fixtures and target remain unchanged; this is not a convergence
//! or whole-study qualification claim.
use super::*;
use crate::cdt::volume_quality::{boundary_compatibility, working::PreparedLocalQuality};
pub(super) fn assert_incompatible(bytes: &[u8]) {
    let decoded: DecodedEnvelope = runmat_canonical_codec::decode_contract_with_control(
        PREFIX, DOMAIN, bytes, LIMITS, &Control,
    )
    .unwrap();
    let input = decoded.input;
    assert_eq!(input.options.quality.minimum_metric_scaled_jacobian, 0.01);
    let expansion =
        runmat_canonical_codec::canonical_chunked_payload_materialization_resident_bytes_estimate(
            [bytes],
            &Control,
        )
        .unwrap();
    let parts = input
        .volume
        .into_parts(input.options, &NeverCancelled, expansion, None)
        .unwrap();
    let prepared =
        PreparedLocalQuality::prepare(&input.metric, &parts.provenance, input.options.quality)
            .unwrap();
    let conflict = boundary_compatibility::certify(&parts.topology, &prepared, &NeverCancelled)
        .unwrap()
        .expect("outward bound proves incompatibility");
    assert!(conflict.jacobian_upper_bound < 0.01);
    // More volume-search work cannot repair a fixed-boundary incompatibility.
    // Explicit rebinding preserves the original bytes and the original target.
    for cap in [
        input.options.sliver.maximum_candidate_evaluations_per_pass,
        1_000_000,
    ] {
        let mut options = input.options;
        options.sliver.maximum_candidate_evaluations_per_pass = cap;
        let snapshot = VolumeQualitySnapshot {
            topology: &parts.topology,
            provenance: &parts.provenance,
            metric: &input.metric,
            fixed_nodes: &input.fixed_nodes,
            options,
        };
        let encoded =
            encode_volume_quality_snapshot(&snapshot, binding(), LIMITS, &Control).unwrap();
        let failure = replay_volume_quality(
            &encoded,
            binding(),
            options,
            LIMITS,
            &Control,
            &NeverCancelled,
        )
        .err()
        .expect("fixed boundary is incompatible");
        assert_eq!(
            failure.kind,
            DelaunayVolumeMeshErrorKind::UnsatisfiableConstraint
        );
        assert!(failure
            .reason
            .contains("certified metric scaled-Jacobian upper bound"));
        assert!(failure.reason.contains("surface remeshing is required"));
    }
}
#[test]
fn an_unknown_vertex_metric_prevents_a_common_metric_claim() {
    use runmat_meshing_size::metric::{
        MetricContribution, MetricContributionScope, MetricSourceKind, MetricTensor3,
    };
    let bytes = include_bytes!("../fixtures/terminal-noncommuting-collar.cbor");
    let decoded: DecodedEnvelope = runmat_canonical_codec::decode_contract_with_control(
        PREFIX, DOMAIN, bytes, LIMITS, &Control,
    )
    .unwrap();
    let mut input = decoded.input;
    let parts = input
        .volume
        .into_parts(input.options, &NeverCancelled, runmat_canonical_codec::canonical_chunked_payload_materialization_resident_bytes_estimate([bytes.as_slice()], &Control).unwrap(), None)
        .unwrap();
    let entity = parts
        .provenance
        .nodes
        .iter()
        .flat_map(|binding| binding.entity_ids.iter())
        .find(|entity| entity.kind == runmat_geometry_core::PersistentEntityKind::Vertex)
        .unwrap()
        .clone();
    input.metric.contributions.insert(
        0,
        MetricContribution {
            source: MetricSourceKind::Point,
            scope: MetricContributionScope::Entity { entity_id: entity },
            metric: MetricTensor3 {
                xx: 8.,
                yy: 8.,
                zz: 8.,
                xy: 0.,
                xz: 0.,
                yz: 0.,
            },
        },
    );
    let prepared =
        PreparedLocalQuality::prepare(&input.metric, &parts.provenance, input.options.quality)
            .unwrap();
    assert!(
        boundary_compatibility::certify(&parts.topology, &prepared, &NeverCancelled)
            .unwrap()
            .is_none()
    );
}
#[test]
fn compatibility_cancellation_is_not_relabelled_as_indeterminate() {
    struct Cancelled;
    impl runmat_meshing_core::MeshingCancellationSignal for Cancelled {
        fn is_cancelled(&self) -> bool {
            true
        }
    }
    let bytes = include_bytes!("../fixtures/terminal-noncommuting-collar.cbor");
    let decoded: DecodedEnvelope = runmat_canonical_codec::decode_contract_with_control(
        PREFIX, DOMAIN, bytes, LIMITS, &Control,
    )
    .unwrap();
    let input = decoded.input;
    let parts = input
        .volume
        .into_parts(input.options, &NeverCancelled, runmat_canonical_codec::canonical_chunked_payload_materialization_resident_bytes_estimate([bytes.as_slice()], &Control).unwrap(), None)
        .unwrap();
    let prepared =
        PreparedLocalQuality::prepare(&input.metric, &parts.provenance, input.options.quality)
            .unwrap();
    assert_eq!(
        boundary_compatibility::certify(&parts.topology, &prepared, &Cancelled)
            .unwrap_err()
            .kind,
        crate::cdt::DelaunayVolumeQualityErrorKind::Cancelled
    );
}
