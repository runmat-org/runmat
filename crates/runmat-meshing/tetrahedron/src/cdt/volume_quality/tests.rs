use runmat_geometry_core::{PersistentEntityId, PersistentEntityKind};
use runmat_meshing_core::{MeshingCancellationSignal, NeverCancelled, StableDigest};
use runmat_meshing_size::metric::{
    MetricCombinationRule, MetricContribution, MetricContributionScope, MetricFieldRequest,
    MetricSourceKind, MetricTensor3,
};

use super::*;
use crate::cdt::{
    assign_delaunay_volume_regions, build_delaunay_volume_topology, DelaunayFacetProvenance,
    DelaunayTopologyOptions, DelaunayVolumeNode,
};

fn region(value: &str) -> PersistentEntityId {
    entity(PersistentEntityKind::Region, value)
}

fn entity(kind: PersistentEntityKind, value: &str) -> PersistentEntityId {
    PersistentEntityId {
        kind,
        source_topology_id: value.to_owned(),
        assembly_path: Vec::new(),
    }
}

fn metric_contexts(_topology: &DelaunayVolumeTopology) -> DelaunayVolumeProvenance {
    DelaunayVolumeProvenance {
        nodes: Vec::new(),
        segments: Vec::new(),
        facets: vec![DelaunayFacetProvenance {
            node_identities: [
                StableDigest::from_bytes([2; 32]),
                StableDigest::from_bytes([3; 32]),
                StableDigest::from_bytes([5; 32]),
            ],
            entity_ids: vec![entity(PersistentEntityKind::Face, "loaded-boundary")],
            region_ids: vec![region("outer")],
        }],
    }
}

fn topology() -> DelaunayVolumeTopology {
    let nodes = [
        [0.0, 0.0, 0.0],
        [1.0, 0.0, 0.0],
        [0.0, 1.0, 0.0],
        [0.0, 0.0, 1.0],
        [2.0, 0.0, 0.0],
    ]
    .into_iter()
    .enumerate()
    .map(|(index, coordinates_m)| DelaunayVolumeNode {
        identity: StableDigest::from_bytes([(index + 1) as u8; 32]),
        coordinates_m,
    })
    .collect();
    let topology = build_delaunay_volume_topology(
        nodes,
        vec![[0, 1, 2, 3], [1, 4, 2, 3]],
        DelaunayTopologyOptions::default(),
        &NeverCancelled,
    )
    .unwrap();
    assign_delaunay_volume_regions(
        topology,
        vec![region("inner"), region("outer")],
        DelaunayTopologyOptions::default(),
        &NeverCancelled,
    )
    .unwrap()
}

fn metric_request() -> MetricFieldRequest {
    MetricFieldRequest {
        combination: MetricCombinationRule::MostRestrictiveIntersection,
        global_metric: MetricTensor3::isotropic_length_m(2.0).unwrap(),
        maximum_grading_ratio: 1.5,
        contributions: vec![
            MetricContribution {
                source: MetricSourceKind::Region,
                scope: MetricContributionScope::Region {
                    region_id: region("outer"),
                },
                metric: MetricTensor3 {
                    xx: 3.75,
                    yy: 0.75,
                    zz: 0.75,
                    xy: 0.0,
                    xz: 0.0,
                    yz: 0.0,
                },
            },
            MetricContribution {
                source: MetricSourceKind::Face,
                scope: MetricContributionScope::Entity {
                    entity_id: entity(PersistentEntityKind::Face, "loaded-boundary"),
                },
                metric: MetricTensor3::isotropic_length_m(2.0).unwrap(),
            },
        ],
    }
}

#[test]
fn quality_resolves_region_metrics_and_selects_the_worst_cell() {
    let topology = topology();
    let contexts = metric_contexts(&topology);
    let quality = evaluate_delaunay_volume_quality(
        &topology,
        &metric_request(),
        &contexts,
        DelaunayVolumeQualityOptions {
            maximum_metric_edge_length: 1.0,
            maximum_radius_edge_ratio: 1.0,
            ..DelaunayVolumeQualityOptions::default()
        },
        &NeverCancelled,
    )
    .unwrap();

    assert_eq!(quality.tetrahedra.len(), 2);
    assert_eq!(
        quality.tetrahedra[0].active_metric_sources,
        vec![MetricSourceKind::Global]
    );
    assert_eq!(
        quality.tetrahedra[1].active_metric_sources,
        vec![
            MetricSourceKind::Global,
            MetricSourceKind::Region,
            MetricSourceKind::Face
        ]
    );
    assert_eq!(quality.tetrahedra[1].applied_metric_contribution_count, 2);
    assert!(
        quality.tetrahedra[1].maximum_metric_edge_length
            > quality.tetrahedra[0].maximum_metric_edge_length
    );
    assert_eq!(
        quality.worst_refinement_tetrahedron,
        Some(quality.tetrahedra[1].node_identities)
    );
}

#[test]
fn quality_is_deterministic_and_independently_rejects_tampering() {
    let topology = topology();
    let contexts = metric_contexts(&topology);
    let options = DelaunayVolumeQualityOptions::default();
    let first = evaluate_delaunay_volume_quality(
        &topology,
        &metric_request(),
        &contexts,
        options,
        &NeverCancelled,
    )
    .unwrap();
    let second = evaluate_delaunay_volume_quality(
        &topology,
        &metric_request(),
        &contexts,
        options,
        &NeverCancelled,
    )
    .unwrap();
    assert_eq!(first, second);

    let mut tampered = first;
    tampered.tetrahedra[0].metric_circumradius *= 2.0;
    assert_eq!(
        validate_delaunay_volume_quality(
            &topology,
            &metric_request(),
            &contexts,
            &tampered,
            options,
            &NeverCancelled,
        )
        .unwrap_err()
        .kind,
        DelaunayVolumeQualityErrorKind::InvalidQuality
    );
}

struct Cancelled;

impl MeshingCancellationSignal for Cancelled {
    fn is_cancelled(&self) -> bool {
        true
    }
}

#[test]
fn quality_rejects_unassigned_topology_limits_bad_policy_and_cancellation() {
    let assigned = topology();
    let contexts = metric_contexts(&assigned);
    let mut unassigned = assigned.clone();
    unassigned.tetrahedra[0].region_id = None;
    assert_eq!(
        evaluate_delaunay_volume_quality(
            &unassigned,
            &metric_request(),
            &contexts,
            DelaunayVolumeQualityOptions::default(),
            &NeverCancelled,
        )
        .unwrap_err()
        .kind,
        DelaunayVolumeQualityErrorKind::InvalidTopology
    );
    assert_eq!(
        evaluate_delaunay_volume_quality(
            &assigned,
            &metric_request(),
            &contexts,
            DelaunayVolumeQualityOptions {
                maximum_tetrahedra: 1,
                ..DelaunayVolumeQualityOptions::default()
            },
            &NeverCancelled,
        )
        .unwrap_err()
        .kind,
        DelaunayVolumeQualityErrorKind::ResourceLimit
    );
    assert_eq!(
        evaluate_delaunay_volume_quality(
            &assigned,
            &metric_request(),
            &contexts,
            DelaunayVolumeQualityOptions {
                maximum_metric_edge_length: 0.0,
                ..DelaunayVolumeQualityOptions::default()
            },
            &NeverCancelled,
        )
        .unwrap_err()
        .kind,
        DelaunayVolumeQualityErrorKind::InvalidOptions
    );
    assert_eq!(
        evaluate_delaunay_volume_quality(
            &assigned,
            &metric_request(),
            &contexts,
            DelaunayVolumeQualityOptions {
                cancellation_check_interval: 1,
                ..DelaunayVolumeQualityOptions::default()
            },
            &Cancelled,
        )
        .unwrap_err()
        .kind,
        DelaunayVolumeQualityErrorKind::Cancelled
    );

    let mut invalid_contexts = contexts;
    invalid_contexts.facets[0].entity_ids.clear();
    assert_eq!(
        evaluate_delaunay_volume_quality(
            &assigned,
            &metric_request(),
            &invalid_contexts,
            DelaunayVolumeQualityOptions::default(),
            &NeverCancelled,
        )
        .unwrap_err()
        .kind,
        DelaunayVolumeQualityErrorKind::InvalidMetricContext
    );
}
