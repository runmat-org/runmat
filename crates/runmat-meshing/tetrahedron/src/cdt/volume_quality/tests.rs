use runmat_geometry_core::{PersistentEntityId, PersistentEntityKind};
use runmat_meshing_core::{MeshingCancellationSignal, NeverCancelled, StableDigest};
use runmat_meshing_size::metric::{
    MetricCombinationRule, MetricContribution, MetricContributionScope, MetricFieldRequest,
    MetricSourceKind, MetricTensor3,
};

use super::*;
#[path = "tests/thresholds.rs"]
mod thresholds;

#[test]
fn violation_priority_ignores_only_low_order_metric_noise() {
    let smaller = StableDigest::from_bytes([1; 32]);
    let larger = StableDigest::from_bytes([2; 32]);
    let identity = |first| [first, smaller, larger, StableDigest::from_bytes([3; 32])];
    let geometry = |value| [[value; 3]; 4];
    let base = 2.0_f64;
    let noisy = f64::from_bits(base.to_bits() + 8);

    assert!(compare_violation_priority(
        base,
        geometry(1),
        identity(smaller),
        noisy,
        geometry(2),
        identity(larger),
    )
    .is_gt());
    assert!(compare_violation_priority(
        2.01,
        geometry(2),
        identity(larger),
        base,
        geometry(1),
        identity(smaller),
    )
    .is_gt());
}
use crate::cdt::{
    assign_delaunay_volume_regions, build_delaunay_volume_topology, DelaunayFacetProvenance,
    DelaunayTopologyOptions, DelaunayVolumeNode,
};

pub(super) fn region(value: &str) -> PersistentEntityId {
    entity(PersistentEntityKind::Region, value)
}

fn entity(kind: PersistentEntityKind, value: &str) -> PersistentEntityId {
    PersistentEntityId::new(kind, value.to_owned(), Vec::new())
}

pub(super) fn metric_contexts(_topology: &DelaunayVolumeTopology) -> DelaunayVolumeProvenance {
    DelaunayVolumeProvenance {
        nodes: Vec::new(),
        segments: Vec::new(),
        facets: vec![DelaunayFacetProvenance {
            node_identities: [
                StableDigest::from_bytes([2; 32]),
                StableDigest::from_bytes([3; 32]),
                StableDigest::from_bytes([5; 32]),
            ],
            chart_id: StableDigest::from_bytes([40; 32]),
            entity_ids: vec![entity(PersistentEntityKind::Face, "loaded-boundary")],
            region_ids: vec![region("outer")],
        }],
    }
}

pub(super) fn topology() -> DelaunayVolumeTopology {
    let nodes = [
        [0.0, 0.0, 0.0],
        [1.0, 0.0, 0.0],
        [0.0, 1.0, 0.0],
        [0.0, 0.0, 1.0],
        [2.0, 0.0, 0.0],
    ]
    .into_iter()
    .enumerate()
    .map(|(index, coordinates_m)| {
        DelaunayVolumeNode::explicit(
            StableDigest::from_bytes([(index + 1) as u8; 32]),
            coordinates_m,
        )
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

fn sliver_topology() -> DelaunayVolumeTopology {
    let nodes = [
        [0.0, 0.0, 0.0],
        [1.0, 0.0, 0.0],
        [0.0, 1.0, 0.0],
        [0.5, 0.5, 1.0e-6],
    ]
    .into_iter()
    .enumerate()
    .map(|(index, coordinates_m)| {
        DelaunayVolumeNode::explicit(
            StableDigest::from_bytes([(index + 11) as u8; 32]),
            coordinates_m,
        )
    })
    .collect();
    let topology = build_delaunay_volume_topology(
        nodes,
        vec![[0, 1, 2, 3]],
        DelaunayTopologyOptions::default(),
        &NeverCancelled,
    )
    .unwrap();
    assign_delaunay_volume_regions(
        topology,
        vec![region("sliver-region")],
        DelaunayTopologyOptions::default(),
        &NeverCancelled,
    )
    .unwrap()
}

pub(super) fn metric_request() -> MetricFieldRequest {
    MetricFieldRequest {
        entity_supports: Vec::new(),
        located_sources: Vec::new(),
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
        cells_across_gap: None,
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
    assert_eq!(quality.tetrahedra[0].protected_facet_count, 0);
    assert_eq!(quality.tetrahedra[1].protected_facet_count, 1);
    assert_eq!(quality.tetrahedra[0].clipped_metric_contribution_count, 1);
    assert_eq!(quality.tetrahedra[1].clipped_metric_contribution_count, 0);
    let characteristic_lengths = [0, 1].map(|index| {
        quality.tetrahedra[index]
            .resolved_metric
            .conservative_minimum_length_m()
            .unwrap()
    });
    assert!(
        characteristic_lengths[0].max(characteristic_lengths[1])
            <= 1.5 * characteristic_lengths[0].min(characteristic_lengths[1])
    );
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

#[test]
fn anisotropic_metric_reports_a_sliver_without_scheduling_size_refinement() {
    let topology = sliver_topology();
    let provenance = DelaunayVolumeProvenance {
        nodes: Vec::new(),
        segments: Vec::new(),
        facets: Vec::new(),
    };
    let request = MetricFieldRequest {
        entity_supports: Vec::new(),
        located_sources: Vec::new(),
        combination: MetricCombinationRule::MostRestrictiveIntersection,
        global_metric: MetricTensor3 {
            xx: 2.0,
            yy: 1.5,
            zz: 0.75,
            xy: 0.25,
            xz: 0.1,
            yz: 0.05,
        },
        maximum_grading_ratio: 1.5,
        contributions: Vec::new(),
        cells_across_gap: None,
    };
    let options = DelaunayVolumeQualityOptions {
        maximum_metric_edge_length: 1.0e12,
        maximum_radius_edge_ratio: 1.0e12,
        minimum_metric_scaled_jacobian: 0.1,
        ..DelaunayVolumeQualityOptions::default()
    };
    let quality = evaluate_delaunay_volume_quality(
        &topology,
        &request,
        &provenance,
        options,
        &NeverCancelled,
    )
    .unwrap();
    let tetrahedron = &quality.tetrahedra[0];

    assert!(tetrahedron.maximum_metric_edge_length / options.maximum_metric_edge_length < 1.0);
    assert!(tetrahedron.metric_radius_edge_ratio / options.maximum_radius_edge_ratio < 1.0);
    assert!(tetrahedron.metric_scaled_jacobian < options.minimum_metric_scaled_jacobian);
    assert_eq!(quality.worst_refinement_tetrahedron, None);
    assert_eq!(
        quality.minimum_metric_scaled_jacobian,
        tetrahedron.metric_scaled_jacobian
    );

    let mut tampered = quality;
    tampered.tetrahedra[0].metric_scaled_jacobian *= 2.0;
    assert_eq!(
        validate_delaunay_volume_quality(
            &topology,
            &request,
            &provenance,
            &tampered,
            options,
            &NeverCancelled,
        )
        .unwrap_err()
        .kind,
        DelaunayVolumeQualityErrorKind::InvalidQuality
    );
}

#[test]
fn constrained_boundary_radius_edge_is_reported_but_shape_drives_admission() {
    assert!(evaluation::radius_edge_is_refinement_owned(0));
    for boundary_node_count in 1..=4 {
        assert!(!evaluation::radius_edge_is_refinement_owned(
            boundary_node_count
        ));
    }
    let topology = sliver_topology();
    let boundary_identities = [11_u8, 12, 13].map(|value| StableDigest::from_bytes([value; 32]));
    let provenance = DelaunayVolumeProvenance {
        nodes: Vec::new(),
        segments: Vec::new(),
        facets: vec![DelaunayFacetProvenance {
            node_identities: boundary_identities,
            chart_id: StableDigest::from_bytes([90; 32]),
            entity_ids: vec![entity(PersistentEntityKind::Face, "sliver-boundary")],
            region_ids: vec![region("sliver-region")],
        }],
    };
    let options = DelaunayVolumeQualityOptions {
        maximum_metric_edge_length: 1.0e12,
        maximum_radius_edge_ratio: 2.0,
        minimum_metric_scaled_jacobian: 1.0e-12,
        ..DelaunayVolumeQualityOptions::default()
    };
    let quality = evaluate_delaunay_volume_quality(
        &topology,
        &MetricFieldRequest {
            entity_supports: Vec::new(),
            located_sources: Vec::new(),
            combination: MetricCombinationRule::MostRestrictiveIntersection,
            global_metric: MetricTensor3::isotropic_length_m(1.0).unwrap(),
            maximum_grading_ratio: 1.5,
            contributions: Vec::new(),
            cells_across_gap: None,
        },
        &provenance,
        options,
        &NeverCancelled,
    )
    .unwrap();

    let tetrahedron = &quality.tetrahedra[0];
    assert_eq!(tetrahedron.boundary_node_count, 3);
    assert!(tetrahedron.metric_radius_edge_ratio > options.maximum_radius_edge_ratio);
    assert!(tetrahedron.refinement_violation_ratio <= 1.0);
    assert!(tetrahedron.optimization_violation_ratio <= 1.0);
    assert!(quality.worst_refinement_tetrahedron.is_none());
    assert!(quality.worst_optimization_tetrahedron.is_none());
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
                minimum_metric_scaled_jacobian: 1.1,
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

fn protected_surface(facets: &[[runmat_meshing_core::StableDigest; 3]]) -> super::ProtectedFacets {
    let edges = facets
        .iter()
        .flat_map(|&[a, b, c]| [[a, b], [a, c], [b, c]])
        .collect();
    super::ProtectedFacets {
        facets: facets.iter().copied().collect(),
        edges,
    }
}

#[test]
fn size_criterion_measures_only_edges_the_volume_stage_can_split() {
    use runmat_meshing_core::StableDigest;
    let identity = |byte: u8| StableDigest::from_bytes([byte; 32]);
    let tetrahedron = [identity(1), identity(2), identity(3), identity(4)];
    // The facet opposite vertex 3 (vertices 0, 1, 2) belongs to the published surface.
    let mut facet = [identity(1), identity(2), identity(3)];
    facet.sort_unstable();
    let protected = protected_surface(&[facet]);
    let mask = super::protected_edge_mask(tetrahedron, &protected);
    assert_eq!(
        mask, 0b001011,
        "edges (0,1), (0,2) and (1,2) lie on the protected facet"
    );
    assert_eq!(
        super::protected_edge_mask(tetrahedron, &protected_surface(&[])),
        0
    );
    // A surface edge is the surface's even when this tetrahedron holds neither surface triangle
    // on it: here edge (0,1) belongs to a facet with a vertex outside the tetrahedron.
    let mut neighbor = [identity(1), identity(2), identity(9)];
    neighbor.sort_unstable();
    assert_eq!(
        super::protected_edge_mask(tetrahedron, &protected_surface(&[neighbor])),
        0b000001
    );

    // A long surface edge (0,1) of 1.25 metric units over a 0.5-unit-high apex: the surface owns
    // the long edge, so only the interior edges count toward refinement.
    let points = [
        [0.0, 0.0, 0.0],
        [1.25, 0.0, 0.0],
        [0.6, 0.5, 0.0],
        [0.6, 0.2, 0.5],
    ];
    let metric = runmat_meshing_size::metric::MetricTensor3::isotropic_length_m(1.0).unwrap();
    let all = super::evaluation::volume_owned_maximum_metric_edge_length(points, metric, 0);
    assert!((all - 1.25).abs() < 1e-12);
    let owned = super::evaluation::volume_owned_maximum_metric_edge_length(points, metric, mask);
    assert!(owned < 1.0, "{owned}");
    let options = super::DelaunayVolumeQualityOptions::default();
    let (refinement, _) = super::violation_ratios(owned, 1.0, 1.0, 3, options);
    assert!(refinement <= 1.0);
    let (refinement, _) = super::violation_ratios(all, 1.0, 1.0, 3, options);
    assert!(refinement > 1.0);
}
