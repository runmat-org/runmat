use runmat_geometry_core::{PersistentEntityId, PersistentEntityKind};
use runmat_meshing_core::{MeshingCancellationSignal, NeverCancelled, StableDigest};
use runmat_meshing_size::metric::{MetricCombinationRule, MetricFieldRequest, MetricTensor3};

use super::*;
use crate::cdt::{
    assign_delaunay_volume_regions, build_delaunay_volume_topology,
    evaluate_delaunay_volume_quality, DelaunayFacetProvenance, DelaunayTopologyOptions,
    DelaunayVolumeProvenance, DelaunayVolumeQualityOptions, DelaunayVolumeRefinementInput,
};

fn entity(kind: PersistentEntityKind, name: &str) -> PersistentEntityId {
    PersistentEntityId::new(kind, name.to_owned(), Vec::new())
}

fn region() -> PersistentEntityId {
    entity(PersistentEntityKind::Region, "solid")
}

fn node(identity: u8, coordinates_m: [f64; 3]) -> DelaunayVolumeNode {
    DelaunayVolumeNode::explicit(StableDigest::from_bytes([identity; 32]), coordinates_m)
}

pub(super) fn metric() -> MetricFieldRequest {
    MetricFieldRequest {
        entity_supports: Vec::new(),
        located_sources: Vec::new(),
        combination: MetricCombinationRule::MostRestrictiveIntersection,
        global_metric: MetricTensor3::isotropic_length_m(1.0).unwrap(),
        maximum_grading_ratio: 1.5,
        contributions: Vec::new(),
        cells_across_gap: None,
    }
}

pub(super) fn quality_options() -> DelaunayVolumeQualityOptions {
    DelaunayVolumeQualityOptions {
        maximum_metric_edge_length: 1.0e12,
        maximum_radius_edge_ratio: 1.0e12,
        minimum_metric_scaled_jacobian: 0.1,
        ..DelaunayVolumeQualityOptions::default()
    }
}

pub(super) fn interior_sliver_fixture() -> (DelaunayVolumeTopology, DelaunayVolumeProvenance) {
    let nodes = vec![
        node(1, [0.0, 0.0, 0.0]),
        node(2, [1.0, 0.0, 0.0]),
        node(3, [0.0, 1.0, 0.0]),
        node(4, [0.0, 0.0, 1.0]),
        node(5, [0.3, 0.3, 1.0e-6]),
    ];
    let topology = build_delaunay_volume_topology(
        nodes,
        vec![[0, 1, 2, 4], [0, 1, 3, 4], [0, 2, 3, 4], [1, 2, 3, 4]],
        DelaunayTopologyOptions::default(),
        &NeverCancelled,
    )
    .unwrap();
    let topology = assign_delaunay_volume_regions(
        topology,
        vec![region(); 4],
        DelaunayTopologyOptions::default(),
        &NeverCancelled,
    )
    .unwrap();
    let mut faces = [[1, 2, 3], [1, 2, 4], [1, 3, 4], [2, 3, 4]]
        .into_iter()
        .enumerate()
        .map(|(index, values)| DelaunayFacetProvenance {
            node_identities: values.map(|value| StableDigest::from_bytes([value; 32])),
            chart_id: StableDigest::from_bytes([(index + 40) as u8; 32]),
            entity_ids: vec![entity(
                PersistentEntityKind::Face,
                &format!("outer-face-{index}"),
            )],
            region_ids: vec![region()],
        })
        .collect::<Vec<_>>();
    faces.sort_by_key(|face| face.node_identities);
    (
        topology,
        DelaunayVolumeProvenance {
            nodes: Vec::new(),
            segments: Vec::new(),
            facets: faces,
        },
    )
}

pub(super) fn quality(
    topology: &DelaunayVolumeTopology,
    provenance: &DelaunayVolumeProvenance,
) -> DelaunayVolumeQuality {
    evaluate_delaunay_volume_quality(
        topology,
        &metric(),
        provenance,
        quality_options(),
        &NeverCancelled,
    )
    .unwrap()
}

#[test]
fn treatment_removes_a_redundant_interior_node_and_eliminates_slivers() {
    let (topology, provenance) = interior_sliver_fixture();
    let request = metric();
    let initial_quality = quality(&topology, &provenance);
    let input = DelaunayVolumeRefinementInput {
        topology: &topology,
        metric_request: &request,
        provenance: &provenance,
        quality: &initial_quality,
        quality_options: quality_options(),
        fixed_node_identities: &[],
    };
    let first = treat_delaunay_volume_slivers(
        input,
        DelaunayVolumeSliverOptions::default(),
        &NeverCancelled,
    )
    .unwrap();
    let second = treat_delaunay_volume_slivers(
        input,
        DelaunayVolumeSliverOptions::default(),
        &NeverCancelled,
    )
    .unwrap();

    assert_eq!(first, second);
    assert_eq!(first.mutations.len(), 1);
    let DelaunayVolumeSliverMutation::Deleted(deletion) = first.mutations[0] else {
        panic!("fixture should improve by removing its redundant interior node");
    };
    assert_eq!(
        deletion.source_node_identity,
        StableDigest::from_bytes([5; 32])
    );
    assert_eq!(first.topology.nodes.len() + 1, topology.nodes.len());
    assert!(
        first.quality.minimum_metric_scaled_jacobian
            > initial_quality.minimum_metric_scaled_jacobian
    );
    assert!(first
        .quality
        .tetrahedra
        .iter()
        .all(|tetrahedron| tetrahedron.metric_scaled_jacobian >= 0.1));
    for facet in &provenance.facets {
        assert!(first.topology.tetrahedra.iter().any(|tetrahedron| {
            let identities = tetrahedron
                .vertex_indices
                .map(|vertex| first.topology.nodes[vertex as usize].identity);
            facet
                .node_identities
                .iter()
                .all(|identity| identities.contains(identity))
        }));
    }
}

#[test]
fn deletion_preserves_existing_size_progress_without_requiring_global_convergence() {
    let (topology, provenance) = interior_sliver_fixture();
    let request = metric();
    let options = DelaunayVolumeQualityOptions {
        maximum_metric_edge_length: 1.0,
        ..quality_options()
    };
    let initial_quality = evaluate_delaunay_volume_quality(
        &topology,
        &request,
        &provenance,
        options,
        &NeverCancelled,
    )
    .unwrap();
    assert!(initial_quality.tetrahedra.iter().any(|tetrahedron| {
        tetrahedron.maximum_metric_edge_length > options.maximum_metric_edge_length
    }));

    let treatment = treat_delaunay_volume_slivers(
        DelaunayVolumeRefinementInput {
            topology: &topology,
            metric_request: &request,
            provenance: &provenance,
            quality: &initial_quality,
            quality_options: options,
            fixed_node_identities: &[],
        },
        DelaunayVolumeSliverOptions::default(),
        &NeverCancelled,
    )
    .unwrap();

    assert!(matches!(
        treatment.mutations.as_slice(),
        [DelaunayVolumeSliverMutation::Deleted(_)]
    ));
    let initial_maximum = initial_quality
        .tetrahedra
        .iter()
        .map(|tetrahedron| tetrahedron.maximum_metric_edge_length)
        .max_by(f64::total_cmp)
        .unwrap();
    let treated_maximum = treatment
        .quality
        .tetrahedra
        .iter()
        .map(|tetrahedron| tetrahedron.maximum_metric_edge_length)
        .max_by(f64::total_cmp)
        .unwrap();
    assert!(treated_maximum <= initial_maximum);
}

#[test]
fn treatment_is_a_checked_noop_and_rejects_tampering() {
    let (topology, provenance) = interior_sliver_fixture();
    let request = metric();
    let initial_quality = quality(&topology, &provenance);
    let permissive_options = DelaunayVolumeQualityOptions {
        minimum_metric_scaled_jacobian: 1.0e-8,
        ..quality_options()
    };
    let permissive_quality = evaluate_delaunay_volume_quality(
        &topology,
        &request,
        &provenance,
        permissive_options,
        &NeverCancelled,
    )
    .unwrap();
    let converged_input = DelaunayVolumeRefinementInput {
        topology: &topology,
        metric_request: &request,
        provenance: &provenance,
        quality: &permissive_quality,
        quality_options: permissive_options,
        fixed_node_identities: &[],
    };
    let no_op = treat_delaunay_volume_slivers(
        converged_input,
        DelaunayVolumeSliverOptions::default(),
        &NeverCancelled,
    )
    .unwrap();
    assert_eq!(no_op.topology, topology);
    assert_eq!(no_op.quality, permissive_quality);
    assert!(no_op.mutations.is_empty());

    let input = DelaunayVolumeRefinementInput {
        topology: &topology,
        metric_request: &request,
        provenance: &provenance,
        quality: &initial_quality,
        quality_options: quality_options(),
        fixed_node_identities: &[],
    };
    let mut treatment = treat_delaunay_volume_slivers(
        input,
        DelaunayVolumeSliverOptions::default(),
        &NeverCancelled,
    )
    .unwrap();
    match &mut treatment.mutations[0] {
        DelaunayVolumeSliverMutation::Relocated(relocation) => {
            relocation.source_node_identity = StableDigest::from_bytes([9; 32]);
        }
        DelaunayVolumeSliverMutation::Deleted(deletion) => {
            deletion.source_node_identity = StableDigest::from_bytes([9; 32]);
        }
        DelaunayVolumeSliverMutation::Reconnected(_) => {
            panic!("fixture should use an interior vertex mutation")
        }
        DelaunayVolumeSliverMutation::Inserted(_) => {
            panic!("fixture should use an interior vertex mutation")
        }
    }
    assert_eq!(
        validate_delaunay_volume_sliver_treatment(
            input,
            &treatment,
            DelaunayVolumeSliverOptions::default(),
            &NeverCancelled,
        )
        .unwrap_err()
        .kind,
        DelaunayVolumeSliverErrorKind::InvalidTopology
    );
}

struct Cancelled;

impl MeshingCancellationSignal for Cancelled {
    fn is_cancelled(&self) -> bool {
        true
    }
}

#[test]
fn treatment_reports_immovable_slivers_limits_and_cancellation() {
    let nodes = vec![
        node(11, [0.0, 0.0, 0.0]),
        node(12, [1.0, 0.0, 0.0]),
        node(13, [0.0, 1.0, 0.0]),
        node(14, [0.5, 0.5, 1.0e-6]),
    ];
    let topology = build_delaunay_volume_topology(
        nodes,
        vec![[0, 1, 2, 3]],
        DelaunayTopologyOptions::default(),
        &NeverCancelled,
    )
    .unwrap();
    let topology = assign_delaunay_volume_regions(
        topology,
        vec![region()],
        DelaunayTopologyOptions::default(),
        &NeverCancelled,
    )
    .unwrap();
    let provenance = DelaunayVolumeProvenance {
        nodes: Vec::new(),
        segments: Vec::new(),
        facets: Vec::new(),
    };
    let request = metric();
    let initial_quality = quality(&topology, &provenance);
    let input = DelaunayVolumeRefinementInput {
        topology: &topology,
        metric_request: &request,
        provenance: &provenance,
        quality: &initial_quality,
        quality_options: quality_options(),
        fixed_node_identities: &[],
    };
    assert_eq!(
        treat_delaunay_volume_slivers(
            input,
            DelaunayVolumeSliverOptions::default(),
            &NeverCancelled,
        )
        .unwrap_err()
        .kind,
        DelaunayVolumeSliverErrorKind::NoAdmissibleOptimization
    );
    assert_eq!(
        treat_delaunay_volume_slivers(
            input,
            DelaunayVolumeSliverOptions {
                maximum_passes: 0,
                ..DelaunayVolumeSliverOptions::default()
            },
            &NeverCancelled,
        )
        .unwrap_err()
        .kind,
        DelaunayVolumeSliverErrorKind::InvalidOptions
    );
    assert_eq!(
        treat_delaunay_volume_slivers(input, DelaunayVolumeSliverOptions::default(), &Cancelled,)
            .unwrap_err()
            .kind,
        DelaunayVolumeSliverErrorKind::Cancelled
    );

    let (movable_topology, movable_provenance) = interior_sliver_fixture();
    let strict_options = DelaunayVolumeQualityOptions {
        minimum_metric_scaled_jacobian: 0.9,
        ..quality_options()
    };
    let strict_quality = evaluate_delaunay_volume_quality(
        &movable_topology,
        &request,
        &movable_provenance,
        strict_options,
        &NeverCancelled,
    )
    .unwrap();
    let strict_input = DelaunayVolumeRefinementInput {
        topology: &movable_topology,
        metric_request: &request,
        provenance: &movable_provenance,
        quality: &strict_quality,
        quality_options: strict_options,
        fixed_node_identities: &[],
    };
    assert_eq!(
        treat_delaunay_volume_slivers(
            strict_input,
            DelaunayVolumeSliverOptions {
                maximum_passes: 1,
                ..DelaunayVolumeSliverOptions::default()
            },
            &NeverCancelled,
        )
        .unwrap_err()
        .kind,
        DelaunayVolumeSliverErrorKind::ResourceLimit
    );
}
