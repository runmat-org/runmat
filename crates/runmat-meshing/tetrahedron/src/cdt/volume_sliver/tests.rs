use runmat_geometry_core::{PersistentEntityId, PersistentEntityKind};
use runmat_meshing_core::{MeshingCancellationSignal, NeverCancelled, StableDigest};
use runmat_meshing_size::metric::{MetricCombinationRule, MetricFieldRequest, MetricTensor3};

use super::*;
use crate::cdt::{
    assign_delaunay_volume_regions, build_delaunay_volume_topology,
    evaluate_delaunay_volume_quality, refine_delaunay_volume, validate_delaunay_volume_refinement,
    DelaunayFacetProvenance, DelaunayTopologyOptions, DelaunayVolumeProvenance,
    DelaunayVolumeQualityOptions, DelaunayVolumeRefinementInput, DelaunayVolumeRefinementMutation,
    DelaunayVolumeRefinementOptions, DelaunayVolumeRefinementStepErrorKind,
};

fn entity(kind: PersistentEntityKind, name: &str) -> PersistentEntityId {
    PersistentEntityId {
        kind,
        source_topology_id: name.to_owned(),
        assembly_path: Vec::new(),
    }
}

fn region() -> PersistentEntityId {
    entity(PersistentEntityKind::Region, "solid")
}

fn node(identity: u8, coordinates_m: [f64; 3]) -> DelaunayVolumeNode {
    DelaunayVolumeNode {
        identity: StableDigest::from_bytes([identity; 32]),
        coordinates_m,
    }
}

fn metric() -> MetricFieldRequest {
    MetricFieldRequest {
        combination: MetricCombinationRule::MostRestrictiveIntersection,
        global_metric: MetricTensor3::isotropic_length_m(1.0).unwrap(),
        maximum_grading_ratio: 1.5,
        contributions: Vec::new(),
    }
}

fn quality_options() -> DelaunayVolumeQualityOptions {
    DelaunayVolumeQualityOptions {
        maximum_metric_edge_length: 1.0e12,
        maximum_radius_edge_ratio: 1.0e12,
        minimum_metric_scaled_jacobian: 0.1,
        ..DelaunayVolumeQualityOptions::default()
    }
}

fn interior_sliver_fixture() -> (DelaunayVolumeTopology, DelaunayVolumeProvenance) {
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

fn quality(
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
fn treatment_relocates_only_the_interior_node_and_eliminates_slivers() {
    let (topology, provenance) = interior_sliver_fixture();
    let request = metric();
    let initial_quality = quality(&topology, &provenance);
    let input = DelaunayVolumeRefinementInput {
        topology: &topology,
        metric_request: &request,
        provenance: &provenance,
        quality: &initial_quality,
        quality_options: quality_options(),
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
    assert_eq!(first.relocations.len(), 1);
    assert_eq!(
        first.relocations[0].source_node_identity,
        StableDigest::from_bytes([5; 32])
    );
    assert_eq!(first.topology.nodes.len(), topology.nodes.len());
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
fn refinement_driver_applies_sliver_treatment_before_steiner_insertion() {
    let (topology, provenance) = interior_sliver_fixture();
    let request = metric();
    let initial_quality = quality(&topology, &provenance);
    let input = DelaunayVolumeRefinementInput {
        topology: &topology,
        metric_request: &request,
        provenance: &provenance,
        quality: &initial_quality,
        quality_options: quality_options(),
    };
    let options = DelaunayVolumeRefinementOptions {
        maximum_insertions: 1,
        ..DelaunayVolumeRefinementOptions::default()
    };
    let refinement = refine_delaunay_volume(input, options, &NeverCancelled).unwrap();

    assert_eq!(refinement.mutations.len(), 1);
    assert!(matches!(
        refinement.mutations[0],
        DelaunayVolumeRefinementMutation::Relocated(_)
    ));
    assert!(refinement.quality.worst_refinement_tetrahedron.is_none());
    validate_delaunay_volume_refinement(input, &refinement, options, &NeverCancelled).unwrap();

    let mut tampered = refinement;
    let DelaunayVolumeRefinementMutation::Relocated(relocation) = &mut tampered.mutations[0] else {
        unreachable!();
    };
    relocation.replacement_node.coordinates_m[0] += 0.01;
    assert_eq!(
        validate_delaunay_volume_refinement(input, &tampered, options, &NeverCancelled)
            .unwrap_err()
            .kind,
        DelaunayVolumeRefinementStepErrorKind::InvalidTopology
    );
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
    };
    let no_op = treat_delaunay_volume_slivers(
        converged_input,
        DelaunayVolumeSliverOptions::default(),
        &NeverCancelled,
    )
    .unwrap();
    assert_eq!(no_op.topology, topology);
    assert_eq!(no_op.quality, permissive_quality);
    assert!(no_op.relocations.is_empty());

    let input = DelaunayVolumeRefinementInput {
        topology: &topology,
        metric_request: &request,
        provenance: &provenance,
        quality: &initial_quality,
        quality_options: quality_options(),
    };
    let mut treatment = treat_delaunay_volume_slivers(
        input,
        DelaunayVolumeSliverOptions::default(),
        &NeverCancelled,
    )
    .unwrap();
    treatment.relocations[0]
        .source_tetrahedron_node_identities
        .swap(0, 1);
    assert_eq!(
        validate_delaunay_volume_sliver_treatment(
            input,
            &treatment,
            DelaunayVolumeSliverOptions::default(),
            &NeverCancelled,
        )
        .unwrap_err()
        .kind,
        DelaunayVolumeSliverErrorKind::InvalidInput
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
    };
    assert_eq!(
        treat_delaunay_volume_slivers(
            input,
            DelaunayVolumeSliverOptions::default(),
            &NeverCancelled,
        )
        .unwrap_err()
        .kind,
        DelaunayVolumeSliverErrorKind::NoAdmissibleRelocation
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
