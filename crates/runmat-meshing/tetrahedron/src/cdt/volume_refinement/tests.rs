use runmat_geometry_core::{PersistentEntityId, PersistentEntityKind};
use runmat_meshing_core::{MeshingCancellationSignal, NeverCancelled, StableDigest};
use runmat_meshing_size::metric::{MetricCombinationRule, MetricFieldRequest, MetricTensor3};

use super::*;
use crate::cdt::{
    assign_delaunay_volume_regions, build_delaunay_volume_topology,
    evaluate_delaunay_volume_quality, insert_delaunay_volume_node, DelaunayFacetProvenance,
    DelaunayInsertionErrorKind, DelaunayInsertionOptions, DelaunaySegmentProvenance,
    DelaunayTopologyOptions,
};

fn region() -> PersistentEntityId {
    PersistentEntityId {
        kind: PersistentEntityKind::Region,
        source_topology_id: "solid".to_owned(),
        assembly_path: Vec::new(),
    }
}

fn named_region(value: &str) -> PersistentEntityId {
    PersistentEntityId {
        kind: PersistentEntityKind::Region,
        source_topology_id: value.to_owned(),
        assembly_path: Vec::new(),
    }
}

fn face(value: &str) -> PersistentEntityId {
    PersistentEntityId {
        kind: PersistentEntityKind::Face,
        source_topology_id: value.to_owned(),
        assembly_path: Vec::new(),
    }
}

fn edge(value: &str) -> PersistentEntityId {
    PersistentEntityId {
        kind: PersistentEntityKind::Edge,
        source_topology_id: value.to_owned(),
        assembly_path: Vec::new(),
    }
}

fn topology(points: [[f64; 3]; 4]) -> DelaunayVolumeTopology {
    let nodes = points
        .into_iter()
        .enumerate()
        .map(|(index, coordinates_m)| DelaunayVolumeNode {
            identity: StableDigest::from_bytes([(index + 1) as u8; 32]),
            coordinates_m,
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
        vec![region()],
        DelaunayTopologyOptions::default(),
        &NeverCancelled,
    )
    .unwrap()
}

fn context(_topology: &DelaunayVolumeTopology) -> DelaunayVolumeProvenance {
    DelaunayVolumeProvenance {
        nodes: Vec::new(),
        segments: Vec::new(),
        facets: Vec::new(),
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

fn anisotropic_metric() -> MetricFieldRequest {
    MetricFieldRequest {
        combination: MetricCombinationRule::MostRestrictiveIntersection,
        global_metric: MetricTensor3 {
            xx: 1.0,
            yy: 1.25,
            zz: 1.5,
            xy: 0.1,
            xz: 0.05,
            yz: 0.02,
        },
        maximum_grading_ratio: 1.5,
        contributions: Vec::new(),
    }
}

fn quality(
    topology: &DelaunayVolumeTopology,
    provenance: &DelaunayVolumeProvenance,
    maximum_edge: f64,
) -> (DelaunayVolumeQuality, DelaunayVolumeQualityOptions) {
    let options = DelaunayVolumeQualityOptions {
        maximum_metric_edge_length: maximum_edge,
        maximum_radius_edge_ratio: 10.0,
        ..DelaunayVolumeQualityOptions::default()
    };
    let quality =
        evaluate_delaunay_volume_quality(topology, &metric(), provenance, options, &NeverCancelled)
            .unwrap();
    (quality, options)
}

fn refinement_input<'a>(
    topology: &'a DelaunayVolumeTopology,
    metric_request: &'a MetricFieldRequest,
    provenance: &'a DelaunayVolumeProvenance,
    quality: &'a DelaunayVolumeQuality,
    quality_options: DelaunayVolumeQualityOptions,
) -> DelaunayVolumeRefinementInput<'a> {
    DelaunayVolumeRefinementInput {
        topology,
        metric_request,
        provenance,
        quality,
        quality_options,
    }
}

fn two_region_topology() -> DelaunayVolumeTopology {
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
        vec![named_region("left"), named_region("right")],
        DelaunayTopologyOptions::default(),
        &NeverCancelled,
    )
    .unwrap()
}

#[test]
fn refinement_selects_a_deterministic_interior_metric_circumcenter() {
    let height = (2.0_f64 / 3.0).sqrt();
    let topology = topology([
        [0.0, 0.0, 0.0],
        [1.0, 0.0, 0.0],
        [0.5, 3.0_f64.sqrt() * 0.5, 0.0],
        [0.5, 3.0_f64.sqrt() / 6.0, height],
    ]);
    let contexts = context(&topology);
    let (quality, quality_options) = quality(&topology, &contexts, 0.5);
    let metric_request = metric();
    let first = select_delaunay_volume_refinement_candidate(
        refinement_input(
            &topology,
            &metric_request,
            &contexts,
            &quality,
            quality_options,
        ),
        DelaunayVolumeRefinementCandidateOptions::default(),
        &NeverCancelled,
    )
    .unwrap()
    .unwrap();
    let second = select_delaunay_volume_refinement_candidate(
        refinement_input(
            &topology,
            &metric_request,
            &contexts,
            &quality,
            quality_options,
        ),
        DelaunayVolumeRefinementCandidateOptions::default(),
        &NeverCancelled,
    )
    .unwrap()
    .unwrap();
    assert_eq!(first, second);
    assert_eq!(
        first.kind,
        DelaunayRefinementCandidateKind::MetricCircumcenter
    );
    assert!(first.source_violation_ratio > 1.0);

    let anisotropic_request = anisotropic_metric();
    let anisotropic_options = DelaunayVolumeQualityOptions {
        maximum_metric_edge_length: 0.5,
        maximum_radius_edge_ratio: 10.0,
        ..DelaunayVolumeQualityOptions::default()
    };
    let anisotropic_quality = evaluate_delaunay_volume_quality(
        &topology,
        &anisotropic_request,
        &contexts,
        anisotropic_options,
        &NeverCancelled,
    )
    .unwrap();
    let anisotropic = select_delaunay_volume_refinement_candidate(
        refinement_input(
            &topology,
            &anisotropic_request,
            &contexts,
            &anisotropic_quality,
            anisotropic_options,
        ),
        DelaunayVolumeRefinementCandidateOptions::default(),
        &NeverCancelled,
    )
    .unwrap()
    .unwrap();
    assert_eq!(
        anisotropic.kind,
        DelaunayRefinementCandidateKind::MetricCircumcenter
    );
    assert_ne!(anisotropic.node.identity, first.node.identity);
}

#[test]
fn refinement_uses_an_interior_centroid_when_circumcenter_is_exterior() {
    let topology = topology([
        [0.0, 0.0, 0.0],
        [2.0, 0.0, 0.0],
        [0.0, 1.0, 0.0],
        [0.0, 0.0, 1.0],
    ]);
    let contexts = context(&topology);
    let (quality, quality_options) = quality(&topology, &contexts, 0.5);
    let metric_request = metric();
    let selected = select_delaunay_volume_refinement_candidate(
        refinement_input(
            &topology,
            &metric_request,
            &contexts,
            &quality,
            quality_options,
        ),
        DelaunayVolumeRefinementCandidateOptions::default(),
        &NeverCancelled,
    )
    .unwrap()
    .unwrap();
    assert_eq!(
        selected.kind,
        DelaunayRefinementCandidateKind::InteriorCentroid
    );
    assert_eq!(selected.node.coordinates_m, [0.5, 0.25, 0.25]);
}

struct Cancelled;

impl MeshingCancellationSignal for Cancelled {
    fn is_cancelled(&self) -> bool {
        true
    }
}

#[test]
fn refinement_handles_convergence_tampering_limits_and_cancellation() {
    let topology = topology([
        [0.0, 0.0, 0.0],
        [1.0, 0.0, 0.0],
        [0.0, 1.0, 0.0],
        [0.0, 0.0, 1.0],
    ]);
    let contexts = context(&topology);
    let (converged, converged_options) = quality(&topology, &contexts, 10.0);
    let metric_request = metric();
    assert!(select_delaunay_volume_refinement_candidate(
        refinement_input(
            &topology,
            &metric_request,
            &contexts,
            &converged,
            converged_options,
        ),
        DelaunayVolumeRefinementCandidateOptions::default(),
        &NeverCancelled,
    )
    .unwrap()
    .is_none());

    let (quality, quality_options) = quality(&topology, &contexts, 0.5);
    let mut candidate = select_delaunay_volume_refinement_candidate(
        refinement_input(
            &topology,
            &metric_request,
            &contexts,
            &quality,
            quality_options,
        ),
        DelaunayVolumeRefinementCandidateOptions::default(),
        &NeverCancelled,
    )
    .unwrap();
    candidate.as_mut().unwrap().node.coordinates_m[0] += 0.1;
    assert_eq!(
        validate_delaunay_volume_refinement_candidate(
            refinement_input(
                &topology,
                &metric_request,
                &contexts,
                &quality,
                quality_options,
            ),
            &candidate,
            DelaunayVolumeRefinementCandidateOptions::default(),
            &NeverCancelled,
        )
        .unwrap_err()
        .kind,
        DelaunayVolumeRefinementCandidateErrorKind::InvalidCandidate
    );
    assert_eq!(
        select_delaunay_volume_refinement_candidate(
            refinement_input(
                &topology,
                &metric_request,
                &contexts,
                &quality,
                quality_options,
            ),
            DelaunayVolumeRefinementCandidateOptions {
                maximum_candidate_evaluations: 0,
                cancellation_check_interval: 1,
            },
            &NeverCancelled,
        )
        .unwrap_err()
        .kind,
        DelaunayVolumeRefinementCandidateErrorKind::InvalidOptions
    );
    assert_eq!(
        select_delaunay_volume_refinement_candidate(
            refinement_input(
                &topology,
                &metric_request,
                &contexts,
                &quality,
                quality_options,
            ),
            DelaunayVolumeRefinementCandidateOptions {
                maximum_candidate_evaluations: 1,
                cancellation_check_interval: 1,
            },
            &NeverCancelled,
        )
        .unwrap_err()
        .kind,
        DelaunayVolumeRefinementCandidateErrorKind::ResourceLimit
    );
    assert_eq!(
        select_delaunay_volume_refinement_candidate(
            refinement_input(
                &topology,
                &metric_request,
                &contexts,
                &quality,
                quality_options,
            ),
            DelaunayVolumeRefinementCandidateOptions::default(),
            &Cancelled,
        )
        .unwrap_err()
        .kind,
        DelaunayVolumeRefinementCandidateErrorKind::Cancelled
    );
}

#[test]
fn refinement_insertion_preserves_region_interface_and_rederives_quality() {
    let two_region = two_region_topology();
    let provenance = DelaunayVolumeProvenance {
        nodes: Vec::new(),
        segments: vec![DelaunaySegmentProvenance {
            node_identities: [
                StableDigest::from_bytes([2; 32]),
                StableDigest::from_bytes([3; 32]),
            ],
            entity_ids: vec![edge("interface-edge")],
            edge_parameters: [0.0, 1.0],
        }],
        facets: vec![
            DelaunayFacetProvenance {
                node_identities: [
                    StableDigest::from_bytes([1; 32]),
                    StableDigest::from_bytes([3; 32]),
                    StableDigest::from_bytes([4; 32]),
                ],
                chart_id: StableDigest::from_bytes([40; 32]),
                entity_ids: vec![face("left-boundary")],
                region_ids: vec![named_region("left")],
            },
            DelaunayFacetProvenance {
                node_identities: [
                    StableDigest::from_bytes([2; 32]),
                    StableDigest::from_bytes([3; 32]),
                    StableDigest::from_bytes([4; 32]),
                ],
                chart_id: StableDigest::from_bytes([41; 32]),
                entity_ids: vec![face("interface")],
                region_ids: vec![named_region("left"), named_region("right")],
            },
        ],
    };
    let metric_request = metric();
    let quality_options = DelaunayVolumeQualityOptions {
        maximum_metric_edge_length: 0.5,
        maximum_radius_edge_ratio: 10.0,
        ..DelaunayVolumeQualityOptions::default()
    };
    let initial_quality = evaluate_delaunay_volume_quality(
        &two_region,
        &metric_request,
        &provenance,
        quality_options,
        &NeverCancelled,
    )
    .unwrap();
    let input = refinement_input(
        &two_region,
        &metric_request,
        &provenance,
        &initial_quality,
        quality_options,
    );
    let candidate = select_delaunay_volume_refinement_candidate(
        input,
        DelaunayVolumeRefinementCandidateOptions::default(),
        &NeverCancelled,
    )
    .unwrap()
    .unwrap();
    assert_eq!(
        insert_delaunay_volume_node(
            two_region.clone(),
            candidate.node,
            DelaunayInsertionOptions::default(),
            &NeverCancelled,
        )
        .unwrap_err()
        .kind,
        DelaunayInsertionErrorKind::InvalidTopology
    );
    let first = insert_delaunay_volume_refinement_candidate(
        input,
        &candidate,
        DelaunayVolumeRefinementStepOptions::default(),
        &NeverCancelled,
    )
    .unwrap();
    let second = insert_delaunay_volume_refinement_candidate(
        input,
        &candidate,
        DelaunayVolumeRefinementStepOptions::default(),
        &NeverCancelled,
    )
    .unwrap();

    assert_eq!(first, second);
    assert_eq!(first.topology.nodes.len(), 6);
    assert_eq!(first.topology.tetrahedra.len(), 5);
    assert_eq!(first.quality.tetrahedra.len(), 5);
    assert_eq!(first.topology.incidence.regions.len(), 2);
    validate_delaunay_volume_refinement_step(
        input,
        &candidate,
        &first,
        DelaunayVolumeRefinementStepOptions::default(),
        &NeverCancelled,
    )
    .unwrap();
    assert_eq!(
        insert_delaunay_volume_refinement_candidate(
            input,
            &candidate,
            DelaunayVolumeRefinementStepOptions {
                insertion: DelaunayInsertionOptions {
                    maximum_protected_faces: 1,
                    ..DelaunayInsertionOptions::default()
                },
                ..DelaunayVolumeRefinementStepOptions::default()
            },
            &NeverCancelled,
        )
        .unwrap_err()
        .kind,
        DelaunayVolumeRefinementStepErrorKind::ResourceLimit
    );
}

#[test]
fn refinement_insertion_rejects_unprotected_region_boundary_and_tampering() {
    let two_region = two_region_topology();
    let provenance = DelaunayVolumeProvenance {
        nodes: Vec::new(),
        segments: Vec::new(),
        facets: Vec::new(),
    };
    let metric_request = metric();
    let quality_options = DelaunayVolumeQualityOptions {
        maximum_metric_edge_length: 0.5,
        maximum_radius_edge_ratio: 10.0,
        ..DelaunayVolumeQualityOptions::default()
    };
    let initial_quality = evaluate_delaunay_volume_quality(
        &two_region,
        &metric_request,
        &provenance,
        quality_options,
        &NeverCancelled,
    )
    .unwrap();
    let input = refinement_input(
        &two_region,
        &metric_request,
        &provenance,
        &initial_quality,
        quality_options,
    );
    let candidate = select_delaunay_volume_refinement_candidate(
        input,
        DelaunayVolumeRefinementCandidateOptions::default(),
        &NeverCancelled,
    )
    .unwrap()
    .unwrap();
    assert_eq!(
        insert_delaunay_volume_refinement_candidate(
            input,
            &candidate,
            DelaunayVolumeRefinementStepOptions::default(),
            &NeverCancelled,
        )
        .unwrap_err()
        .kind,
        DelaunayVolumeRefinementStepErrorKind::InvalidTopology
    );

    let single = topology([
        [0.0, 0.0, 0.0],
        [1.0, 0.0, 0.0],
        [0.0, 1.0, 0.0],
        [0.0, 0.0, 1.0],
    ]);
    let single_provenance = context(&single);
    let (single_quality, single_options) = quality(&single, &single_provenance, 0.5);
    let single_input = refinement_input(
        &single,
        &metric_request,
        &single_provenance,
        &single_quality,
        single_options,
    );
    let single_candidate = select_delaunay_volume_refinement_candidate(
        single_input,
        DelaunayVolumeRefinementCandidateOptions::default(),
        &NeverCancelled,
    )
    .unwrap()
    .unwrap();
    let mut step = insert_delaunay_volume_refinement_candidate(
        single_input,
        &single_candidate,
        DelaunayVolumeRefinementStepOptions::default(),
        &NeverCancelled,
    )
    .unwrap();
    step.quality.tetrahedra[0].maximum_metric_edge_length *= 2.0;
    assert_eq!(
        validate_delaunay_volume_refinement_step(
            single_input,
            &single_candidate,
            &step,
            DelaunayVolumeRefinementStepOptions::default(),
            &NeverCancelled,
        )
        .unwrap_err()
        .kind,
        DelaunayVolumeRefinementStepErrorKind::InvalidQuality
    );
}

#[test]
fn refinement_loop_preserves_converged_input_and_rejects_false_lineage() {
    let topology = topology([
        [0.0, 0.0, 0.0],
        [1.0, 0.0, 0.0],
        [0.0, 1.0, 0.0],
        [0.0, 0.0, 1.0],
    ]);
    let provenance = context(&topology);
    let metric_request = metric();
    let (quality, quality_options) = quality(&topology, &provenance, 10.0);
    let input = refinement_input(
        &topology,
        &metric_request,
        &provenance,
        &quality,
        quality_options,
    );
    let first = refine_delaunay_volume(
        input,
        DelaunayVolumeRefinementOptions::default(),
        &NeverCancelled,
    )
    .unwrap();
    let second = refine_delaunay_volume(
        input,
        DelaunayVolumeRefinementOptions::default(),
        &NeverCancelled,
    )
    .unwrap();
    assert_eq!(first, second);
    assert_eq!(first.topology, topology);
    assert_eq!(first.quality, quality);
    assert!(first.mutations.is_empty());

    let mut tampered = first;
    tampered
        .mutations
        .push(DelaunayVolumeRefinementMutation::Inserted(
            DelaunayVolumeNode {
                identity: StableDigest::from_bytes([9; 32]),
                coordinates_m: [0.25; 3],
            },
        ));
    assert_eq!(
        validate_delaunay_volume_refinement(
            input,
            &tampered,
            DelaunayVolumeRefinementOptions::default(),
            &NeverCancelled,
        )
        .unwrap_err()
        .kind,
        DelaunayVolumeRefinementStepErrorKind::InvalidInput
    );
}

#[test]
fn refinement_loop_fails_closed_on_nonconvergence_budget_and_cancellation() {
    let topology = topology([
        [0.0, 0.0, 0.0],
        [1.0, 0.0, 0.0],
        [0.0, 1.0, 0.0],
        [0.0, 0.0, 1.0],
    ]);
    let provenance = context(&topology);
    let metric_request = metric();
    let (quality, quality_options) = quality(&topology, &provenance, 0.5);
    let input = refinement_input(
        &topology,
        &metric_request,
        &provenance,
        &quality,
        quality_options,
    );
    let options = DelaunayVolumeRefinementOptions {
        maximum_insertions: 1,
        ..DelaunayVolumeRefinementOptions::default()
    };
    let first = refine_delaunay_volume(input, options, &NeverCancelled).unwrap_err();
    let second = refine_delaunay_volume(input, options, &NeverCancelled).unwrap_err();
    assert_eq!(first, second);
    assert_eq!(
        first.kind,
        DelaunayVolumeRefinementStepErrorKind::ResourceLimit
    );
    assert!(first.reason.contains("exhausted 1 insertions"));
    assert!(first.reason.contains("worst violation ratio"));
    assert_eq!(
        refine_delaunay_volume(
            input,
            DelaunayVolumeRefinementOptions {
                maximum_insertions: 0,
                ..DelaunayVolumeRefinementOptions::default()
            },
            &NeverCancelled,
        )
        .unwrap_err()
        .kind,
        DelaunayVolumeRefinementStepErrorKind::InvalidOptions
    );
    assert_eq!(
        refine_delaunay_volume(input, options, &Cancelled)
            .unwrap_err()
            .kind,
        DelaunayVolumeRefinementStepErrorKind::Cancelled
    );
}
