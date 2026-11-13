use std::collections::BTreeSet;

use runmat_geometry_core::{PersistentEntityId, PersistentEntityKind};
use runmat_meshing_core::{MeshingCancellationSignal, NeverCancelled, StableDigest};
use runmat_meshing_size::metric::{MetricCombinationRule, MetricFieldRequest, MetricTensor3};

use super::*;
use crate::cdt::{
    assign_delaunay_volume_regions, build_delaunay_volume_topology, coarsen_marked_delaunay_volume,
    evaluate_delaunay_volume_quality, insert_delaunay_volume_node, refine_marked_delaunay_volume,
    validate_marked_delaunay_volume_coarsening, validate_marked_delaunay_volume_refinement,
    DelaunayAdaptiveCoarseningErrorKind, DelaunayAdaptiveCoarseningOptions,
    DelaunayAdaptiveRefinementDecision, DelaunayAdaptiveRefinementErrorKind,
    DelaunayAdaptiveRefinementMark, DelaunayAdaptiveRefinementOptions, DelaunayFacetProvenance,
    DelaunayInsertionErrorKind, DelaunayInsertionOptions, DelaunaySegmentProvenance,
    DelaunayTopologyOptions,
};

pub(super) fn region() -> PersistentEntityId {
    PersistentEntityId::new(PersistentEntityKind::Region, "solid".to_owned(), Vec::new())
}

fn named_region(value: &str) -> PersistentEntityId {
    PersistentEntityId::new(PersistentEntityKind::Region, value.to_owned(), Vec::new())
}

fn face(value: &str) -> PersistentEntityId {
    PersistentEntityId::new(PersistentEntityKind::Face, value.to_owned(), Vec::new())
}

fn edge(value: &str) -> PersistentEntityId {
    PersistentEntityId::new(PersistentEntityKind::Edge, value.to_owned(), Vec::new())
}

fn topology(points: [[f64; 3]; 4]) -> DelaunayVolumeTopology {
    let nodes = points
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

pub(super) fn context(_topology: &DelaunayVolumeTopology) -> DelaunayVolumeProvenance {
    DelaunayVolumeProvenance {
        nodes: Vec::new(),
        segments: Vec::new(),
        facets: Vec::new(),
    }
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

fn anisotropic_metric() -> MetricFieldRequest {
    MetricFieldRequest {
        entity_supports: Vec::new(),
        located_sources: Vec::new(),
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
        cells_across_gap: None,
    }
}

pub(super) fn quality(
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

#[test]
fn batch_admission_requires_lexicographic_quality_progress() {
    let topology = topology([
        [0.0, 0.0, 0.0],
        [1.0, 0.0, 0.0],
        [0.0, 1.0, 0.0],
        [0.0, 0.0, 1.0],
    ]);
    let provenance = context(&topology);
    let (baseline, _) = quality(&topology, &provenance, 0.5);
    let template = baseline.tetrahedra[0].clone();
    let with_ratios = |ratios: &[f64]| {
        let mut result = baseline.clone();
        result.tetrahedra = ratios
            .iter()
            .map(|ratio| {
                let mut tetrahedron = template.clone();
                tetrahedron.refinement_violation_ratio = *ratio;
                tetrahedron
            })
            .collect();
        result
    };
    let prior = with_ratios(&[4.0, 2.0]);

    assert!(super::batch::quality_strictly_improves(
        &prior,
        &with_ratios(&[4.0, 1.5])
    ));
    assert!(super::batch::quality_strictly_improves(
        &prior,
        &with_ratios(&[3.9, 3.0])
    ));
    assert!(!super::batch::quality_strictly_improves(
        &prior,
        &with_ratios(&[4.0, 2.0])
    ));
    assert!(!super::batch::quality_strictly_improves(
        &prior,
        &with_ratios(&[4.0, 2.0, 1.1])
    ));
}

#[test]
fn batch_refinement_applies_separated_cavities_in_one_transaction() {
    let batch_scheduler::Fixture {
        topology,
        provenance,
        metric_request,
        quality,
        quality_options,
    } = batch_scheduler::fixture();
    let mut excluded = BTreeSet::new();

    let outcome = super::batch::insert_independent_refinement_batch(
        refinement_input(
            &topology,
            &metric_request,
            &provenance,
            &quality,
            quality_options,
        ),
        &CandidateQueries::new(&topology, &NeverCancelled).unwrap(),
        super::batch::RefinementBatchOptions {
            step: DelaunayVolumeRefinementStepOptions::default(),
            maximum_insertions: 8,
            frontier: &mut Default::default(),
        },
        &mut excluded,
        &NeverCancelled,
    )
    .unwrap();

    let super::batch::RefinementBatchOutcome::Admitted(batch) = outcome else {
        panic!("two separated violating cavities should form one admitted batch");
    };
    assert_eq!(batch.inserted_nodes.len(), 2);
    assert_eq!(batch.step.topology.nodes.len(), topology.nodes.len() + 2);
    assert!(super::batch::quality_strictly_improves(
        &quality,
        &batch.step.quality
    ));
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
        fixed_node_identities: &[],
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
        vec![named_region("left"), named_region("right")],
        DelaunayTopologyOptions::default(),
        &NeverCancelled,
    )
    .unwrap()
}

fn two_region_provenance() -> DelaunayVolumeProvenance {
    DelaunayVolumeProvenance {
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
    }
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
    assert_ne!(
        anisotropic.node.coordinates_m.canonical_value(),
        first.node.coordinates_m.canonical_value()
    );
    assert_eq!(anisotropic.node.identity, first.node.identity);
}

#[test]
fn refinement_selects_the_next_canonical_candidate_after_sliver_deletion() {
    let topology = topology([
        [0.0, 0.0, 0.0],
        [2.0, 0.0, 0.0],
        [0.0, 1.0, 0.0],
        [0.0, 0.0, 1.0],
    ]);
    let provenance = context(&topology);
    let request = metric();
    let (quality, options) = quality(&topology, &provenance, 0.5);
    let input = refinement_input(&topology, &request, &provenance, &quality, options);
    let first = select_delaunay_volume_refinement_candidate(
        input,
        DelaunayVolumeRefinementCandidateOptions::default(),
        &NeverCancelled,
    )
    .unwrap()
    .unwrap();
    let second = select_delaunay_volume_refinement_candidate_excluding(
        input,
        DelaunayVolumeRefinementCandidateOptions::default(),
        &BTreeSet::from([first.node.identity]),
        &NeverCancelled,
    )
    .unwrap()
    .unwrap();

    assert_ne!(second.node.identity, first.node.identity);
    assert_ne!(
        second.node.coordinates_m.canonical_value(),
        first.node.coordinates_m.canonical_value()
    );
    validate_delaunay_volume_refinement_candidate(
        input,
        &Some(second),
        DelaunayVolumeRefinementCandidateOptions::default(),
        &NeverCancelled,
    )
    .unwrap();
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
        DelaunayRefinementCandidateKind::InteriorPoint
    );
    assert_eq!(
        selected.node.coordinates_m.canonical_value(),
        [0.5, 0.25, 0.25]
    );
}

// witness: FEA-MESH-DISCRETIZATION-001/volume_quality_convergence @
//   shared_infrastructure/canonical_mixed_topology/numerical_kernel; coverage: partial;
//   boundary: The size-defect edge candidate splits only an edge with an interior endpoint and
//   lies strictly inside its source; whole-part runtime is measured on the cantilever study.
#[test]
fn size_refinement_splits_the_longest_edge_with_an_interior_endpoint() {
    let corner = [2.0, 0.0, 0.0];
    let interior = [0.3, 0.2, 0.2];
    let topology = insert_delaunay_volume_node(
        topology([[0.0, 0.0, 0.0], corner, [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]]),
        DelaunayVolumeNode::explicit(StableDigest::from_bytes([9; 32]), interior),
        DelaunayInsertionOptions::default(),
        &NeverCancelled,
    )
    .unwrap();
    let contexts = context(&topology);
    let (evaluated, options) = quality(&topology, &contexts, 1.2);
    let points = |index: usize| {
        topology.tetrahedra[index].vertex_indices.map(|vertex| {
            topology.nodes[vertex as usize]
                .coordinates_m
                .canonical_value()
        })
    };
    // The hull edges (corner to corner, up to 2.24) are longer than the interior
    // edge to the far corner (1.73), but they cannot be split from inside.
    let source = (0..topology.tetrahedra.len())
        .find(|&index| points(index).contains(&corner) && points(index).contains(&[0.0, 1.0, 0.0]))
        .unwrap();
    let cell = evaluated
        .tetrahedra
        .iter()
        .find(|cell| {
            let mut identities = cell.node_identities;
            identities.sort_unstable();
            let mut expected = topology.tetrahedra[source]
                .vertex_indices
                .map(|vertex| topology.nodes[vertex as usize].identity);
            expected.sort_unstable();
            identities == expected
        })
        .unwrap();
    let point = candidate::metric_edge_midpoint(&topology, source, cell, options).unwrap();
    let midpoint: [f64; 3] = std::array::from_fn(|axis| 0.5 * (interior[axis] + corner[axis]));
    let offset = (0..3)
        .map(|axis| (point[axis] - midpoint[axis]).powi(2))
        .sum::<f64>()
        .sqrt();
    assert!(
        offset < 0.1,
        "{point:?} is not near the interior edge midpoint {midpoint:?}"
    );
    assert!(candidate::strictly_inside(points(source), point).unwrap());
    // No edge over the bound with an interior endpoint: no candidate.
    let (relaxed, relaxed_options) = quality(&topology, &contexts, 2.0);
    let relaxed_cell = relaxed
        .tetrahedra
        .iter()
        .find(|candidate| candidate.node_identities == cell.node_identities)
        .unwrap();
    assert!(
        candidate::metric_edge_midpoint(&topology, source, relaxed_cell, relaxed_options).is_none()
    );
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
    let admitted_candidate = candidate.clone();
    {
        let quantity_change = 0.1;
        let quantity_index = 0;
        let quantity_slot = &mut candidate.as_mut().unwrap().node.coordinates_m;
        let mut quantity_coordinates = quantity_slot.canonical_value();
        quantity_coordinates[quantity_index] += quantity_change;
        *quantity_slot = runmat_types::physical::Quantity::<
            runmat_types::physical::kind::Position,
            _,
        >::new(quantity_coordinates);
    };
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
    let mut false_construction = admitted_candidate;
    false_construction
        .as_mut()
        .unwrap()
        .construction_node_identities[0] = StableDigest::ZERO;
    assert_eq!(
        validate_delaunay_volume_refinement_candidate(
            refinement_input(
                &topology,
                &metric_request,
                &contexts,
                &quality,
                quality_options,
            ),
            &false_construction,
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
    let provenance = two_region_provenance();
    let metric_request = metric();
    let quality_options = DelaunayVolumeQualityOptions {
        maximum_metric_edge_length: 0.5,
        maximum_radius_edge_ratio: 10.0,
        minimum_metric_scaled_jacobian: 1.0e-12,
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
    assert!(first.insertions.is_empty());

    let mut tampered = first;
    tampered.insertions.push(DelaunayVolumeNode::explicit(
        StableDigest::from_bytes([9; 32]),
        [0.25; 3],
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

#[test]
fn marked_adaptation_is_local_deterministic_replayable_and_bounded() {
    let topology = two_region_topology();
    let provenance = two_region_provenance();
    let metric_request = metric();
    let quality_options = DelaunayVolumeQualityOptions {
        maximum_metric_edge_length: 10.0,
        maximum_radius_edge_ratio: 10.0,
        ..DelaunayVolumeQualityOptions::default()
    };
    let quality = evaluate_delaunay_volume_quality(
        &topology,
        &metric_request,
        &provenance,
        quality_options,
        &NeverCancelled,
    )
    .unwrap();
    let input = refinement_input(
        &topology,
        &metric_request,
        &provenance,
        &quality,
        quality_options,
    );
    let marks = quality
        .tetrahedra
        .iter()
        .map(|tetrahedron| DelaunayAdaptiveRefinementMark {
            node_identities: tetrahedron.node_identities,
            indicator_value: 1.0,
        })
        .collect::<Vec<_>>();
    let first = refine_marked_delaunay_volume(
        input,
        &marks,
        DelaunayAdaptiveRefinementOptions::default(),
        &NeverCancelled,
    )
    .unwrap();
    let second = refine_marked_delaunay_volume(
        input,
        &marks.iter().copied().rev().collect::<Vec<_>>(),
        DelaunayAdaptiveRefinementOptions::default(),
        &NeverCancelled,
    )
    .unwrap();
    assert_eq!(first, second);
    assert_eq!(first.topology.nodes.len(), topology.nodes.len() + 2);
    assert!(first.decisions.iter().all(|decision| matches!(
        decision,
        DelaunayAdaptiveRefinementDecision::Inserted { .. }
    )));
    validate_marked_delaunay_volume_refinement(
        input,
        &marks,
        &first,
        DelaunayAdaptiveRefinementOptions::default(),
        &NeverCancelled,
    )
    .unwrap();
    let mut false_lineage = first.clone();
    let DelaunayAdaptiveRefinementDecision::Inserted { lineage, .. } =
        &mut false_lineage.decisions[0]
    else {
        panic!("fixture must produce an insertion");
    };
    lineage.created_tetrahedra.pop();
    assert_eq!(
        validate_marked_delaunay_volume_refinement(
            input,
            &marks,
            &false_lineage,
            DelaunayAdaptiveRefinementOptions::default(),
            &NeverCancelled,
        )
        .unwrap_err()
        .kind,
        DelaunayAdaptiveRefinementErrorKind::InvalidResult
    );

    let inserted_nodes = first
        .decisions
        .iter()
        .filter_map(|decision| match decision {
            DelaunayAdaptiveRefinementDecision::Inserted { lineage, .. } => {
                Some(lineage.node.identity)
            }
            DelaunayAdaptiveRefinementDecision::CoveredByPriorInsertion { .. } => None,
        })
        .collect::<Vec<_>>();
    let coarsened = coarsen_marked_delaunay_volume(
        input,
        &first,
        &inserted_nodes,
        DelaunayAdaptiveCoarseningOptions::default(),
        &NeverCancelled,
    )
    .unwrap();
    assert_eq!(coarsened.topology, topology);
    assert_eq!(coarsened.quality, quality);
    assert_eq!(
        coarsened.removed_node_identities,
        inserted_nodes.iter().copied().rev().collect::<Vec<_>>()
    );
    let repeated = coarsen_marked_delaunay_volume(
        input,
        &first,
        &inserted_nodes.iter().copied().rev().collect::<Vec<_>>(),
        DelaunayAdaptiveCoarseningOptions::default(),
        &NeverCancelled,
    )
    .unwrap();
    assert_eq!(coarsened, repeated);
    validate_marked_delaunay_volume_coarsening(
        input,
        &first,
        &inserted_nodes,
        &coarsened,
        DelaunayAdaptiveCoarseningOptions::default(),
        &NeverCancelled,
    )
    .unwrap();
    let partial = coarsen_marked_delaunay_volume(
        input,
        &first,
        &inserted_nodes[..1],
        DelaunayAdaptiveCoarseningOptions::default(),
        &NeverCancelled,
    )
    .unwrap();
    assert_eq!(partial.topology.nodes.len(), topology.nodes.len() + 1);

    let mut tampered_coarsening = coarsened.clone();
    tampered_coarsening.quality.tetrahedra[0].maximum_metric_edge_length *= 2.0;
    assert_eq!(
        validate_marked_delaunay_volume_coarsening(
            input,
            &first,
            &inserted_nodes,
            &tampered_coarsening,
            DelaunayAdaptiveCoarseningOptions::default(),
            &NeverCancelled,
        )
        .unwrap_err()
        .kind,
        DelaunayAdaptiveCoarseningErrorKind::InvalidResult
    );
    assert_eq!(
        coarsen_marked_delaunay_volume(
            input,
            &first,
            &inserted_nodes,
            DelaunayAdaptiveCoarseningOptions {
                maximum_removals: 1,
                ..DelaunayAdaptiveCoarseningOptions::default()
            },
            &NeverCancelled,
        )
        .unwrap_err()
        .kind,
        DelaunayAdaptiveCoarseningErrorKind::ResourceLimit
    );
    assert_eq!(
        coarsen_marked_delaunay_volume(
            input,
            &first,
            &[inserted_nodes[0], inserted_nodes[0]],
            DelaunayAdaptiveCoarseningOptions::default(),
            &NeverCancelled,
        )
        .unwrap_err()
        .kind,
        DelaunayAdaptiveCoarseningErrorKind::InvalidRemovals
    );
    assert_eq!(
        coarsen_marked_delaunay_volume(
            input,
            &first,
            &[StableDigest::from_bytes([99; 32])],
            DelaunayAdaptiveCoarseningOptions::default(),
            &NeverCancelled,
        )
        .unwrap_err()
        .kind,
        DelaunayAdaptiveCoarseningErrorKind::InvalidRemovals
    );
    assert_eq!(
        coarsen_marked_delaunay_volume(
            input,
            &first,
            &inserted_nodes,
            DelaunayAdaptiveCoarseningOptions {
                cancellation_check_interval: 0,
                ..DelaunayAdaptiveCoarseningOptions::default()
            },
            &NeverCancelled,
        )
        .unwrap_err()
        .kind,
        DelaunayAdaptiveCoarseningErrorKind::InvalidOptions
    );
    assert_eq!(
        coarsen_marked_delaunay_volume(
            input,
            &first,
            &inserted_nodes,
            DelaunayAdaptiveCoarseningOptions::default(),
            &Cancelled,
        )
        .unwrap_err()
        .kind,
        DelaunayAdaptiveCoarseningErrorKind::Cancelled
    );

    let mut tampered = first.clone();
    tampered.decisions.swap(0, 1);
    assert_eq!(
        validate_marked_delaunay_volume_refinement(
            input,
            &marks,
            &tampered,
            DelaunayAdaptiveRefinementOptions::default(),
            &NeverCancelled,
        )
        .unwrap_err()
        .kind,
        DelaunayAdaptiveRefinementErrorKind::InvalidResult
    );
    let mut tampered = first.clone();
    tampered.quality.tetrahedra[0].maximum_metric_edge_length *= 2.0;
    assert_eq!(
        validate_marked_delaunay_volume_refinement(
            input,
            &marks,
            &tampered,
            DelaunayAdaptiveRefinementOptions::default(),
            &NeverCancelled,
        )
        .unwrap_err()
        .kind,
        DelaunayAdaptiveRefinementErrorKind::InvalidResult
    );
    assert_eq!(
        refine_marked_delaunay_volume(
            input,
            &marks,
            DelaunayAdaptiveRefinementOptions {
                maximum_insertions: 1,
                ..DelaunayAdaptiveRefinementOptions::default()
            },
            &NeverCancelled,
        )
        .unwrap_err()
        .kind,
        DelaunayAdaptiveRefinementErrorKind::ResourceLimit
    );
    assert_eq!(
        refine_marked_delaunay_volume(
            input,
            &[marks[0], marks[0]],
            DelaunayAdaptiveRefinementOptions::default(),
            &NeverCancelled,
        )
        .unwrap_err()
        .kind,
        DelaunayAdaptiveRefinementErrorKind::InvalidMarks
    );
    for invalid_mark in [
        DelaunayAdaptiveRefinementMark {
            indicator_value: f64::NAN,
            ..marks[0]
        },
        DelaunayAdaptiveRefinementMark {
            indicator_value: 0.0,
            ..marks[0]
        },
        DelaunayAdaptiveRefinementMark {
            node_identities: [StableDigest::from_bytes([99; 32]); 4],
            ..marks[0]
        },
    ] {
        assert_eq!(
            refine_marked_delaunay_volume(
                input,
                &[invalid_mark],
                DelaunayAdaptiveRefinementOptions::default(),
                &NeverCancelled,
            )
            .unwrap_err()
            .kind,
            DelaunayAdaptiveRefinementErrorKind::InvalidMarks
        );
    }
    assert_eq!(
        refine_marked_delaunay_volume(
            input,
            &marks,
            DelaunayAdaptiveRefinementOptions {
                maximum_marks: 1,
                ..DelaunayAdaptiveRefinementOptions::default()
            },
            &NeverCancelled,
        )
        .unwrap_err()
        .kind,
        DelaunayAdaptiveRefinementErrorKind::ResourceLimit
    );
    assert_eq!(
        refine_marked_delaunay_volume(
            input,
            &marks,
            DelaunayAdaptiveRefinementOptions {
                cancellation_check_interval: 0,
                ..DelaunayAdaptiveRefinementOptions::default()
            },
            &NeverCancelled,
        )
        .unwrap_err()
        .kind,
        DelaunayAdaptiveRefinementErrorKind::InvalidOptions
    );
    assert_eq!(
        refine_marked_delaunay_volume(
            input,
            &marks,
            DelaunayAdaptiveRefinementOptions::default(),
            &Cancelled,
        )
        .unwrap_err()
        .kind,
        DelaunayAdaptiveRefinementErrorKind::Cancelled
    );
}

#[path = "tests/owned_state.rs"]
mod owned_state;

#[path = "tests/batch_scheduler.rs"]
mod batch_scheduler;
