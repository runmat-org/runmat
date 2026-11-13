use super::*;

pub(super) struct Fixture {
    pub(super) topology: DelaunayVolumeTopology,
    pub(super) provenance: DelaunayVolumeProvenance,
    pub(super) metric_request: MetricFieldRequest,
    pub(super) quality: DelaunayVolumeQuality,
    pub(super) quality_options: DelaunayVolumeQualityOptions,
}
pub(super) fn fixture() -> Fixture {
    let coordinates = [
        [0.0, 0.0, 0.0],
        [1.0, 0.0, 0.0],
        [0.0, 1.0, 0.0],
        [0.0, 0.0, 1.0],
        [10.0, 0.0, 0.0],
        [11.0, 0.0, 0.0],
        [10.0, 1.0, 0.0],
        [10.0, 0.0, 1.0],
    ];
    let nodes = coordinates
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
        vec![[0, 1, 2, 3], [4, 5, 6, 7]],
        DelaunayTopologyOptions::default(),
        &NeverCancelled,
    )
    .unwrap();
    let topology = assign_delaunay_volume_regions(
        topology,
        vec![region(), region()],
        DelaunayTopologyOptions::default(),
        &NeverCancelled,
    )
    .unwrap();
    let provenance = context(&topology);
    let metric_request = metric();
    let (mut quality, quality_options) = quality(&topology, &provenance, 0.75);
    for tetrahedron in &mut quality.tetrahedra {
        tetrahedron.refinement_violation_ratio = 100.0;
    }
    quality.worst_refinement_tetrahedron = quality
        .tetrahedra
        .iter()
        .map(|tetrahedron| tetrahedron.node_identities)
        .min();
    Fixture {
        topology,
        provenance,
        metric_request,
        quality,
        quality_options,
    }
}

// witness: FEA-SHARED-RESOURCES-001/phase_lifetimes @
//   shared_infrastructure/canonical_mixed_topology/numerical_kernel;
//   boundary: A controlled selection fixture admits one valid ready cavity at its original
//   evaluation ceiling while leaving unresolved quality visible. A maximum batch size is not a
//   minimum publication size; full physical convergence and original-model performance remain
//   separate gates.
#[test]
fn valid_partial_batch_is_admitted_at_the_original_evaluation_ceiling() {
    let fixture = fixture();
    let mut options = DelaunayVolumeRefinementStepOptions::default();
    options.candidate.maximum_candidate_evaluations = 2;
    let outcome = super::super::batch::insert_independent_refinement_batch(
        refinement_input(
            &fixture.topology,
            &fixture.metric_request,
            &fixture.provenance,
            &fixture.quality,
            fixture.quality_options,
        ),
        &CandidateQueries::new(&fixture.topology, &NeverCancelled).unwrap(),
        super::super::batch::RefinementBatchOptions {
            step: options,
            maximum_insertions: 128,
            frontier: &mut Default::default(),
        },
        &mut BTreeSet::new(),
        &NeverCancelled,
    )
    .unwrap();
    let super::super::batch::RefinementBatchOutcome::Admitted(batch) = outcome else {
        panic!("one valid ready cavity must not be discarded while seeking 128 slots");
    };
    assert_eq!(batch.inserted_nodes.len(), 1);
    assert_eq!(
        batch.step.topology.nodes.len(),
        fixture.topology.nodes.len() + 1
    );
    assert!(super::super::batch::quality_strictly_improves(
        &fixture.quality,
        &batch.step.quality
    ));
    assert!(
        batch.step.quality.worst_refinement_tetrahedron.is_some(),
        "this is accepted progress, not convergence"
    );
}

// witness: FEA-SHARED-RESOURCES-001/phase_lifetimes @
//   shared_infrastructure/canonical_mixed_topology/numerical_kernel;
//   boundary: A cooperative handoff retains the accepted insertion and unresolved quality without
//   labeling the intermediate checkpoint converged. This verifies owner handoff, not a completed
//   original-model mesh.
#[test]
fn cooperative_checkpoint_retains_progress_and_unresolved_quality() {
    let fixture = fixture();
    let mut options = DelaunayVolumeRefinementOptions::default();
    options.step.candidate.maximum_candidate_evaluations = 2;
    let progress = super::super::iteration::refine_owned_with_schedule(
        fixture.topology,
        fixture.quality,
        super::super::context::RefinementContext {
            metric_request: &fixture.metric_request,
            provenance: &fixture.provenance,
            quality_options: fixture.quality_options,
            fixed_node_identities: &[],
        },
        options,
        &NeverCancelled,
        super::super::iteration::RefinementSchedule::Cooperative,
    )
    .unwrap();
    assert!(!progress.stalled);
    assert_eq!(progress.refinement.insertions.len(), 1);
    assert!(progress
        .refinement
        .quality
        .worst_refinement_tetrahedron
        .is_some());
}

// witness: FEA-SHARED-RESOURCES-001/phase_lifetimes @
//   shared_infrastructure/canonical_mixed_topology/numerical_kernel;
//   boundary: When shape blocks floor-preserving insertion, the checkpoint moves the unchanged
//   original topology to the shape owner and matches independently recomputed quality. Unchanged
//   allocation pointers are bookkeeping evidence; complete shape repair and resource qualification
//   remain separate.
#[test]
fn shape_blocked_checkpoint_preserves_original_state_for_the_shape_owner() {
    let topology = topology([[0., 0., 0.], [1., 0., 0.], [0., 1., 0.], [0.2, 0.2, 1e-9]]);
    let provenance = context(&topology);
    let metric_request = metric();
    let (quality, quality_options) = quality(&topology, &provenance, 0.5);
    assert!(quality.worst_refinement_tetrahedron.is_some());
    assert!(quality.worst_optimization_tetrahedron.is_some());
    let nodes = topology.nodes.as_ptr();
    let cells = topology.tetrahedra.as_ptr();
    let mut options = DelaunayVolumeRefinementOptions::default();
    options.step.candidate.maximum_candidate_evaluations = 2;
    let progress = super::super::iteration::refine_owned_with_schedule(
        topology,
        quality,
        super::super::context::RefinementContext {
            metric_request: &metric_request,
            provenance: &provenance,
            quality_options,
            fixed_node_identities: &[],
        },
        options,
        &NeverCancelled,
        super::super::iteration::RefinementSchedule::Cooperative,
    )
    .unwrap();
    assert!(!progress.stalled);
    assert!(progress.refinement.insertions.is_empty());
    assert_eq!(progress.refinement.topology.nodes.as_ptr(), nodes);
    assert_eq!(progress.refinement.topology.tetrahedra.as_ptr(), cells);
    assert_eq!(
        progress.refinement.quality,
        evaluate_delaunay_volume_quality(
            &progress.refinement.topology,
            &metric_request,
            &provenance,
            quality_options,
            &NeverCancelled
        )
        .unwrap()
    );
}
