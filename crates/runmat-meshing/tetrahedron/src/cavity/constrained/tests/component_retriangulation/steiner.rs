use super::super::*;
use runmat_meshing_core::NeverCancelled;

#[test]
fn constructive_interior_candidates_refill_a_schonhardt_cavity() {
    let rotation = std::f64::consts::FRAC_PI_6;
    let nodes = (0..6)
        .map(|node_id| {
            let layer = usize::from(node_id >= 3);
            let index = node_id % 3;
            let angle = std::f64::consts::TAU * index as f64 / 3.0 + rotation * layer as f64;
            ConstrainedCavityNode {
                node_id: node_id as u32,
                coordinates_m: [angle.cos(), angle.sin(), layer as f64],
            }
        })
        .collect::<Vec<_>>();
    let node_map = nodes
        .iter()
        .map(|node| (node.node_id, node.coordinates_m))
        .collect::<BTreeMap<_, _>>();
    let mut cavity = ConstrainedCavity {
        removed_tetrahedron_ids: vec![0],
        boundary_faces: [
            [0, 2, 1],
            [3, 4, 5],
            [0, 1, 4],
            [0, 4, 3],
            [1, 2, 5],
            [1, 5, 4],
            [2, 0, 3],
            [2, 3, 5],
        ]
        .into_iter()
        .map(|node_ids| ConstrainedCavityBoundaryFace {
            node_ids,
            outside_tetrahedron_ids: Vec::new(),
            source_face_id: None,
            source_edge_ids: [None; 3],
            region_ids: Vec::new(),
        })
        .collect(),
        protected_node_ids: Vec::new(),
        target_volume_m3: 1.0,
    };
    cavity.target_volume_m3 = cavity
        .boundary_faces
        .iter()
        .map(|face| {
            let [first, second, third] = face.node_ids.map(|node| node_map[&node]);
            first
                .into_iter()
                .zip([
                    second[1] * third[2] - second[2] * third[1],
                    second[2] * third[0] - second[0] * third[2],
                    second[0] * third[1] - second[1] * third[0],
                ])
                .map(|(left, right)| left * right)
                .sum::<f64>()
                / 6.0
        })
        .sum::<f64>()
        .abs();
    let options = ConstrainedCavityRefillOptions {
        min_scaled_jacobian: 0.0,
        ..ConstrainedCavityRefillOptions::default()
    };
    assert!(retriangulate_constrained_cavity_from_nodes(
        &cavity,
        &nodes,
        options,
        ConstrainedCavityRefillBudget::default(),
        &NeverCancelled,
    )
    .unwrap()
    .is_none());

    let candidates = generate_constrained_cavity_interior_steiner_candidates(
        &cavity,
        &nodes,
        options,
        ConstrainedCavitySteinerCandidateBudget::default(),
        &NeverCancelled,
    )
    .unwrap();
    assert!(!candidates.is_empty());
    let mut reordered_nodes = nodes.clone();
    reordered_nodes.reverse();
    assert_eq!(
        candidates,
        generate_constrained_cavity_interior_steiner_candidates(
            &cavity,
            &reordered_nodes,
            options,
            ConstrainedCavitySteinerCandidateBudget::default(),
            &NeverCancelled,
        )
        .unwrap()
    );
    assert!(matches!(
        generate_constrained_cavity_interior_steiner_candidates(
            &cavity,
            &nodes,
            options,
            ConstrainedCavitySteinerCandidateBudget {
                maximum_evaluations: 1,
                ..ConstrainedCavitySteinerCandidateBudget::default()
            },
            &NeverCancelled,
        ),
        Err(ConstrainedCavityRefillError::ResourceLimit { .. })
    ));
    struct Cancelled;
    impl runmat_meshing_core::MeshingCancellationSignal for Cancelled {
        fn is_cancelled(&self) -> bool {
            true
        }
    }
    assert_eq!(
        generate_constrained_cavity_interior_steiner_candidates(
            &cavity,
            &nodes,
            options,
            ConstrainedCavitySteinerCandidateBudget {
                cancellation_check_interval: 1,
                ..ConstrainedCavitySteinerCandidateBudget::default()
            },
            &Cancelled,
        )
        .unwrap_err(),
        ConstrainedCavityRefillError::Cancelled
    );
    let refill = candidates
        .into_iter()
        .take(64)
        .find_map(|coordinates_m| {
            let mut candidate_nodes = nodes.clone();
            candidate_nodes.push(ConstrainedCavityNode {
                node_id: nodes.len() as u32,
                coordinates_m,
            });
            retriangulate_constrained_cavity_from_nodes(
                &cavity,
                &candidate_nodes,
                options,
                ConstrainedCavityRefillBudget::default(),
                &NeverCancelled,
            )
            .unwrap()
        })
        .expect("a constructive interior candidate should refill the Schonhardt cavity");
    assert!(!refill.inserted_nodes.is_empty());
}
