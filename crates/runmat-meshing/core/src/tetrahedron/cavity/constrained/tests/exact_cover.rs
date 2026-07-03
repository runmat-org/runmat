use super::*;

mod cover_selection;
mod diagnostics;
mod forced_mate_traces;
mod search_limits;
mod search_mechanics;
mod steiner_diagnostics;

#[test]
fn boundary_node_exact_cover_supports_bounded_multi_ring_bipyramid() {
    let ring_count = 7_u32;
    let top_node_id = ring_count;
    let bottom_node_id = ring_count + 1;
    let mut nodes = (0..ring_count)
        .map(|node_id| {
            let angle = std::f64::consts::TAU * node_id as f64 / ring_count as f64;
            ConstrainedCavityNode {
                node_id,
                coordinates_m: [angle.cos(), angle.sin(), 0.0],
            }
        })
        .collect::<Vec<_>>();
    nodes.push(ConstrainedCavityNode {
        node_id: top_node_id,
        coordinates_m: [0.0, 0.0, 1.0],
    });
    nodes.push(ConstrainedCavityNode {
        node_id: bottom_node_id,
        coordinates_m: [0.0, 0.0, -1.0],
    });

    let options = refill_options();
    let node_map = nodes
        .iter()
        .map(|node| (node.node_id, node.coordinates_m))
        .collect::<BTreeMap<_, _>>();
    let mut boundary_faces = Vec::<ConstrainedCavityBoundaryFace>::new();
    let mut expected_tetrahedra = Vec::<ConstrainedCavityRefillTetrahedron>::new();
    for node_id in 0..ring_count {
        let next_node_id = (node_id + 1) % ring_count;
        boundary_faces.push(ConstrainedCavityBoundaryFace {
            node_ids: [top_node_id, node_id, next_node_id],
            outside_tetrahedron_ids: Vec::new(),
            source_face_id: None,
            source_edge_ids: [None, None, None],
            region_ids: Vec::new(),
        });
        boundary_faces.push(ConstrainedCavityBoundaryFace {
            node_ids: [bottom_node_id, next_node_id, node_id],
            outside_tetrahedron_ids: Vec::new(),
            source_face_id: None,
            source_edge_ids: [None, None, None],
            region_ids: Vec::new(),
        });
        let tetrahedron_node_ids = [top_node_id, bottom_node_id, node_id, next_node_id];
        expected_tetrahedra.push(
            raw_refill_tetrahedron_with_rejection_reason(
                tetrahedron_node_ids,
                tetrahedron_node_ids.map(|id| node_map[&id]),
                options,
            )
            .expect("ring bipyramid tetrahedron should pass quality gates"),
        );
    }
    let expected_volume_m3 = expected_tetrahedra
        .iter()
        .map(|tetrahedron| tetrahedron.volume_m3)
        .sum::<f64>();
    let cavity = ConstrainedCavity {
        removed_tetrahedron_ids: vec![0],
        boundary_faces,
        protected_node_ids: Vec::new(),
        target_volume_m3: expected_volume_m3,
    };
    validate_constrained_cavity(&cavity).expect("ring bipyramid cavity should validate");
    let boundary_nodes = boundary_node_coordinates(&cavity, &nodes)
        .expect("fixture nodes should cover cavity boundary");
    let boundary_triangles = cavity_boundary_triangles(&cavity, &boundary_nodes)
        .expect("fixture boundary should build triangles");

    let refill = boundary_node_exact_cover_refill_candidate(
        &cavity,
        &boundary_nodes,
        &boundary_triangles,
        options,
    )
    .expect("exact cover should evaluate")
    .expect("bounded ring bipyramid should have an exact cover");

    assert_eq!(refill.tetrahedra.len(), ring_count as usize);
    validate_constrained_cavity_boundary_preserved(&cavity, &refill.boundary_faces)
        .expect("exact cover should preserve the larger cavity boundary");
    validate_constrained_cavity_refill_volume(
        cavity.target_volume_m3,
        refill.total_volume_m3,
        options.volume_relative_tolerance,
    )
    .expect("exact cover should preserve the larger cavity volume");
}

#[test]
fn exact_cover_on_demand_interior_mates_recovers_forced_mate() {
    let options = refill_options();
    let central = synthetic_refill_tetrahedron([0, 1, 2, 3], 1.0);
    let caps = [
        synthetic_refill_tetrahedron([0, 2, 1, 4], 1.0),
        synthetic_refill_tetrahedron([0, 1, 3, 5], 1.0),
        synthetic_refill_tetrahedron([0, 3, 2, 6], 1.0),
        synthetic_refill_tetrahedron([1, 2, 3, 7], 1.0),
    ];
    let shared_faces = BTreeSet::from([
        sorted_face([0, 1, 2]),
        sorted_face([0, 1, 3]),
        sorted_face([0, 2, 3]),
        sorted_face([1, 2, 3]),
    ]);
    let cavity = ConstrainedCavity {
        removed_tetrahedron_ids: vec![0],
        boundary_faces: caps
            .iter()
            .flat_map(|tetrahedron| tetrahedron_faces(tetrahedron.node_ids))
            .map(sorted_face)
            .filter(|face| !shared_faces.contains(face))
            .map(|node_ids| ConstrainedCavityBoundaryFace {
                node_ids,
                outside_tetrahedron_ids: Vec::new(),
                source_face_id: None,
                source_edge_ids: [None, None, None],
                region_ids: Vec::new(),
            })
            .collect(),
        protected_node_ids: Vec::new(),
        target_volume_m3: 5.0,
    };
    let refill = exact_cover_refill_from_on_demand_interior_mates(
        &cavity,
        caps.to_vec(),
        caps.into_iter().chain([central]).collect(),
        options,
    )
    .expect("on-demand exact cover should evaluate")
    .expect("on-demand mate injection should recover the cover");

    assert_eq!(refill.tetrahedra.len(), 5);
    validate_constrained_cavity_boundary_preserved(&cavity, &refill.boundary_faces)
        .expect("on-demand exact cover should preserve boundary");
    validate_constrained_cavity_refill_volume(
        cavity.target_volume_m3,
        refill.total_volume_m3,
        options.volume_relative_tolerance,
    )
    .expect("on-demand exact cover should preserve volume");
}
