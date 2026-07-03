use super::*;

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
