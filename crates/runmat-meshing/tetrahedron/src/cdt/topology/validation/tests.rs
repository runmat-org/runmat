use super::*;
use runmat_geometry_core::PersistentEntityKind;
use runmat_meshing_core::NeverCancelled;

fn fixture() -> DelaunayVolumeTopology {
    let nodes = [
        [0., 0., 0.],
        [1., 0., 0.],
        [0., 1., 0.],
        [0., 0., 1.],
        [2., 0., 0.],
    ]
    .into_iter()
    .enumerate()
    .map(|(index, point)| {
        DelaunayVolumeNode::explicit(StableDigest::from_bytes([(index + 1) as u8; 32]), point)
    })
    .collect();
    let region = PersistentEntityId::new(PersistentEntityKind::Region, "solid".into(), vec![]);
    build_delaunay_volume_topology_with_regions(
        nodes,
        vec![([0, 1, 2, 3], Some(region)), ([1, 4, 2, 3], None)],
        DelaunayTopologyOptions::default(),
        &NeverCancelled,
    )
    .unwrap()
}

fn reference_accepts(topology: &DelaunayVolumeTopology) -> bool {
    build_delaunay_volume_topology_with_regions(
        topology.nodes.clone(),
        topology
            .tetrahedra
            .iter()
            .map(|cell| (cell.vertex_indices, cell.region_id.clone()))
            .collect(),
        DelaunayTopologyOptions::default(),
        &NeverCancelled,
    )
    .is_ok_and(|rebuilt| rebuilt == *topology)
}

// witness: FEA-SHARED-RESOURCES-001/bounded_local_search @
//   shared_infrastructure/canonical_mixed_topology/numerical_kernel;
//   boundary: Borrowed verification matches independent complete reconstruction on an assigned and
//   unassigned two-cell mesh. Twenty deliberately corrupted node, cell, orientation, neighbor,
//   boundary, vertex-star and region-incidence records reject under both validators. Compact face
//   scratch replaces a second complete mesh. This component family does not establish arbitrary
//   assembly scaling, native peak memory or complete quality convergence.
#[test]
fn borrowed_validation_matches_independent_reconstruction_and_rejects_tampering() {
    let base = fixture();
    assert!(reference_accepts(&base));
    validate_canonical_volume_topology(&base, DelaunayTopologyOptions::default(), &NeverCancelled)
        .unwrap();
    type TopologyCorruption = Box<dyn Fn(&mut DelaunayVolumeTopology)>;
    let corruptions: Vec<TopologyCorruption> = vec![
        Box::new(|mesh| mesh.nodes[0].identity = StableDigest::ZERO),
        Box::new(|mesh| {
            let mut coordinates = mesh.nodes[0].coordinates_m.canonical_value();
            coordinates[0] = f64::NAN;
            mesh.nodes[0].coordinates_m = runmat_types::physical::Quantity::new(coordinates);
        }),
        Box::new(|mesh| mesh.nodes[1].identity = mesh.nodes[0].identity),
        Box::new(|mesh| mesh.tetrahedra.swap(0, 1)),
        Box::new(|mesh| mesh.tetrahedra[0].vertex_indices.swap(0, 1)),
        Box::new(|mesh| mesh.tetrahedra[0].vertex_indices[0] = u32::MAX),
        Box::new(|mesh| mesh.tetrahedra[0].neighbors = [None; 4]),
        Box::new(|mesh| mesh.tetrahedra[0].neighbors[0] = Some(u32::MAX)),
        Box::new(|mesh| {
            mesh.incidence.vertex_stars[0].clear();
        }),
        Box::new(|mesh| {
            mesh.incidence.vertex_stars[0].push(1);
        }),
        Box::new(|mesh| {
            mesh.incidence.vertex_stars[0].push(0);
        }),
        Box::new(|mesh| {
            mesh.incidence.vertex_stars.pop();
        }),
        Box::new(|mesh| {
            mesh.incidence.boundary_facets.pop();
        }),
        Box::new(|mesh| {
            mesh.incidence.boundary_facets[0]
                .oriented_vertex_indices
                .swap(0, 1)
        }),
        Box::new(|mesh| {
            mesh.incidence.boundary_facets[0].opposite_vertex_slot =
                (mesh.incidence.boundary_facets[0].opposite_vertex_slot + 1) % 4
        }),
        Box::new(|mesh| {
            mesh.incidence.regions[0]
                .region_id
                .source_topology_id
                .clear()
        }),
        Box::new(|mesh| mesh.incidence.regions[0].tetrahedron_indices.clear()),
        Box::new(|mesh| mesh.incidence.regions[0].tetrahedron_indices.push(1)),
        Box::new(|mesh| mesh.incidence.unassigned_tetrahedron_indices.clear()),
        Box::new(|mesh| mesh.incidence.unassigned_tetrahedron_indices.push(0)),
    ];
    for (index, corrupt) in corruptions.into_iter().enumerate() {
        let mut changed = base.clone();
        corrupt(&mut changed);
        assert!(
            !reference_accepts(&changed),
            "corruption {index} must be meaningful"
        );
        assert!(
            validate_canonical_volume_topology(
                &changed,
                DelaunayTopologyOptions::default(),
                &NeverCancelled
            )
            .is_err(),
            "corruption {index} escaped validation"
        );
    }
}

#[test]
fn borrowed_validation_preserves_resource_and_cancellation_rejection() {
    struct Cancelled;
    impl MeshingCancellationSignal for Cancelled {
        fn is_cancelled(&self) -> bool {
            true
        }
    }
    let base = fixture();
    let options = DelaunayTopologyOptions::default();
    assert_eq!(
        validate_canonical_volume_topology(&base, options, &Cancelled)
            .unwrap_err()
            .kind,
        DelaunayTopologyErrorKind::Cancelled
    );
    let limited = DelaunayTopologyOptions {
        maximum_tetrahedra: 1,
        ..options
    };
    assert_eq!(
        validate_canonical_volume_topology(&base, limited, &NeverCancelled)
            .unwrap_err()
            .kind,
        DelaunayTopologyErrorKind::ResourceLimit
    );
}
