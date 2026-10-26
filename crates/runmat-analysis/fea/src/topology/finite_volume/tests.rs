use super::{super::AnalysisProjection, geometry, *};
use runmat_meshing_core::{fixtures, ElementOrder, StableDigest};

fn policy() -> FiniteVolumeProjectionPolicy {
    FiniteVolumeProjectionPolicy {
        max_cells: Some(10000),
        max_face_incidences: Some(40000),
        max_boundary_faces: Some(40000),
        max_scratch_bytes: Some(100_000_000),
        max_traversal_operations: Some(100_000_000),
    }
}

fn projected(order: ElementOrder) -> AnalysisProjection {
    AnalysisProjection::derive(&fixtures::canonical_tetrahedron_solver_mesh(order)).unwrap()
}

fn two_cells() -> AnalysisProjection {
    // Two unit right tetrahedra share z=0; exact volumes are 1/6 each.
    let mut source = projected(ElementOrder::Tet4);
    let mut node = source.nodes[3].clone();
    node.coordinates_m = runmat_types::physical::Quantity::<
        runmat_types::physical::kind::Position,
        _,
    >::new([0.0, 0.0, -1.0]);
    node.node_id = 5;
    node.stable_identity = StableDigest::from_bytes([231; 32]);
    source.nodes.push(node);
    let mut second = source.elements[0].clone();
    second.node_indices = vec![0, 2, 1, 4];
    second.element_id = 2;
    second.stable_identity = StableDigest::from_bytes([232; 32]);
    second.centroid_m = [0.25, 0.25, -0.25];
    source.elements.push(second);
    let template = source.boundary_faces[0].clone();
    source.boundary_faces.clear();
    let mut candidates =
        std::collections::BTreeMap::<[StableDigest; 3], (usize, [usize; 3], usize)>::new();
    for (i, element) in source.elements.iter().enumerate() {
        for local in geometry::FACE_CORNERS {
            let nodes = local.map(|j| element.node_indices[j]);
            let key = geometry::key(&source, &nodes).unwrap();
            candidates
                .entry(key)
                .and_modify(|v| v.2 += 1)
                .or_insert((i, nodes, 1));
        }
    }
    for (key, (cell, nodes, count)) in candidates {
        if count == 2 {
            continue;
        }
        let mut face = template.clone();
        face.node_indices = nodes.to_vec();
        face.adjacent_element_indices = vec![cell];
        face.face_id = source.boundary_faces.len() as u64 + 100;
        face.stable_identity = StableDigest::from_bytes(
            runmat_execution::Digest::sha256_parts(key.map(|k| k.0))
                .bytes()
                .to_owned(),
        );
        let (centroid, area, magnitude) = geometry::face_geometry(
            nodes.map(|n| source.nodes[n].coordinates_m.canonical_value()),
            source.elements[cell].centroid_m,
        )
        .unwrap();
        face.centroid_m = centroid;
        face.oriented_area_vector_m2 = area;
        face.area_m2 = runmat_types::physical::Area::new(magnitude);
        source.boundary_faces.push(face);
    }
    source
}

// witness: FEA-FLUID-TOPOLOGY-001/inventory @
//   incompressible_laminar_steady/canonical_tetrahedra/numerical_kernel;
//   boundary: Two affine Tet4 cells; oriented surface closure, volume and original face mapping. No
//   skew/curved mesh envelope proof.
#[test]
fn exact_two_cell_geometry_is_conservative_and_retains_original_boundary_mapping() {
    let source = two_cells();
    let fv = FiniteVolumeTopology::derive(&source, policy()).unwrap();
    assert_eq!(fv.cells.len(), 2);
    assert_eq!(fv.faces.len(), 7);
    let internal = fv.faces.iter().find(|f| f.neighbor.is_some()).unwrap();
    assert_eq!(internal.area_m2.canonical_value(), 0.5);
    assert_eq!(internal.centroid_m, [1.0 / 3.0, 1.0 / 3.0, 0.0]);
    assert!((fv.cells.iter().map(|c| c.volume_m3).sum::<f64>() - 1.0 / 3.0).abs() < 1e-15);
    for (i, cell) in fv.cells.iter().enumerate() {
        assert!((cell.volume_m3 - 1.0 / 6.0).abs() < 1e-15);
        let mut sum = [0.0; 3];
        for incidence in cell.faces {
            let face = &fv.faces[incidence.face_index];
            assert_eq!(incidence.sign, if face.owner == i { 1 } else { -1 });
            for (a, b) in sum.iter_mut().zip(face.owner_area_vector_m2) {
                *a += f64::from(incidence.sign) * b;
            }
        }
        assert_eq!(sum, [0.0; 3]);
    }
    for face in fv.faces.iter().filter(|f| f.neighbor.is_none()) {
        let original =
            &source.boundary_faces[face.boundary_source.unwrap().projection_boundary_index];
        assert_eq!(original.provenance, source.boundary_faces[0].provenance);
        assert_eq!(
            original.stable_identity,
            face.boundary_source.unwrap().face_identity
        );
    }
    fv.validate_for(&source).unwrap();
}

#[test]
fn refined_cube_exact_volume_and_both_affine_orders_have_closed_faces() {
    for order in [ElementOrder::Tet4, ElementOrder::Tet10] {
        for refinement in [1, 2, 3] {
            let mesh = if order == ElementOrder::Tet4 {
                fixtures::slab_mesh(refinement)
            } else {
                fixtures::quadratic_slab_mesh(refinement)
            };
            let source = AnalysisProjection::derive(&mesh).unwrap();
            let fv = FiniteVolumeTopology::derive(&source, policy()).unwrap();
            assert!((fv.cells.iter().map(|c| c.volume_m3).sum::<f64>() - 1.0).abs() < 1e-13);
            let outer = fv
                .faces
                .iter()
                .filter(|f| f.neighbor.is_none())
                .map(|f| f.area_m2.canonical_value())
                .sum::<f64>();
            assert!((outer - 6.0).abs() < 1e-13);
        }
    }
}

// witness: FEA-FLUID-TOPOLOGY-001/geometry @
//   incompressible_laminar_steady/canonical_tetrahedra/numerical_kernel;
//   boundary: Two-cell source access reordering, not execution placement equivalence.
#[test]
fn stable_cell_face_physics_is_invariant_to_reordered_source_access_indices() {
    let source = two_cells();
    let expected = FiniteVolumeTopology::derive(&source, policy()).unwrap();
    let mut shuffled = source.clone();
    shuffled.elements.reverse();
    shuffled.boundary_faces.reverse();
    for face in &mut shuffled.boundary_faces {
        for cell in &mut face.adjacent_element_indices {
            *cell = 1 - *cell;
        }
    }
    // Also reorder nodes, adjusting every original access index.
    shuffled.nodes.reverse();
    let n = shuffled.nodes.len();
    for element in &mut shuffled.elements {
        for node in &mut element.node_indices {
            *node = n - 1 - *node;
        }
    }
    for face in &mut shuffled.boundary_faces {
        for node in &mut face.node_indices {
            *node = n - 1 - *node;
        }
    }
    let actual = FiniteVolumeTopology::derive(&shuffled, policy()).unwrap();
    assert_eq!(actual.geometry_digest, expected.geometry_digest);
    for (a, b) in actual.cells.iter().zip(&expected.cells) {
        assert_eq!(a.element_identity, b.element_identity);
        assert_eq!(a.faces, b.faces);
        assert_eq!(a.centroid_m, b.centroid_m);
    }
    for (a, b) in actual.faces.iter().zip(&expected.faces) {
        assert_eq!(a.corner_identities, b.corner_identities);
        assert_eq!(a.owner, b.owner);
        assert_eq!(a.neighbor, b.neighbor);
        assert_eq!(a.owner_area_vector_m2, b.owner_area_vector_m2);
    }
}

#[test]
fn rigid_rotation_tiny_scale_and_translation_preserve_vector_geometry() {
    let mut mesh = fixtures::canonical_tetrahedron_solver_mesh(ElementOrder::Tet4);
    for node in &mut mesh.topology.nodes {
        let [x, y, z] = node.coordinates_m.canonical_value();
        node.coordinates_m = runmat_types::physical::Quantity::<
            runmat_types::physical::kind::Position,
            _,
        >::new([-y * 1e-6 + 2e-6, x * 1e-6 - 3e-6, z * 1e-6 + 4e-6]);
    }
    mesh.seal_canonical_digest().unwrap();
    let source = AnalysisProjection::derive(&mesh).unwrap();
    let fv = FiniteVolumeTopology::derive(&source, policy()).unwrap();
    assert!((fv.cells[0].volume_m3 / (1e-18 / 6.0) - 1.0).abs() < 1e-13);
    let mut sum = [0.0; 3];
    for face in fv.faces {
        for (a, b) in sum.iter_mut().zip(face.owner_area_vector_m2) {
            *a += b;
        }
    }
    assert!(geometry::length(sum) < 1e-25);
}

// witness: FEA-FLUID-TOPOLOGY-001/quality @
//   incompressible_laminar_steady/canonical_tetrahedra/numerical_kernel;
//   boundary: Selected malformed source cases only, not complete quality-envelope convergence.
#[test]
fn rejects_nonmanifold_missing_boundary_inversion_curvature_and_mutated_record() {
    let mut source = two_cells();
    let mut third = source.elements[1].clone();
    third.element_id = 3;
    third.stable_identity = StableDigest::from_bytes([233; 32]);
    source.elements.push(third);
    assert!(FiniteVolumeTopology::derive(&source, policy()).is_err());
    let mut source = two_cells();
    source.boundary_faces.pop();
    assert_eq!(
        FiniteVolumeTopology::derive(&source, policy()).unwrap_err(),
        FiniteVolumeProjectionError::Invalid("missing original exterior boundary")
    );
    let mut source = projected(ElementOrder::Tet4);
    source.elements[0].node_indices.swap(1, 2);
    assert!(FiniteVolumeTopology::derive(&source, policy()).is_err());
    let mut source = projected(ElementOrder::Tet10);
    let mid = source.elements[0].node_indices[4];
    {
        let quantity_change = 0.01;
        let quantity_index = 2;
        let quantity_slot = &mut source.nodes[mid].coordinates_m;
        let mut quantity_coordinates = quantity_slot.canonical_value();
        quantity_coordinates[quantity_index] += quantity_change;
        *quantity_slot = runmat_types::physical::Quantity::<
            runmat_types::physical::kind::Position,
            _,
        >::new(quantity_coordinates);
    };
    assert!(matches!(
        FiniteVolumeTopology::derive(&source, policy()),
        Err(FiniteVolumeProjectionError::CurvedTet10(_))
    ));
    let source = two_cells();
    let mut fv = FiniteVolumeTopology::derive(&source, policy()).unwrap();
    fv.cells[0].volume_m3 *= 2.0;
    assert_eq!(
        fv.validate_for(&source).unwrap_err(),
        FiniteVolumeProjectionError::SourceMismatch
    );
}

// witness: FEA-FLUID-TOPOLOGY-001/budget @
//   incompressible_laminar_steady/canonical_tetrahedra/numerical_kernel;
//   boundary: Exact finite-volume construction counts and one-below limits; requested bytes are not
//   RSS.
#[test]
fn exact_admission_bounds_succeed_and_each_one_below_rejects() {
    let source = two_cells();
    let r = FiniteVolumeTopology::reserve(
        &source,
        policy(),
        &crate::topology::FiniteVolumeCellSelection::AllElements,
    )
    .unwrap();
    let exact = FiniteVolumeProjectionPolicy {
        max_cells: Some(r.cell_count),
        max_face_incidences: Some(r.face_incidence_count),
        max_boundary_faces: Some(r.boundary_face_count),
        max_scratch_bytes: Some(r.scratch_bytes),
        max_traversal_operations: Some(r.traversal_operations),
    };
    FiniteVolumeTopology::derive(&source, exact).unwrap();
    for field in 0..5 {
        let mut p = exact;
        match field {
            0 => p.max_cells = p.max_cells.map(|v| v - 1),
            1 => p.max_face_incidences = p.max_face_incidences.map(|v| v - 1),
            2 => p.max_boundary_faces = p.max_boundary_faces.map(|v| v - 1),
            3 => p.max_scratch_bytes = p.max_scratch_bytes.map(|v| v - 1),
            _ => p.max_traversal_operations = p.max_traversal_operations.map(|v| v - 1),
        }
        assert!(matches!(
            FiniteVolumeTopology::derive(&source, p),
            Err(FiniteVolumeProjectionError::Limit(_))
        ));
    }
}

#[test]
fn count_reservations_reject_checked_overflow_without_allocating() {
    assert_eq!(
        super::derive::checked_counts(usize::MAX, 0).unwrap_err(),
        FiniteVolumeProjectionError::Overflow
    );
    assert_eq!(
        super::derive::checked_counts(1, usize::MAX).unwrap_err(),
        FiniteVolumeProjectionError::Overflow
    );
}
