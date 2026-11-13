use runmat_meshing_core::{fixtures, ElementOrder};

use super::{boundary::project_boundary_face, AnalysisProjection};

// witness: FEA-STRUCTURAL-SOLID-001/quadrature_mass @
//   linear_structural/tet4|tet10/numerical_kernel; coverage: partial;
//   boundary: Oblique Tet4 and Tet10 volume quadrature reproduces all world-affine gradients; mass
//   matrices and face quadrature are other tests.
#[test]
fn oblique_both_order_volume_quadrature_reproduces_all_world_affine_gradients() {
    for order in [ElementOrder::Tet4, ElementOrder::Tet10] {
        let mut mesh = fixtures::canonical_tetrahedron_solver_mesh(order);
        for node in &mut mesh.topology.nodes {
            let [x, y, z] = node.coordinates_m.canonical_value();
            node.coordinates_m = runmat_types::physical::Quantity::<
                runmat_types::physical::kind::Position,
                _,
            >::new([
                2.0 * x + y + 3.0 * z + 0.3,
                4.0 * y + z - 0.7,
                x + 2.0 * y + 5.0 * z + 1.2,
            ]);
        }
        mesh.seal_canonical_digest().unwrap();
        mesh.validate().unwrap();
        let projection = AnalysisProjection::derive(&mesh).unwrap();
        for element in &projection.elements {
            projection.visit_element_quadrature(element, |point| {
                for world_axis in 0..3 {
                    for axis in 0..3 {
                        let actual = element.node_indices.iter().zip(point.physical_shape_gradients())
                            .map(|(&node, gradient)| projection.nodes[node].coordinates_m.canonical_value()[world_axis] * gradient[axis])
                            .sum::<f64>();
                        let expected = if axis == world_axis { 1.0 } else { 0.0 };
                        assert!((actual - expected).abs() < 1e-12, "order={order:?}, world_axis={world_axis}, axis={axis}, actual={actual}");
                    }
                }
            }).unwrap();
        }
    }
}

#[test]
fn curved_boundary_face_rejects_a_folded_signed_jacobian() {
    let mesh = fixtures::canonical_tetrahedron_solver_mesh(ElementOrder::Tet10);
    let projection = AnalysisProjection::derive(&mesh).expect("straight Tet10 projection");
    let face = &mesh.topology.boundary_faces[0];
    let projected_face = &projection.boundary_faces[0];
    let mut nodes = projection.nodes.clone();
    // This mild shift leaves every seven-point Dunavant sample positive, but
    // reverses signed area close to the u=1 corner (q(1,0) = -0.04).
    let midside_index = projected_face.node_indices[3];
    nodes[midside_index].coordinates_m = runmat_types::physical::Quantity::<
        runmat_types::physical::kind::Position,
        _,
    >::new([0.76, 0.0, 0.0]);

    let error = project_boundary_face(
        face,
        projected_face.node_indices.clone(),
        projected_face.adjacent_element_indices.clone(),
        &nodes,
        &projection.elements,
    )
    .expect_err("a folded pressure face must be rejected");

    assert!(matches!(
        error,
        super::AnalysisProjectionError::InvalidBoundaryGeometry { face_id: 10 }
    ));
}

// witness: FEA-SHARED-TOPOLOGY-001/projection_inventory @
//   shared_infrastructure/canonical_mixed_topology/numerical_kernel;
//   boundary: Single affine Tet10 tetrahedron checks node/element/face inventory, positive volume,
//   integrated shapes and world gradient, outward normals and repeatable projection digest; no
//   arbitrary mixed topology or distributed equivalence.
#[test]
fn canonical_tetrahedron_projection_has_indexed_oriented_geometry() {
    let mesh = fixtures::canonical_tetrahedron_solver_mesh(ElementOrder::Tet10);
    let projection = AnalysisProjection::derive(&mesh).expect("valid analysis projection");

    assert_eq!(projection.mesh_digest, mesh.canonical_digest);
    assert_eq!(projection.schema_version, 1);
    assert_eq!(
        projection.projection_digest,
        AnalysisProjection::derive(&mesh).unwrap().projection_digest
    );
    assert_eq!(projection.nodes.len(), 10);
    assert_eq!(projection.elements.len(), 1);
    assert_eq!(projection.boundary_faces.len(), 4);
    assert_eq!(projection.elements[0].node_indices.len(), 10);
    assert!(projection.elements[0].signed_volume_m3 > 0.0);
    let mut integrated_volume = 0.0;
    let mut integrated_shapes = [0.0; 10];
    projection
        .visit_element_quadrature(&projection.elements[0], |point| {
            integrated_volume += point.volume_weight_m3;
            for (node, &shape) in point.shape_values().iter().enumerate() {
                integrated_shapes[node] += shape * point.volume_weight_m3;
            }
            let gradient_sum =
                point
                    .physical_shape_gradients()
                    .iter()
                    .fold([0.0; 3], |mut sum, gradient| {
                        for axis in 0..3 {
                            sum[axis] += gradient[axis];
                        }
                        sum
                    });
            let reproduced_x_gradient = point
                .physical_shape_gradients()
                .iter()
                .zip(&projection.elements[0].node_indices)
                .fold([0.0; 3], |mut sum, (gradient, &node_index)| {
                    let x = projection.nodes[node_index].coordinates_m.canonical_value()[0];
                    for axis in 0..3 {
                        sum[axis] += gradient[axis] * x;
                    }
                    sum
                });
            for component in gradient_sum {
                assert!(component.abs() <= 1.0e-12);
            }
            assert!((reproduced_x_gradient[0] - 1.0).abs() <= 1.0e-12);
            assert!(reproduced_x_gradient[1].abs() <= 1.0e-12);
            assert!(reproduced_x_gradient[2].abs() <= 1.0e-12);
        })
        .unwrap();
    assert!((integrated_volume - projection.elements[0].signed_volume_m3).abs() <= 1.0e-12);
    let volume = projection.elements[0].signed_volume_m3;
    for value in &integrated_shapes[..4] {
        assert!((*value + volume / 20.0).abs() <= 1.0e-12);
    }
    for value in &integrated_shapes[4..] {
        assert!((*value - volume / 5.0).abs() <= 1.0e-12);
    }
    for face in &projection.boundary_faces {
        let interior_direction = std::array::from_fn::<_, 3, _>(|axis| {
            projection.elements[face.adjacent_element_indices[0]].centroid_m[axis]
                - face.centroid_m[axis]
        });
        let inward_dot = face
            .oriented_area_vector_m2
            .iter()
            .zip(interior_direction)
            .map(|(normal, direction)| normal * direction)
            .sum::<f64>();
        assert!(inward_dot < 0.0);
        assert!(face.area_m2.canonical_value() > 0.0);
        assert_eq!(face.node_indices.len(), 6);
    }
    assert!(projection
        .node_neighbors
        .iter()
        .all(|neighbors| !neighbors.is_empty()));
    assert!(projection.characteristic_length_m > 0.0);
}

// witness: FEA-SHARED-TOPOLOGY-001/projection_invariants @
//   shared_infrastructure/canonical_mixed_topology/numerical_kernel;
//   boundary: Tet4 node-coordinate substitution without digest resealing rejects projection; no
//   exhaustive provenance or interface invalid-case corpus.
#[test]
fn projection_rejects_a_substituted_mesh_digest() {
    let mut mesh = fixtures::canonical_tetrahedron_solver_mesh(ElementOrder::Tet4);
    {
        let quantity_change = 0.25;
        let quantity_index = 0;
        let quantity_slot = &mut mesh.topology.nodes[0].coordinates_m;
        let mut quantity_coordinates = quantity_slot.canonical_value();
        quantity_coordinates[quantity_index] += quantity_change;
        *quantity_slot = runmat_types::physical::Quantity::<
            runmat_types::physical::kind::Position,
            _,
        >::new(quantity_coordinates);
    };

    let error = AnalysisProjection::derive(&mesh).unwrap_err();

    assert!(error
        .to_string()
        .contains("canonical solver mesh is invalid"));
}
