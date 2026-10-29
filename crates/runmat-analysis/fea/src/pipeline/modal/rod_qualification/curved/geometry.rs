use super::*;
use crate::assembly::elements::solid::{tetrahedron10_is_affine, Tetrahedron10ElementGeometry};
use crate::topology::AnalysisProjection;
use runmat_meshing_core::TETRAHEDRON_MIDSIDE_EDGE_CORNERS;

#[test]
fn analytic_curved_geometry_moments_and_affine_axial_recovery_are_exact() {
    for cells in [1, 2, 4] {
        let mesh = curved_mesh(cells);
        let projection = AnalysisProjection::derive(&mesh).unwrap();
        let mut volume = 0.0;
        let mut first = [0.0; 3];
        for element in &projection.elements {
            let coordinates = std::array::from_fn(|i| {
                projection.nodes[element.node_indices[i]]
                    .coordinates_m
                    .canonical_value()
            });
            let geometry = Tetrahedron10ElementGeometry {
                nodes_m: coordinates,
            };
            assert!(!tetrahedron10_is_affine(geometry));
            // At least one edge varies in z. Its exact quadratic bow relative
            // to the endpoint chord is eta*(delta-z)^2/4, not roundoff.
            let mut bowed = false;
            for (edge, [a, b]) in TETRAHEDRON_MIDSIDE_EDGE_CORNERS.into_iter().enumerate() {
                let delta_z = coordinates[a][2] - coordinates[b][2];
                let bow = coordinates[edge + 4][1] - 0.5 * (coordinates[a][1] + coordinates[b][1]);
                assert!((bow - ETA * delta_z * delta_z / 4.0).abs() < 1e-13);
                bowed |= bow.abs() > 1e-8;
            }
            assert!(bowed);
            projection
                .visit_element_quadrature(element, |point| {
                    volume += point.volume_weight_m3;
                    for (axis, value) in first.iter_mut().enumerate() {
                        *value += point.position_m[axis] * point.volume_weight_m3;
                    }
                })
                .unwrap();
            // A physical affine patch exercises the shared curved inverse
            // Jacobian/recovery path. nu=0 gives sigma_xx=E*a, all else zero.
            let a = 0.017;
            let displacement = std::array::from_fn(|dof| {
                if dof % 3 == 0 {
                    a * coordinates[dof / 3][0]
                } else {
                    0.0
                }
            });
            for recovered in geometry.quadrature_strain(displacement).unwrap() {
                assert!((recovered.strain[0] - a).abs() < 1e-12);
                for value in &recovered.strain[1..] {
                    assert!(value.abs() < 1e-12);
                }
                let material = crate::assembly::elements::solid::SolidMaterial {
                    youngs_modulus_pa: runmat_types::physical::YoungsModulus::new(YOUNGS),
                    poisson_ratio: runmat_types::physical::PoissonRatio::new(0.0),
                    density_kg_per_m3: runmat_types::physical::MassDensity::new(1.0),
                };
                let response = crate::assembly::elements::solid::response::response_from_strain(
                    material,
                    recovered.strain,
                    0.0,
                )
                .unwrap();
                assert!((response.stress[0] - YOUNGS * a).abs() < 1e-10);
                for value in &response.stress[1..] {
                    assert!(value.abs() < 1e-10);
                }
                assert!(recovered.integration_weight_m3 > 0.0);
            }
        }
        assert!((volume - 1.0).abs() < 1e-10);
        for (actual, expected) in first.into_iter().zip([0.5, 0.5 + ETA / 6.0, 0.5]) {
            assert!((actual - expected).abs() < 1e-10);
        }
        for end in ["slab_left", "slab_right"] {
            let selector = runmat_meshing_core::fixtures::fixture_face(end);
            let faces = projection
                .boundary_faces_matching(&selector)
                .collect::<Vec<_>>();
            let area: f64 = faces
                .iter()
                .map(|face| face.area_m2.canonical_value())
                .sum();
            let centroid: [f64; 3] = std::array::from_fn(|axis| {
                faces
                    .iter()
                    .map(|face| face.area_m2.canonical_value() * face.centroid_m[axis])
                    .sum::<f64>()
                    / area
            });
            assert!((area - 1.0).abs() < 1e-10);
            assert!((centroid[1] - 0.5 - ETA / 6.0).abs() < 1e-10);
            assert!((centroid[2] - 0.5).abs() < 1e-10);
        }
        let curved_side_area = 0.5 * ((1.0 + ETA * ETA).sqrt() + ETA.asinh() / ETA);
        for face in projection.boundary_faces_matching(
            &runmat_meshing_core::fixtures::fixture_face("slab_insulated"),
        ) {
            projection
                .visit_boundary_quadrature(face, |point| {
                    assert!(point.area_vector_weight_m2[0].abs() < 1e-12);
                })
                .unwrap();
        }
        let outward_sum: [f64; 3] = std::array::from_fn(|axis| {
            projection
                .boundary_faces
                .iter()
                .map(|face| face.oriented_area_vector_m2[axis])
                .sum()
        });
        assert!(outward_sum.iter().all(|value| value.abs() < 1e-10));
        let surface_area: f64 = projection
            .boundary_faces
            .iter()
            .map(|face| face.area_m2.canonical_value())
            .sum();
        assert!((surface_area - (4.0 + 2.0 * curved_side_area)).abs() < 1e-9);
    }
}
