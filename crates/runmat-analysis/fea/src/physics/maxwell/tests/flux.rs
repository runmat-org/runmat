//! Independent face-normal and closed-boundary flux witnesses for recovered complex B.
use super::{fixture, *};
use crate::topology::AnalysisProjection;
use std::collections::BTreeMap;

#[test]
fn recovered_complex_flux_has_normal_continuity_and_pec_boundary_closure() {
    for order in [ElementOrder::Tet4, ElementOrder::Tet10] {
        let (mesh, input) = fixture::problem(
            order,
            2,
            ComplexPhasor {
                real: 0.6,
                imaginary: 0.8,
            },
        );
        let projection = AnalysisProjection::derive(&mesh).unwrap();
        let result = solve(&mesh, &input, 1e-10).unwrap();
        assert!(result.relative_charge_continuity_residual < 8e-10);
        let field: BTreeMap<_, _> = result
            .samples
            .iter()
            .map(|sample| {
                (
                    sample.element_identity,
                    sample.flux_density_t.canonical_value(),
                )
            })
            .collect();
        let mut faces = BTreeMap::<[usize; 3], Vec<ComplexVector3>>::new();
        for element in &projection.elements {
            for face in [[0, 1, 2], [0, 1, 3], [0, 2, 3], [1, 2, 3]] {
                let mut nodes = face.map(|i| element.node_indices[i]);
                nodes.sort_unstable();
                faces
                    .entry(nodes)
                    .or_default()
                    .push(field[&element.stable_identity]);
            }
        }
        for (nodes, fluxes) in faces.iter().filter(|(_, fields)| fields.len() == 2) {
            let a = projection.nodes[nodes[0]].coordinates_m.canonical_value();
            let b = projection.nodes[nodes[1]].coordinates_m.canonical_value();
            let c = projection.nodes[nodes[2]].coordinates_m.canonical_value();
            let u: [f64; 3] = std::array::from_fn(|axis| b[axis] - a[axis]);
            let v: [f64; 3] = std::array::from_fn(|axis| c[axis] - a[axis]);
            let normal = [
                u[1] * v[2] - u[2] * v[1],
                u[2] * v[0] - u[0] * v[2],
                u[0] * v[1] - u[1] * v[0],
            ];
            for imaginary in [false, true] {
                let part = |value: ComplexPhasor| {
                    if imaginary {
                        value.imaginary
                    } else {
                        value.real
                    }
                };
                let jump: f64 = (0..3)
                    .map(|axis| (part(fluxes[0][axis]) - part(fluxes[1][axis])) * normal[axis])
                    .sum();
                assert!(jump.abs() < 1e-12, "normal flux jump {jump}");
            }
        }
        // PEC tangential E=0 implies zero normal curl E on each exterior face,
        // stronger than net cancellation alone. Both real and imaginary parts.
        for imaginary in [false, true] {
            let mut net = 0.0;
            for face in &projection.boundary_faces {
                let owner = &projection.elements[face.adjacent_element_indices[0]];
                let flux: f64 = field[&owner.stable_identity]
                    .iter()
                    .zip(face.oriented_area_vector_m2)
                    .map(|(b, n)| (if imaginary { b.imaginary } else { b.real }) * n)
                    .sum();
                assert!(flux.abs() < 1e-12, "PEC normal flux {flux}");
                net += flux;
            }
            assert!(net.abs() < 1e-11, "closed complex flux {net}");
        }
    }
}
