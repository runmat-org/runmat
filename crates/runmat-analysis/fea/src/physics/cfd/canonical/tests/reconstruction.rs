//! Original mesh/face authority and independent affine velocity gradient.
use super::*;

#[test]
fn prepared_velocity_stencils_reproduce_original_affine_face_authority() {
    use runmat_analysis_core::{FluidConstraintKind, PrescribedValue, Velocity};
    let field = |x: [f64; 3]| [1.0 + x[1] + 0.3 * x[2], -0.2 + 0.4 * x[2], 0.1];
    for n in [1, 2, 3] {
        for rotated in [false, true] {
            let mut f = case(n, ReferenceCase::Couette, rotated);
            for constraint in &mut f.constraints {
                let FluidConstraintKind::PrescribedVelocity { target, value } =
                    &mut constraint.kind
                else {
                    panic!("original velocity fixture")
                };
                let face = f
                    .source
                    .boundary_faces
                    .iter()
                    .find(|face| {
                        face.provenance
                            .iter()
                            .any(|entity| target.matches_persistent_entity(entity))
                    })
                    .unwrap();
                let x = if rotated {
                    [face.centroid_m[1], face.centroid_m[2], face.centroid_m[0]]
                } else {
                    face.centroid_m
                };
                let v = field(x);
                *value = PrescribedValue::Constant(
                    (if rotated { reference::rotate(v) } else { v }).map(Velocity::new),
                );
            }
            let p = CanonicalFluidProblem::prepare(f.input()).unwrap();
            let values = f
                .topology
                .cells
                .iter()
                .map(|c| {
                    let x = if rotated {
                        [c.centroid_m[1], c.centroid_m[2], c.centroid_m[0]]
                    } else {
                        c.centroid_m
                    };
                    let v = field(x);
                    if rotated {
                        reference::rotate(v)
                    } else {
                        v
                    }
                })
                .collect::<Vec<_>>();
            let expected = if rotated {
                [[0.0; 3], [0.3, 0.0, 1.0], [0.4, 0.0, 0.0]]
            } else {
                [[0.0, 1.0, 0.3], [0.0, 0.0, 0.4], [0.0; 3]]
            };
            for gradient in geometry_ops::velocity_gradients(&p, &values).unwrap() {
                for k in 0..3 {
                    assert!(
                        norm(sub(gradient[k], expected[k])) < 2e-14,
                        "cells={n}, rotated={rotated}, original gradient={gradient:?}"
                    );
                }
            }
            assert!(super::super::velocity_gradient::evaluate(&p, &[]).is_err());
            assert!(super::super::velocity_gradient::evaluate(
                &p,
                &vec![[f64::NAN; 3]; values.len()]
            )
            .is_err());
        }
    }
}
