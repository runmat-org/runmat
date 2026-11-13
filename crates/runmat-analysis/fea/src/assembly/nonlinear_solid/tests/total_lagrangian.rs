//! Total Lagrangian kernel: objectivity, the Green-Lagrange/second Piola-Kirchhoff law and the
//! consistent (material plus geometric) tangent against independent finite differences.
use super::*;
use runmat_analysis_core::{ProcedureStep, StructuralKinematics, StructuralNonlinearSettings};

fn large(
    order: ElementOrder,
    plastic: bool,
) -> (AnalysisModel, SolverMeshArtifact, AssemblySummary) {
    let (mut model, mesh, initial) = fixture(order, plastic);
    model.sequence.steps = vec![ProcedureStep::structural_nonlinear_with(
        "large",
        StructuralNonlinearSettings {
            kinematics: StructuralKinematics::LargeDeformation,
        },
    )];
    (model, mesh, initial)
}

/// u = (G - I) X for a constant deformation gradient G.
fn homogeneous(mesh: &SolverMeshArtifact, initial: &AssemblySummary, g: [[f64; 3]; 3]) -> Vec<f64> {
    let mut u = vec![0.0; initial.dof_count];
    for (i, node) in mesh.topology.nodes.iter().enumerate() {
        let x = node.coordinates_m;
        for (axis, kind) in [
            StructuralDofKind::Ux,
            StructuralDofKind::Uy,
            StructuralDofKind::Uz,
        ]
        .into_iter()
        .enumerate()
        {
            u[initial.structural_dof_layout.index(i, kind).unwrap()] = (0..3)
                .map(|j| g[axis][j] * x.canonical_value()[j])
                .sum::<f64>()
                - x.canonical_value()[axis];
        }
    }
    u
}

fn rotation(angle: f64) -> [[f64; 3]; 3] {
    // Rotation about the unit axis (1, 2, 3)/sqrt(14).
    let n = [1.0, 2.0, 3.0].map(|v: f64| v / 14.0_f64.sqrt());
    let (s, c) = angle.sin_cos();
    std::array::from_fn(|i| {
        std::array::from_fn(|j| {
            let skew = [[0.0, -n[2], n[1]], [n[2], 0.0, -n[0]], [-n[1], n[0], 0.0]];
            (if i == j { c } else { 0.0 }) + s * skew[i][j] + (1.0 - c) * n[i] * n[j]
        })
    })
}

#[test]
fn rigid_rotation_is_strain_and_stress_free_and_small_strain_is_not() {
    for order in [ElementOrder::Tet4, ElementOrder::Tet10] {
        let (model, mesh, initial) = large(order, true);
        let basis = CanonicalNonlinearSolidBasis::prepare(&model, &mesh, &initial).unwrap();
        let u = homogeneous(&mesh, &initial, rotation(1.2));
        let evaluation = basis.evaluate(&u, &basis.zero_state()).unwrap();
        let force = evaluation
            .internal_force
            .iter()
            .fold(0.0_f64, |n, v| n.hypot(*v));
        assert!(
            force < 1e-10,
            "order={order:?} rigid rotation force {force}"
        );
        assert!(evaluation
            .point_evidence
            .iter()
            .all(|p| p.total_engineering_strain.iter().all(|e| e.abs() < 1e-13)));
        let (small_model, _, _) = fixture(order, true);
        let small = CanonicalNonlinearSolidBasis::prepare(&small_model, &mesh, &initial).unwrap();
        assert_ne!(small.identity(), basis.identity());
        let spurious = small.evaluate(&u, &small.zero_state()).unwrap();
        assert!(
            spurious
                .internal_force
                .iter()
                .fold(0.0_f64, |n, v| n.hypot(*v))
                > 1.0
        );
    }
}

#[test]
fn rotated_uniaxial_stretch_gives_saint_venant_kirchhoff_cauchy_stress() {
    let (e, nu) = (1200.0_f64, 0.25_f64);
    let lambda = e * nu / ((1.0 + nu) * (1.0 - 2.0 * nu));
    let mu = e / (2.0 * (1.0 + nu));
    for order in [ElementOrder::Tet4, ElementOrder::Tet10] {
        let (model, mesh, initial) = large(order, false);
        let basis = CanonicalNonlinearSolidBasis::prepare(&model, &mesh, &initial).unwrap();
        let stretch = 1.3;
        let r = rotation(0.7);
        // F = R diag(stretch, 1, 1)
        let g = std::array::from_fn(|i| {
            std::array::from_fn(|j| r[i][j] * if j == 0 { stretch } else { 1.0 })
        });
        let evaluation = basis
            .evaluate(&homogeneous(&mesh, &initial, g), &basis.zero_state())
            .unwrap();
        let e11 = 0.5 * (stretch * stretch - 1.0);
        let s = [(lambda + 2.0 * mu) * e11, lambda * e11, lambda * e11];
        // sigma = R diag(stretch² S11, S22, S33) Rᵀ / stretch
        let d = [
            stretch * stretch * s[0] / stretch,
            s[1] / stretch,
            s[2] / stretch,
        ];
        let sigma = |i: usize, j: usize| (0..3).map(|k| r[i][k] * d[k] * r[j][k]).sum::<f64>();
        let expected = [
            sigma(0, 0),
            sigma(1, 1),
            sigma(2, 2),
            sigma(1, 2),
            sigma(0, 2),
            sigma(0, 1),
        ];
        for point in &evaluation.point_evidence {
            for (actual, expected) in point.stress_pa.iter().zip(expected) {
                assert!(
                    (actual - expected).abs() < 1e-9 * s[0].abs(),
                    "order={order:?} {actual} {expected}"
                );
            }
            assert!((point.total_engineering_strain[0] - e11).abs() < 1e-13);
        }
    }
}

#[test]
fn total_lagrangian_tangent_differentiates_internal_force_elastic_and_plastic() {
    for plastic in [false, true] {
        for order in [ElementOrder::Tet4, ElementOrder::Tet10] {
            let (model, mesh, initial) = large(order, plastic);
            let basis = CanonicalNonlinearSolidBasis::prepare(&model, &mesh, &initial).unwrap();
            let committed = basis.zero_state();
            let r = rotation(0.9);
            let g = std::array::from_fn(|i| {
                std::array::from_fn(|j| {
                    r[i][j] + [[0.08, 0.01, 0.02], [0.0, -0.01, 0.03], [0.01, 0.0, 0.02]][i][j]
                })
            });
            let mut u = homogeneous(&mesh, &initial, g);
            for (i, value) in u.iter_mut().enumerate() {
                *value += 0.01 * (i as f64 * 0.7).cos();
            }
            let evaluation = basis.evaluate(&u, &committed).unwrap();
            if plastic {
                assert!(evaluation
                    .trial_state
                    .points
                    .iter()
                    .any(|p| p.equivalent_plastic_strain > 0.0));
            }
            let direction = (0..u.len())
                .map(|i| (i as f64 + 0.3).sin())
                .collect::<Vec<_>>();
            let h = 1e-7;
            let shifted = |sign: f64| {
                let v = u
                    .iter()
                    .zip(&direction)
                    .map(|(u, d)| u + sign * h * d)
                    .collect::<Vec<_>>();
                basis.evaluate(&v, &committed).unwrap().internal_force
            };
            let (plus, minus) = (shifted(1.0), shifted(-1.0));
            let exact = spmv(&evaluation.tangent, &direction).unwrap();
            let error = exact
                .iter()
                .zip(plus.iter().zip(&minus))
                .map(|(expected, (p, m))| ((p - m) / (2.0 * h) - expected).powi(2))
                .sum::<f64>()
                .sqrt();
            let norm = exact.iter().map(|x| x * x).sum::<f64>().sqrt();
            assert!(
                error / norm < 1e-6,
                "order={order:?} plastic={plastic} relative={}",
                error / norm
            );
        }
    }
}
