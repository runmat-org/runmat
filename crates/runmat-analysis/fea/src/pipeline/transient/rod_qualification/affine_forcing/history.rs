//! Independent continuum support traction and whole-body momentum at every retained sample.
use super::*;

pub(super) fn errors(
    mesh: &SolverMeshArtifact,
    damped: bool,
    result: &FeaTransientRunResult,
) -> [f64; 3] {
    let samples = result.time_points_s.len();
    assert_eq!(result.reaction_force_snapshots.len(), samples);
    // Solid translational DOFs have no rotational reaction field.
    assert!(result.reaction_moment_snapshots.is_empty());
    assert_eq!(result.dynamic_balance_snapshots.len(), samples);
    let mut squares = [0.0; 3];
    let scale = YOUNGS * AMPLITUDE;
    for (sample, &time) in result.time_points_s.iter().enumerate() {
        let ([q, v, a], _) = reference(time, damped);
        let alpha = if damped { ALPHA } else { 0.0 };
        let beta = if damped { BETA } else { 0.0 };
        // Unit length/area/rho: integral x dx=1/2. No assembled matrices enter reference.
        let tip = YOUNGS * (q + beta * v);
        let exact_reaction = -tip;
        let exact_inertia = a / 2.0;
        let exact_damping = alpha * v / 2.0;
        let exact_applied = tip + (a + alpha * v) / 2.0;
        assert!((exact_applied + exact_reaction - exact_inertia - exact_damping).abs() < 1e-14);
        let reactions = result.reaction_force_snapshots[sample]
            .as_host_f64()
            .unwrap();
        assert_eq!(reactions.len(), mesh.topology.nodes.len() * 3);
        let mut root = 0.0;
        let mut transverse = [0.0; 2];
        for (node, vector) in mesh.topology.nodes.iter().zip(reactions.chunks_exact(3)) {
            transverse[0] += vector[1];
            transverse[1] += vector[2];
            if node.coordinates_m.canonical_value()[0] == 0.0 {
                root += vector[0];
            } else {
                assert!(vector[0].abs() < 1e-11);
            }
        }
        assert!(transverse.iter().all(|value| value.abs() < 1e-11));
        let balance = &result.dynamic_balance_snapshots[sample];
        balance.validate().unwrap();
        let axial = |values: &[f64]| values.chunks_exact(3).map(|v| v[0]).sum::<f64>();
        let inertia = axial(&balance.inertia_force);
        let damping = axial(&balance.damping_force);
        let elastic = axial(&balance.elastic_force);
        let applied = axial(&balance.applied_load);
        assert!(
            (applied - exact_applied).abs() < 1e-11,
            "original physical load at t={time}"
        );
        assert!(
            elastic.abs() < 1e-11,
            "internal elastic forces cancel at t={time}"
        );
        let defect = inertia + damping + elastic - applied - root;
        assert!(
            defect.abs() < 1e-8 * scale,
            "original dynamic momentum at t={time}: {defect}"
        );
        for (sum, error) in squares.iter_mut().zip([
            root - exact_reaction,
            inertia - exact_inertia,
            damping - exact_damping,
        ]) {
            *sum += (error / scale).powi(2);
        }
    }
    squares.map(|sum| (sum / samples as f64).sqrt())
}

#[test]
fn affine_continuum_reaction_and_original_momentum_histories_converge() {
    for quadratic in [false, true] {
        let mesh = super::super::mesh(quadratic, 1);
        for damped in [false, true] {
            let errors = [20, 40, 80].map(|n| errors(&mesh, damped, &run_forced(&mesh, damped, n)));
            eprintln!(
                "affine reaction/inertia/damping quadratic={quadratic} damped={damped}: {errors:?}"
            );
            for quantity in 0..3 {
                if quantity == 2 && !damped {
                    assert!(errors.iter().all(|e| e[quantity] < 1e-12));
                    continue;
                }
                assert!(
                    errors[1][quantity] < errors[0][quantity] / 3.0,
                    "quantity={quantity} errors={errors:?}"
                );
                assert!(
                    errors[2][quantity] < errors[1][quantity] / 3.0,
                    "quantity={quantity} errors={errors:?}"
                );
                assert!(
                    errors[2][quantity] < 0.0002,
                    "quantity={quantity} errors={errors:?}"
                );
            }
        }
    }
}
