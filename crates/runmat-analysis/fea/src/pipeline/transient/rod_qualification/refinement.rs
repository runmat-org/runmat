use super::*;

pub(super) fn errors(
    mesh: &SolverMeshArtifact,
    damped: bool,
    result: &FeaTransientRunResult,
) -> [f64; 4] {
    let omega_squared = YOUNGS * std::f64::consts::FRAC_PI_2.powi(2);
    let scales = [
        AMPLITUDE,
        AMPLITUDE * omega_squared.sqrt(),
        AMPLITUDE * omega_squared,
    ];
    let mut squares = [0.0; 4];
    let initial_energy = reference::energies(0.0, damped)[0];
    for (sample, &time) in result.time_points_s.iter().enumerate() {
        let exact = reference::state(time, damped);
        let fields = [
            &result.displacement_snapshots[sample],
            &result.velocity_snapshots[sample],
            &result.acceleration_snapshots[sample],
        ];
        for quantity in 0..3 {
            let values = fields[quantity].as_host_f64().unwrap();
            assert_eq!(values.len(), mesh.topology.nodes.len() * 3);
            for (node, value) in mesh.topology.nodes.iter().zip(values.chunks_exact(3)) {
                assert!(value[1].abs() < 1.0e-12 && value[2].abs() < 1.0e-12);
                if node.coordinates_m.canonical_value()[0] == 0.0 {
                    assert!(value[0].abs() < 1.0e-12);
                }
                let difference = value[0]
                    - reference::shape(node.coordinates_m.canonical_value()[0]) * exact[quantity];
                squares[quantity] += (difference / scales[quantity]).powi(2);
            }
        }
        let expected = reference::energies(time, damped);
        let actual = [
            &result.strain_energy_snapshots[sample],
            &result.kinetic_energy_snapshots[sample],
            &result.damping_dissipation_snapshots[sample],
        ];
        for (field, expected) in actual.into_iter().zip(expected) {
            squares[3] += ((field.as_host_f64().unwrap()[0] - expected) / initial_energy).powi(2);
        }
        assert!(
            result.external_work_snapshots[sample]
                .as_host_f64()
                .unwrap()[0]
                .abs()
                < 1.0e-12
        );
        assert!(result.support_work_snapshots[sample].as_host_f64().unwrap()[0].abs() < 1.0e-12);
        assert!(
            result.relative_energy_balance_error_snapshots[sample]
                .as_host_f64()
                .unwrap()[0]
                .abs()
                < 1.0e-6
        );
    }
    let samples = result.time_points_s.len() as f64;
    for square in &mut squares[..3] {
        *square = (*square / (samples * mesh.topology.nodes.len() as f64)).sqrt();
    }
    squares[3] = (squares[3] / (3.0 * samples)).sqrt();
    squares
}

#[test]
fn transient_rod_tet4_and_tet10_converge_to_the_independent_damped_and_undamped_continuum() {
    // At 400 intervals, omega*dt=pi/400. Average-acceleration Newmark's independent
    // phase error estimate omega³*T*dt²/12 is about 1.6e-5 rad over this half-period.
    // This keeps temporal error below the intended spatial qualification bounds. The
    // bounds are normalized to fixed physical peaks/initial energy, never a vanishing
    // instantaneous mode value. Acceleration includes its complete nodal history.
    for quadratic in [false, true] {
        for damped in [false, true] {
            // Each order is refined until its spatial error sits below the bound: Tet10 nodal
            // acceleration converges at about h^2 (0.142, 0.029, 0.0061, 0.0014 over cells
            // 1, 2, 4, 8 at 400 intervals) and Tet4 acceleration at about h.
            // The fixed 5% acceleration criterion needs the next Tet4 rung:
            // 4/8/16 cells converge at h, but 16 cells still gives 0.050625.
            // Keep both rate gates and the independent absolute bound.
            let cells = if quadratic { [2, 4, 8] } else { [8, 16, 32] };
            let errors = cells.map(|cells| {
                let mesh = mesh(quadratic, cells);
                errors(&mesh, damped, &run(&mesh, damped, 400))
            });
            let factor = if quadratic { 3.0 } else { 1.6 };
            let bound = if quadratic { 0.005 } else { 0.05 };
            for quantity in 0..4 {
                assert!(
                    errors[1][quantity] < errors[0][quantity] / factor,
                    "quadratic={quadratic} damped={damped} quantity={quantity} errors={errors:?}"
                );
                assert!(
                    errors[2][quantity] < errors[1][quantity] / factor,
                    "quadratic={quadratic} damped={damped} quantity={quantity} errors={errors:?}"
                );
                assert!(
                    errors[2][quantity] < bound,
                    "quadratic={quadratic} damped={damped} quantity={quantity} errors={errors:?}"
                );
            }
            eprintln!(
                "transient continuum quadratic={quadratic} damped={damped} errors={errors:?}"
            );
        }
    }
}

#[test]
fn transient_rod_time_refinement_is_independent_of_spatial_refinement() {
    // Fixed fine quadratic topology isolates time discretization; there is no eigen extraction
    // or solver-to-solver reference and the continuum spatial floor is not subtracted.
    // At 40 intervals the same phase estimate is 0.001615 rad; the 0.2% normalized
    // U/V bound is therefore a physical time-accuracy criterion, not a residual target.
    // Eight quadratic cells put the spatial floor (displacement 1e-5, velocity 3.4e-5) far below
    // the temporal error being refined; on four cells the velocity floor (about 3.6e-4 damped,
    // more undamped) masks the time step's second order.
    let mesh = mesh(true, 8);
    for damped in [false, true] {
        let errors =
            [10, 20, 40].map(|intervals| errors(&mesh, damped, &run(&mesh, damped, intervals)));
        for quantity in 0..2 {
            assert!(
                errors[1][quantity] < errors[0][quantity] / 3.0,
                "damped={damped} quantity={quantity} errors={errors:?}"
            );
            assert!(
                errors[2][quantity] < errors[1][quantity] / 3.0,
                "damped={damped} quantity={quantity} errors={errors:?}"
            );
            assert!(
                errors[2][quantity] < 0.002,
                "damped={damped} quantity={quantity} errors={errors:?}"
            );
        }
        eprintln!("transient temporal damped={damped} errors={errors:?}");
    }
}

#[test]
#[ignore = "diagnostic: temporal order of the cold-block rod configuration"]
fn diagnostic_cold_block_rod_temporal_order() {
    let fixed = mesh(true, 4);
    for intervals in [80, 160, 320, 640] {
        eprintln!(
            "rod temporal intervals={intervals} errors={:?}",
            errors(&fixed, true, &run(&fixed, true, intervals))
        );
    }
    for cells in [1, 2, 4, 8] {
        let refined = mesh(true, cells);
        eprintln!(
            "rod spatial cells={cells} errors={:?}",
            errors(&refined, true, &run(&refined, true, 400))
        );
    }
}
