use super::*;

#[test]
fn curved_tet10_harmonic_loading_field_and_reaction_converge_to_continuum() {
    let wave_numbers = [0.5, 1.0, 2.0];
    let mut previous = [f64::INFINITY; 3];
    let mut previous_reaction = [f64::INFINITY; 3];
    for cells in [1, 2, 4] {
        let mesh = curved_mesh(cells);
        let model = rod_model(&mesh, &wave_numbers);
        for (case, k) in wave_numbers.into_iter().enumerate() {
            let assembly = assemble(&model, &mesh, k);
            for (load, phase) in [
                (assembly.real.operator.load(), PHASOR[0]),
                (&assembly.imaginary_load[..], PHASOR[1]),
            ] {
                let mut resultant = [0.0; 3];
                let mut moment = [0.0; 3];
                for (node, f) in mesh.topology.nodes.iter().zip(load.chunks_exact(3)) {
                    let [x, y, z] = node.coordinates_m.canonical_value();
                    for axis in 0..3 {
                        resultant[axis] += f[axis];
                    }
                    moment[0] += y * f[2] - z * f[1];
                    moment[1] += z * f[0] - x * f[2];
                    moment[2] += x * f[1] - y * f[0];
                }
                for (actual, expected) in resultant.into_iter().zip([phase, 0.0, 0.0]) {
                    assert!((actual - expected).abs() < 1e-12);
                }
                for (actual, expected) in
                    moment
                        .into_iter()
                        .zip([0.0, phase * 0.5, -phase * (0.5 + ETA / 6.0)])
                {
                    assert!((actual - expected).abs() < 1e-12);
                }
            }
            let omega = k * WAVE_SPEED;
            let response =
                solve_direct_harmonic(&assembly.real, &assembly.imaginary_load, omega, policy())
                    .unwrap();
            let error = field_error(
                &mesh,
                k,
                &response.displacement_real,
                &response.displacement_imaginary,
            );
            assert!(error < previous[case] / 6.0);
            previous[case] = error;
            let physical = evaluate_harmonic_physical_state(
                &assembly.real,
                omega,
                &response.displacement_real,
                &response.displacement_imaginary,
                assembly.real.operator.load(),
                &assembly.imaginary_load,
            )
            .unwrap();
            assert!(
                physical
                    .balance
                    .closure_metrics(&physical.reaction)
                    .unwrap()
                    .1
                    < 1e-8
            );
            let mut reaction_error = 0.0_f64;
            for (reaction, phase) in [
                (physical.reaction.real(), PHASOR[0]),
                (physical.reaction.imaginary(), PHASOR[1]),
            ] {
                let actual: f64 = reaction.chunks_exact(3).map(|v| v[0]).sum();
                reaction_error = reaction_error.max((actual / (-phase / k.cos()) - 1.0).abs());
            }
            assert!(reaction_error < previous_reaction[case] / 6.0);
            previous_reaction[case] = reaction_error;
            if cells == 4 {
                assert!(error < 0.002 && reaction_error < 0.002);
            }
            let input = 0.5
                * omega
                * assembly
                    .real
                    .operator
                    .load()
                    .iter()
                    .zip(&assembly.imaginary_load)
                    .zip(
                        response
                            .displacement_real
                            .iter()
                            .zip(&response.displacement_imaginary),
                    )
                    .map(|((fr, fi), (ur, ui))| fi * ur - fr * ui)
                    .sum::<f64>();
            assert!(input.abs() < 1e-9);
            assert!(response.relative_residual <= policy().relative_tolerance);
            eprintln!("curved Tet10 eta={ETA}, cells={cells}, k={k}, field={error}, reaction={reaction_error}, input={input}");
        }
    }
}
