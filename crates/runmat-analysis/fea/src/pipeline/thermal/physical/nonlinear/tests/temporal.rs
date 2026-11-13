//! Independent continuum cooling with all recovered field metrics retained.
use super::*;

// witness: FEA-THERMAL-SOLVE-001/transient @ thermal_transient/tet4|tet10/public_study;
//   check: radiating-cooling; boundary: Tet4/Tet10 manufactured radiating cooling, 8/16/32 steps, first-order
//   temperature/gradient/flux/storage and power closure; not temperature-dependent conductivity
//   coverage.
#[test]
fn manufactured_continuum_cooling_retains_first_order_backward_euler_and_power_closure() {
    for quadratic in [false, true] {
        let mesh = mesh(quadratic);
        let errors = [8, 16, 32].map(|intervals| {
            let input = manufactured(intervals, &mesh);
            let result = input.execute(&mesh).unwrap();
            assert_acceptance(&result, intervals);
            if intervals == 32 { retention::verify(&input, &mesh, &result); }
            assert_eq!(result.time_points_s.len(), intervals + 1);
            let temperature_error = result.time_points_s.iter().zip(&result.temperature_snapshots)
                .flat_map(|(&time, field)| mesh.topology.nodes.iter().zip(field.as_host_f64().unwrap())
                    .map(move |(node, &temperature)| (temperature
                        - (400.0 - 10.0 * node.coordinates_m.canonical_value()[0] - 20.0 * time * time)).abs()))
                .fold(0.0_f64, f64::max);
            let gradient_error = result.temperature_gradient_snapshots.iter().skip(1)
                .flat_map(|field| field.as_host_f64().unwrap().chunks_exact(3))
                .map(|gradient| norm(gradient.iter().zip([-10.0, 0.0, 0.0])
                    .map(|(actual, expected)| actual - expected)) / 10.0)
                .fold(0.0_f64, f64::max);
            let flux_error = result.heat_flux_snapshots.iter().skip(1)
                .flat_map(|field| field.as_host_f64().unwrap().chunks_exact(3))
                .map(|flux| norm(flux.iter().zip([20.0, 0.0, 0.0])
                    .map(|(actual, expected)| actual - expected)) / 20.0)
                .fold(0.0_f64, f64::max);
            let storage_error = result.time_points_s.iter().zip(&result.energy_storage_rate_snapshots)
                .skip(1).flat_map(|(&time, field)| field.as_host_f64().unwrap()
                    .iter().map(move |actual| (actual + 40.0 * time).abs() / 40.0))
                .fold(0.0_f64, f64::max);
            eprintln!("radiation manufactured quadratic={quadratic} intervals={intervals} temperature_abs_K={temperature_error:e} gradient_relative={gradient_error:e} flux_relative={flux_error:e} storage_relative={storage_error:e}");
            [temperature_error, gradient_error, flux_error, storage_error]
        });
        for quantity in 0..4 {
            assert!(
                errors[1][quantity] < errors[0][quantity] / 1.8
                    && errors[2][quantity] < errors[1][quantity] / 1.8,
                "quadratic={quadratic} quantity={quantity} original continuum errors={errors:?}"
            );
        }
        assert!(
            errors[2][0] < 0.1,
            "quadratic={quadratic} errors={errors:?}"
        );
    }
}
