//! Original conduction/radiation references through native execution and durable publication.
use runmat_analysis_core::*;
use runmat_analysis_execution::CompletedThermalStudy;
use runmat_analysis_fea::{
    execution::ThermalWorkload,
    physics::thermal::qualification::{self as fixture, TARGET},
};
use runmat_meshing_core::SolverMeshArtifact;
#[path = "thermal_radiation/fields.rs"]
mod fields;
#[path = "thermal_radiation/lifecycle.rs"]
mod lifecycle;

pub(super) async fn run() {
    run_intervals(&[8, 16, 32]).await;
}

/// Original finest case only, for diagnosing admission without repeating coarser evidence.
/// This is not the complete temporal-convergence qualification matrix.
pub(super) async fn run_fine() {
    run_intervals(&[32]).await;
}

async fn run_intervals(intervals: &[usize]) {
    let (surface, power) = fixture::reference::steady_surface_and_power();
    for quadratic in [false, true] {
        let mesh = fixture::mesh(quadratic);
        let mut input = fixture::workload();
        fixture::left_temperature(
            &mut input,
            PrescribedValue::Constant(Temperature::new(400.0)),
        );
        let done = lifecycle::run(input, &mesh).await;
        fields::acceptance(&done, 1);
        for (node, &value) in mesh.topology.nodes.iter().zip(
            fields::original(&done, &mesh, 0, 0)
                .storage()
                .as_host_f64()
                .unwrap(),
        ) {
            assert!(
                (value - (400.0 + (surface - 400.0) * node.coordinates_m.canonical_value()[0]))
                    .abs()
                    / 400.0
                    < 1e-8
            );
        }
        let balance = done.result.manifest.energy_balances[0];
        assert!((balance.boundary_loss_w.canonical_value() / power - 1.0).abs() < 1e-8);
        assert!((balance.prescribed_boundary_power_w.canonical_value() / power + 1.0).abs() < 1e-8);
        assert_eq!(
            balance.applied_power_w,
            runmat_types::physical::Quantity::new(0.0)
        );
        assert_eq!(
            balance.storage_rate_w,
            runmat_types::physical::Quantity::new(0.0)
        );
        for (index, expected) in [(1, [surface - 400.0, 0.0, 0.0]), (2, [power, 0.0, 0.0])] {
            let field = fields::original(&done, &mesh, 0, index);
            for values in field.storage().as_host_f64().unwrap().chunks_exact(3) {
                assert!(
                    norm(values.iter().zip(expected).map(|(v, e)| v - e)) / expected[0].abs()
                        < 1e-8
                );
            }
        }
        eprintln!("native gray radiation quadratic={quadratic}: original scalar reference, heat power and cold nonlinear acceptance PASS");
        cooling(&mesh, quadratic, intervals).await;
    }
}
async fn cooling(mesh: &SolverMeshArtifact, quadratic: bool, intervals: &[usize]) {
    let producer = lifecycle::run(fixture::initial::linear_profile(), mesh).await;
    let field = fields::original(&producer, mesh, 0, 0);
    let binding = fixture::initial::temperature_binding(&producer.result.manifest, field, mesh);
    drop(producer);
    let mut errors = Vec::new();
    for &intervals in intervals {
        let done = lifecycle::run(
            fixture::cooling::manufactured(intervals, binding.clone()),
            mesh,
        )
        .await;
        fields::acceptance(&done, intervals);
        let mut metrics = [0.0_f64; 4];
        for (ordinal, &time) in done.result.manifest.time_points_s.iter().enumerate() {
            let field = fields::original(&done, mesh, ordinal, 0);
            for (node, &value) in mesh
                .topology
                .nodes
                .iter()
                .zip(field.storage().as_host_f64().unwrap())
            {
                metrics[0] = metrics[0].max(
                    (value
                        - fixture::reference::cooling_temperature(
                            node.coordinates_m.canonical_value()[0],
                            time,
                        ))
                    .abs(),
                );
            }
            if ordinal == 0 {
                continue;
            }
            for (index, expected) in [(1, [-10.0_f64, 0.0, 0.0]), (2, [20.0_f64, 0.0, 0.0])] {
                let field = fields::original(&done, mesh, ordinal, index);
                for values in field.storage().as_host_f64().unwrap().chunks_exact(3) {
                    metrics[index] = metrics[index].max(
                        norm(values.iter().zip(expected).map(|(v, e)| v - e)) / expected[0].abs(),
                    );
                }
            }
            let field = fields::original(&done, mesh, ordinal, 3);
            for &value in field.storage().as_host_f64().unwrap() {
                metrics[3] = metrics[3].max((value + 40.0 * time).abs() / 40.0);
            }
        }
        eprintln!("native radiation cooling quadratic={quadratic} intervals={intervals} original temperature/gradient/flux/storage errors={metrics:?}");
        errors.push(metrics);
    }
    if intervals == [8, 16, 32] {
        for q in 0..4 {
            assert!(
                errors[1][q] < errors[0][q] / 1.8 && errors[2][q] < errors[1][q] / 1.8,
                "quadratic={quadratic} q={q} errors={errors:?}"
            );
        }
    }
    assert!(
        errors.last().unwrap()[0] < 0.1,
        "original temperature error {errors:?}"
    );
}
fn norm(values: impl Iterator<Item = f64>) -> f64 {
    values.map(|v| v * v).sum::<f64>().sqrt()
}
