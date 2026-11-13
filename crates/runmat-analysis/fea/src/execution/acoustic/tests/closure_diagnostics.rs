use super::*;

#[test]
fn rejected_acoustic_closure_reports_actual_metrics_without_changing_bounds() {
    let workload = workload();
    let result = workload.execute(&slab_mesh(2)).unwrap();
    let original = serde_json::to_value(result).unwrap();
    for (field, value, reason) in [
        (
            "relative_residual",
            2e-10,
            "algebraic residual exceeds tolerance",
        ),
        (
            "power_relative_error",
            2e-8,
            "power closure exceeds tolerance",
        ),
        ("mean_impedance_loss_w", -1.0, "invalid impedance loss"),
    ] {
        let mut data = original.clone();
        data[field] = serde_json::json!(value);
        let message = serde_json::from_value::<AcousticHarmonicSolverResult>(data)
            .unwrap_err()
            .to_string();
        assert!(message.contains(reason), "{message}");
        for metric in [
            "relative_residual=",
            "solver_relative_tolerance=",
            "power_relative_error=",
            "power_relative_tolerance=",
            "mean_input_power_w=",
            "mean_impedance_loss_w=",
            "refinement_steps=",
            "mesh=",
            "frequency_hz=",
        ] {
            assert!(message.contains(metric), "missing {metric}: {message}");
        }
        assert!(message.contains(&format!("{value:e}")), "{message}");
    }
}
