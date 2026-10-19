//! Original cantilever refinement gates; diagnostic output keeps their verdicts separate.
use super::*;

pub(super) fn qualify() {
    let model = fixture_model(FixtureId::CantileverLinearStatic);
    let mut errors = Vec::new();
    let mut peak_displacements = Vec::new();
    for divisions in [1, 2, 3] {
        let mesh = structured_cantilever(divisions, ElementOrder::Tet10);
        let result = crate::run_linear_static_with_options(
            &model,
            ComputeBackend::Cpu,
            LinearStaticSolveOptions {
                formulation_requirement: runmat_analysis_core::FormulationRequirement::Experimental,
                solver_mesh: Some(mesh.clone()),
                require_solver_mesh_for_solid: true,
                ..LinearStaticSolveOptions::default()
            },
        )
        .expect("the refined physical Tet10 cantilever must solve");
        let stress = result
            .field(FEA_FIELD_STRUCTURAL_INTEGRATION_POINT_STRESS)
            .expect("the solved field must retain physical integration-point stress");
        peak_displacements.push(
            result
                .field(FEA_FIELD_STRUCTURAL_DISPLACEMENT)
                .unwrap()
                .as_host_f64()
                .unwrap()
                .iter()
                .map(|value| value.abs())
                .fold(0.0_f64, f64::max),
        );
        assert_eq!(
            stress.shape,
            vec![mesh.topology.volume_elements.len() * 4, 6]
        );
        let stress_values = stress.as_host_f64().unwrap();
        let stress_peak = stress_values.iter().map(|v| v.abs()).fold(0.0, f64::max);
        let stress_range = std::array::from_fn::<_, 6, _>(|component| {
            stress_values
                .chunks_exact(6)
                .map(|v| v[component])
                .fold((f64::INFINITY, f64::NEG_INFINITY), |(low, high), v| {
                    (low.min(v), high.max(v))
                })
        });
        eprintln!("original cantilever divisions={divisions} stress_peak={stress_peak} ranges={stress_range:?}");
        errors.push(
            estimate_structural_recovery_error(
                &mesh,
                "elements",
                stress,
                StructuralRecoveryEstimatorOptions::default(),
            )
            .expect("the solved point field must drive the estimator")
            .total_error,
        );
    }

    eprintln!(
        "original cantilever recovery errors={errors:?} peak_displacements={peak_displacements:?}"
    );
    assert!(
        errors.windows(2).all(|pair| pair[1] < 0.9 * pair[0]),
        "{errors:?}"
    );
    let coarse_change = (peak_displacements[1] - peak_displacements[0]).abs();
    let fine_change = (peak_displacements[2] - peak_displacements[1]).abs();
    assert!(fine_change < 0.25 * coarse_change, "{peak_displacements:?}");
}
