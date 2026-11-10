//! Public CPU route, independently checked against the manufactured cube.
//! This is not native-worker, process-restart, contact, or production evidence.
use super::run_nonlinear_with_options;
use crate::{
    contracts::ComputeBackend,
    solve::nonlinear::{solid::continuum_qualification as reference, NonlinearSolveOptions},
    topology::AnalysisProjection,
};
use runmat_meshing_core::SolverMeshArtifact;

fn options(mesh: &SolverMeshArtifact, increments: usize) -> NonlinearSolveOptions {
    NonlinearSolveOptions {
        solver_mesh: Some(mesh.clone()),
        increment_count: increments,
        max_newton_iters: 32,
        tolerance: 1e-9,
        linear_tolerance: 1e-12,
        max_line_search_backtracks: 16,
        ..NonlinearSolveOptions::default()
    }
}

fn relative(actual: &[f64], expected: &[f64]) -> f64 {
    assert_eq!(actual.len(), expected.len());
    (actual
        .iter()
        .zip(expected)
        .map(|(a, b)| (a - b).powi(2))
        .sum::<f64>()
        / expected.iter().map(|x| x * x).sum::<f64>())
    .sqrt()
}

#[test]
fn public_manufactured_cube_fields_history_and_reactions() {
    check_public_manufactured_cube(false, 1);
}

#[test]
fn public_manufactured_tet4_cube_eight_increment_fields_history_and_reactions() {
    check_public_manufactured_cube(false, 8);
}

#[test]
fn public_manufactured_tet10_cube_fields_history_and_reactions() {
    check_public_manufactured_cube(true, 1);
}

#[test]
fn public_manufactured_tet10_cube_eight_increment_fields_history_and_reactions() {
    check_public_manufactured_cube(true, 8);
}

fn check_public_manufactured_cube(quadratic: bool, increments: usize) {
    let mut errors = Vec::new();
    for cells in [1, 2, 4] {
        let (model, mesh, _) = reference::fixture::prepare(cells, quadratic);
        let result =
            run_nonlinear_with_options(&model, ComputeBackend::Cpu, options(&mesh, increments))
                .unwrap();
        let count = result.load_factors.len();
        assert!(count >= increments);
        assert_eq!(result.load_factors.last(), Some(&1.0));
        assert!(result.contact_pressure_snapshots.is_empty());
        assert!(result.contact_gap_snapshots.is_empty());
        assert!(result.thermo_mechanical_temperature_snapshots.is_empty());
        assert!(result.load_factors.windows(2).all(|p| p[1] > p[0]));
        for length in [
            result.displacement_snapshots.len(),
            result.von_mises_snapshots.len(),
            result.plastic_strain_snapshots.len(),
            result.equivalent_plastic_strain_snapshots.len(),
            result.residual_norms.len(),
            result.iteration_counts.len(),
        ] {
            assert_eq!(length, count);
        }
        assert!(result
            .residual_norms
            .iter()
            .all(|x| x.is_finite() && *x >= 0.0));
        let expected_u = mesh
            .topology
            .nodes
            .iter()
            .flat_map(|node| {
                [
                    reference::displacement(node.coordinates_m.canonical_value()[0]),
                    0.0,
                    0.0,
                ]
            })
            .collect::<Vec<_>>();
        let u = result
            .displacement_snapshots
            .last()
            .unwrap()
            .as_host_f64()
            .unwrap();
        assert_eq!(
            result.displacement_snapshots.last().unwrap().shape,
            vec![mesh.topology.nodes.len(), 3]
        );
        let projection = AnalysisProjection::derive(&mesh).unwrap();
        let mut expected_vm = Vec::new();
        let mut expected_stress = Vec::new();
        let mut expected_alpha = Vec::new();
        let mut expected_plastic = Vec::new();
        for element in &projection.elements {
            let mut volume = 0.0;
            let mut mean_stress = [0.0; 6];
            let mut mean_alpha = 0.0;
            projection
                .visit_element_quadrature(element, |point| {
                    volume += point.volume_weight_m3;
                    let stress = reference::stress(point.position_m[0]);
                    for (mean, value) in mean_stress.iter_mut().zip(stress) {
                        *mean += point.volume_weight_m3 * value;
                    }
                    mean_alpha += point.volume_weight_m3 * reference::alpha(point.position_m[0]);
                })
                .unwrap();
            mean_stress.iter_mut().for_each(|x| *x /= volume);
            expected_stress.extend(mean_stress);
            mean_alpha /= volume;
            expected_vm.push(
                (((mean_stress[0] - mean_stress[1]).powi(2)
                    + (mean_stress[1] - mean_stress[2]).powi(2)
                    + (mean_stress[2] - mean_stress[0]).powi(2))
                    / 2.0)
                    .sqrt(),
            );
            expected_alpha.push(mean_alpha);
            expected_plastic.extend([
                mean_alpha,
                -mean_alpha / 2.0,
                -mean_alpha / 2.0,
                0.0,
                0.0,
                0.0,
            ]);
        }
        let vm = result
            .von_mises_snapshots
            .last()
            .unwrap()
            .as_host_f64()
            .unwrap();
        let alpha = result
            .equivalent_plastic_strain_snapshots
            .last()
            .unwrap()
            .as_host_f64()
            .unwrap();
        let plastic = result
            .plastic_strain_snapshots
            .last()
            .unwrap()
            .as_host_f64()
            .unwrap();
        let actual_errors = [
            relative(u, &expected_u),
            relative(vm, &expected_vm),
            relative(alpha, &expected_alpha),
            relative(plastic, &expected_plastic),
            relative(
                result
                    .run
                    .field(crate::contracts::FEA_FIELD_STRUCTURAL_STRESS)
                    .unwrap()
                    .as_host_f64()
                    .unwrap(),
                &expected_stress,
            ),
        ];
        eprintln!("public nonlinear cube quadratic={quadratic} cells={cells} increments={increments} errors={actual_errors:?}");
        if quadratic && increments == 1 {
            assert!(
                actual_errors.iter().all(|x| *x < 1e-7),
                "Tet10 public cells={cells} increments={increments}: {actual_errors:?}"
            );
        } else if quadratic {
            // Several increments carry the discrete path error of the force-controlled yield
            // front crossing elements mid-step (see the continuum cube); it converges below.
            assert!(
                actual_errors.iter().all(|x| *x < 1e-4),
                "Tet10 public cells={cells} increments={increments}: {actual_errors:?}"
            );
        } else {
            assert!(
                actual_errors.iter().all(|x| *x < 0.35),
                "Tet4 public cells={cells} increments={increments}: {actual_errors:?}"
            );
        }
        let constraints = result.run.structural_constraint_result.as_ref().unwrap();
        assert!(
            (constraints.resultant_force_newtons[0] + reference::stress(0.0)[0]).abs()
                / reference::stress(0.0)[0]
                < 1e-8
        );
        assert!(constraints.resultant_force_newtons[1..]
            .iter()
            .all(|x| x.abs() < 1e-8));
        assert!(constraints.maximum_displacement_error_meters < 1e-10);
        let cold: crate::FeaRunResult =
            serde_json::from_value(serde_json::to_value(&result.run).unwrap()).unwrap();
        cold.validate_solver_formulation().unwrap();
        assert_eq!(cold.identity, result.run.identity);
        assert_eq!(
            cold.structural_constraint_result,
            result.run.structural_constraint_result
        );
        assert_eq!(cold.fields, result.run.fields);
        errors.push(actual_errors);
    }
    if quadratic && increments > 1 {
        let order = (errors[0][0] / errors[2][0]).log2() / 2.0;
        assert!(
            order >= 1.5 && errors.windows(2).all(|pair| pair[1][0] < pair[0][0]),
            "Tet10 public path error over cells 1, 2, 4: {errors:?}, order {order}"
        );
    }
    if !quadratic {
        assert!(errors[2].iter().all(|x| *x < 0.2), "Tet4 fine: {errors:?}");
        // The one-cell mesh samples the element fields at centroids where they happen to be
        // accurate (von Mises, alpha and stress errors 0.0017, 0.019, 0.020 rise on the first
        // refinement), so the rate is asserted on the asymptotic 2 -> 4 window: displacement at
        // O(h^2) (ratio > 3) and every element field at a ratio above 1.5.
        assert!(
            errors[1][0] / errors[2][0] > 3.0
                && errors[1]
                    .iter()
                    .zip(errors[2])
                    .all(|(coarse, fine)| (*coarse < 1e-10 && fine < 1e-10) || coarse / fine > 1.5),
            "Tet4 public refinement: {errors:?}"
        );
    }
}

#[test]
fn public_nonlinear_gpu_falls_back_and_rejects_missing_mesh_and_nonzero_initial_factor() {
    use crate::contracts::accelerator_admission::{
        with_accelerator_admission_policy, AcceleratorAdmissionPolicy,
        ACCELERATOR_ADMISSION_FALLBACK_CODE,
    };
    let (mut model, mesh, _) = reference::fixture::prepare(1, false);
    let cpu = run_nonlinear_with_options(&model, ComputeBackend::Cpu, options(&mesh, 1)).unwrap();
    let fallback =
        run_nonlinear_with_options(&model, ComputeBackend::Gpu, options(&mesh, 1)).unwrap();
    assert_eq!(fallback.run.requested_backend, ComputeBackend::Gpu);
    assert_eq!(fallback.run.execution_backend, crate::ExecutionBackend::Cpu);
    assert_eq!(fallback.run.fields, cpu.run.fields);
    assert_eq!(
        fallback
            .run
            .diagnostics
            .iter()
            .filter(|diagnostic| diagnostic.code == ACCELERATOR_ADMISSION_FALLBACK_CODE)
            .count(),
        1
    );
    assert!(matches!(
        with_accelerator_admission_policy(AcceleratorAdmissionPolicy::Required, || {
            run_nonlinear_with_options(&model, ComputeBackend::Gpu, options(&mesh, 1))
        }),
        Err(crate::FeaRunError::AcceleratorRequired(_))
    ));
    let mut missing = options(&mesh, 1);
    missing.solver_mesh = None;
    assert!(run_nonlinear_with_options(&model, ComputeBackend::Cpu, missing).is_err());
    model.sequence.steps[0].independent_domain =
        runmat_analysis_core::IndependentDomain::LoadFactor {
            start: 0.1,
            end: 1.0,
        };
    assert!(run_nonlinear_with_options(&model, ComputeBackend::Cpu, options(&mesh, 1)).is_err());
}

#[test]
fn public_nonlinear_does_not_ignore_an_authored_checkpoint_resume() {
    let (mut model, mesh, _) = reference::fixture::prepare(1, false);
    model.sequence.steps[0].restart = runmat_analysis_core::RestartPolicy::ResumeCheckpoint {
        checkpoint_id: "accepted-j2-prefix".into(),
        required_state_digest: "accepted-constitutive-history".into(),
    };
    runmat_analysis_core::validate_model(&model).unwrap();
    let error =
        run_nonlinear_with_options(&model, ComputeBackend::Cpu, options(&mesh, 1)).unwrap_err();
    assert!(
        matches!(error, crate::FeaRunError::InvalidModel(ref message)
        if message.contains("fresh execution cannot consume a restart request")),
        "{error}"
    );
}

#[test]
fn public_nonlinear_does_not_silently_omit_boundary_pressure() {
    let (mut model, mesh, _) = reference::fixture::prepare(1, false);
    model
        .boundaries
        .push(runmat_analysis_core::AnalysisBoundary {
            boundary_id: "additional-pressure".into(),
            target: runmat_analysis_core::RegionTarget::boundary(
                runmat_meshing_core::fixtures::fixture_face("slab_right"),
            ),
            operator: runmat_analysis_core::BoundaryOperator::StructuralPressure {
                value: runmat_analysis_core::PrescribedValue::Constant(
                    runmat_analysis_core::Pressure::new(1.0),
                ),
            },
        });
    runmat_analysis_core::validate_model(&model).unwrap();
    let error =
        run_nonlinear_with_options(&model, ComputeBackend::Cpu, options(&mesh, 1)).unwrap_err();
    assert!(
        matches!(error, crate::FeaRunError::InvalidModel(ref message)
        if message.contains("boundary operators are not realized")),
        "{error}"
    );
}

#[test]
fn public_completion_retains_original_history_and_cold_scope_validation() {
    use crate::{
        assembly::nonlinear_solid::CanonicalNonlinearSolidBasis,
        solve::nonlinear::solid::{SolidNewtonPolicy, SolidNewtonProblem},
        topology::{CanonicalSolidIntegrationPlan, CanonicalSolidIntegrationRule},
    };
    for quadratic in [false, true] {
        let (model, mesh, summary) = reference::fixture::prepare(1, quadratic);
        let mut opts = options(&mesh, 1);
        opts.max_accepted_increments = 8;
        let resource =
            super::preflight::prepare(&model, &model.sequence.steps[0], ComputeBackend::Cpu, &opts)
                .unwrap();
        let result = run_nonlinear_with_options(&model, ComputeBackend::Cpu, opts.clone()).unwrap();
        let profile = runmat_canonical_codec::profile_serialized(
            &result,
            runmat_canonical_codec::CanonicalLimits::platform_content(65_536, 32),
        )
        .unwrap();
        let demand = resource.input.structural_operator().unwrap();
        assert!(demand.publication.result_artifact_bytes.0 >= profile.encoded_bytes_upper_bound);
        let state = &result.accepted_state;
        let plan =
            CanonicalSolidIntegrationPlan::derive(&mesh, CanonicalSolidIntegrationRule::Keast15)
                .unwrap();
        assert_eq!(state.integration_identity(), plan.identity());
        assert_eq!(state.load_factor(), 1.0);
        assert_eq!(state.point_history().count(), plan.total_point_count());
        assert_eq!(
            state.displacement(),
            result
                .displacement_snapshots
                .last()
                .unwrap()
                .as_host_f64()
                .unwrap()
        );
        let encoded = serde_json::to_vec(state).unwrap();
        let cold: crate::solve::nonlinear::AcceptedSolidState =
            serde_json::from_slice(&encoded).unwrap();
        assert_eq!(&cold, state);
        let basis = CanonicalNonlinearSolidBasis::prepare(&model, &mesh, &summary).unwrap();
        let policy = SolidNewtonPolicy {
            max_iterations: opts.max_newton_iters,
            relative_residual_tolerance: opts.tolerance,
            linear_relative_tolerance: opts.linear_tolerance,
            line_search: opts.line_search,
            max_backtracks: opts.max_line_search_backtracks,
            backtrack_reduction: opts.line_search_reduction,
            max_cutbacks: opts.max_load_cutbacks,
            max_accepted_subincrements: opts.max_accepted_increments,
            minimum_load_increment: opts.minimum_load_increment,
        };
        let problem = SolidNewtonProblem::prepare(&basis, &summary, policy).unwrap();
        problem.validate_accepted(&cold).unwrap();
        let mut tampered = cold.clone();
        tampered.constitutive.points[0].equivalent_plastic_strain += 0.001;
        assert!(problem.validate_accepted(&tampered).is_err());
        if quadratic {
            // Spatially varying original alpha cannot be replaced with one
            // volume-mean field value per element without losing information.
            let alpha = state
                .point_history()
                .take(15)
                .map(|(_, _, p)| p.equivalent_plastic_strain)
                .collect::<Vec<_>>();
            assert!(alpha.iter().any(|a| (a - alpha[0]).abs() > 1e-6));
        }
    }
}
