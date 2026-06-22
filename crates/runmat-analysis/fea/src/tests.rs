use crate::{
    fea_electro_thermal_temperature_field_id, fea_electro_thermal_thermal_residual_field_id,
    fea_nonlinear_contact_gap_field_id, fea_nonlinear_contact_pressure_field_id,
    fea_nonlinear_equivalent_plastic_strain_field_id, fea_nonlinear_load_factor_field_id,
    fea_nonlinear_plastic_strain_field_id, fea_nonlinear_residual_norm_field_id,
    fea_nonlinear_von_mises_field_id, fea_thermal_boundary_heat_flux_field_id,
    fea_thermal_heat_flux_field_id, fea_thermal_heat_source_field_id,
    fea_thermal_temperature_gradient_field_id, fea_thermo_mechanical_coupling_residual_field_id,
    fea_thermo_mechanical_displacement_field_id, fea_thermo_mechanical_temperature_field_id,
    fea_thermo_mechanical_thermal_strain_field_id, fea_thermo_mechanical_thermal_stress_field_id,
    fea_thermo_mechanical_von_mises_field_id, fea_transient_acceleration_field_id,
    fea_transient_kinetic_energy_field_id, fea_transient_residual_norm_field_id,
    fea_transient_strain_energy_field_id, fea_transient_velocity_field_id,
    fea_transient_von_mises_field_id,
    fixtures::{fixture_model, FixtureId},
    parity::{assert_vectors_within_tolerance, ParityTolerance},
    solve::{nonlinear::NonlinearSolveOptions, transient::TransientSolveOptions},
    ComputeBackend, FeaContactInterfaceContext, FeaElectroThermalContext, FeaPrepContext,
    FeaRunResult, FeaThermoMechanicalContext, LinearStaticSolveOptions, ModalSolveOptions,
    ThermalSolveOptions, FEA_FIELD_ELECTRO_THERMAL_CURRENT_DENSITY,
    FEA_FIELD_ELECTRO_THERMAL_ELECTRIC_FIELD, FEA_FIELD_ELECTRO_THERMAL_ELECTRIC_POTENTIAL,
    FEA_FIELD_ELECTRO_THERMAL_JOULE_HEAT, FEA_FIELD_MODAL_EIGENVALUE, FEA_FIELD_MODAL_FREQUENCY_HZ,
    FEA_FIELD_MODAL_MODAL_MASS, FEA_FIELD_MODAL_MODAL_STIFFNESS, FEA_FIELD_MODAL_M_ORTHOGONALITY,
    FEA_FIELD_MODAL_PARTICIPATION_FACTOR, FEA_FIELD_MODAL_RELATIVE_FREQUENCY_SEPARATION,
    FEA_FIELD_MODAL_RESIDUAL_NORM, FEA_FIELD_STRUCTURAL_DISPLACEMENT,
    FEA_FIELD_STRUCTURAL_EQUATION_SCALE, FEA_FIELD_STRUCTURAL_REACTION_FORCE,
    FEA_FIELD_STRUCTURAL_RESIDUAL_NORM, FEA_FIELD_STRUCTURAL_STRAIN, FEA_FIELD_STRUCTURAL_STRESS,
    FEA_FIELD_STRUCTURAL_TOTAL_STRAIN_ENERGY, FEA_FIELD_STRUCTURAL_VON_MISES,
};

fn field<'a>(result: &'a FeaRunResult, field_id: &str) -> &'a runmat_analysis_core::AnalysisField {
    result.field(field_id).expect("field should be present")
}

fn host_field<'a>(result: &'a FeaRunResult, field_id: &str) -> &'a [f64] {
    field(result, field_id)
        .as_host_f64()
        .expect("field should be host-backed")
}

fn assert_thermo_mechanical_consistency_diagnostic(
    diagnostics: &[crate::diagnostics::FeaDiagnostic],
) {
    assert!(diagnostics.iter().any(|diag| {
        diag.code == "FEA_TM_CONSISTENCY"
            && diag.message.contains("constitutive_residual_ratio=")
            && diag.message.contains("thermal_strain_energy_density_mean=")
            && diag.message.contains("consistency_coverage_ratio=")
    }));
}

fn assert_thermo_mechanical_temperature_field_diagnostic(
    diagnostics: &[crate::diagnostics::FeaDiagnostic],
) {
    assert!(diagnostics.iter().any(|diag| {
        diag.code == "FEA_TM_TEMPERATURE_FIELD"
            && diag
                .message
                .contains("temperature_field_basis=nodal_coupling")
            && diag.message.contains("node_count=")
            && diag.message.contains("strain_temperature_residual_ratio=")
            && diag.message.contains("strain_temperature_coverage_ratio=")
    }));
}

#[test]
fn canonical_cantilever_benchmark_runs() {
    let model = fixture_model(FixtureId::CantileverLinearStatic);
    let result =
        crate::run_linear_static(&model, ComputeBackend::Cpu).expect("solve should succeed");

    assert_eq!(
        field(&result, FEA_FIELD_STRUCTURAL_DISPLACEMENT).element_count(),
        3
    );
    assert_eq!(
        field(&result, FEA_FIELD_STRUCTURAL_VON_MISES).element_count(),
        1
    );
    assert_eq!(
        field(&result, FEA_FIELD_STRUCTURAL_DISPLACEMENT).shape,
        vec![1, 3]
    );
    assert_eq!(
        field(&result, FEA_FIELD_STRUCTURAL_STRAIN).shape,
        vec![1, 6]
    );
    assert_eq!(
        field(&result, FEA_FIELD_STRUCTURAL_STRESS).shape,
        vec![1, 6]
    );
    assert_eq!(
        field(&result, FEA_FIELD_STRUCTURAL_REACTION_FORCE).shape,
        vec![1, 3]
    );
    let displacement = host_field(&result, FEA_FIELD_STRUCTURAL_DISPLACEMENT);
    assert!(displacement[1] < 0.0);
    assert!(displacement[1] < -8.0e-6 && displacement[1] > -1.2e-5);

    let stress = host_field(&result, FEA_FIELD_STRUCTURAL_VON_MISES);
    assert!(stress[0] > 1.0e6 && stress[0] < 2.0e6);
    assert!(host_field(&result, FEA_FIELD_STRUCTURAL_STRAIN)
        .iter()
        .any(|value| value.abs() > 0.0));
    let stress_tensor = host_field(&result, FEA_FIELD_STRUCTURAL_STRESS);
    assert!(stress_tensor.iter().any(|value| value.abs() > 0.0));
    assert!(stress_tensor[0].abs() > 0.0);
    assert!(stress_tensor[0].abs() < stress_tensor[1].abs());
    assert!(host_field(&result, FEA_FIELD_STRUCTURAL_REACTION_FORCE)
        .iter()
        .all(|value| value.is_finite()));
    assert!(host_field(&result, FEA_FIELD_STRUCTURAL_TOTAL_STRAIN_ENERGY)[0] > 0.0);
    assert!(host_field(&result, FEA_FIELD_STRUCTURAL_RESIDUAL_NORM)[0] <= 1.0e-6);
    assert!(host_field(&result, FEA_FIELD_STRUCTURAL_EQUATION_SCALE)[0] >= 1.0);
}

#[test]
fn moment_loads_require_rotational_dofs() {
    let mut model = fixture_model(FixtureId::CantileverLinearStatic);
    model.loads[0].kind = runmat_analysis_core::LoadKind::Moment {
        mx: 0.0,
        my: 0.0,
        mz: 125.0,
    };

    let err = crate::run_linear_static(&model, ComputeBackend::Cpu)
        .expect_err("moment loads should require rotational DOFs");
    let message = err.to_string();
    assert!(message.contains("moment loads require rotational-DOF structural elements"));
    assert!(message.contains("load_id=tip_load"));
}

#[test]
fn thermo_mechanical_linear_static_emits_coupled_fields() {
    let model = fixture_model(FixtureId::ThermoMechanicalKickoff);
    let result = crate::run_linear_static_with_options(
        &model,
        ComputeBackend::Cpu,
        LinearStaticSolveOptions {
            thermo_mechanical_context: Some(FeaThermoMechanicalContext {
                enabled: true,
                reference_temperature_k: 293.15,
                applied_temperature_delta_k: 65.0,
                thermal_expansion_coefficient: 1.2e-5,
                field_source: None,
                region_temperature_deltas: Vec::new(),
                time_profile: Vec::new(),
            }),
            ..LinearStaticSolveOptions::default()
        },
    )
    .expect("thermo-mechanical linear static solve should succeed");

    assert_eq!(
        field(&result, &fea_thermo_mechanical_temperature_field_id(0)).field_id,
        fea_thermo_mechanical_temperature_field_id(0)
    );
    assert_eq!(
        field(&result, &fea_thermo_mechanical_temperature_field_id(0)).shape,
        vec![field(&result, FEA_FIELD_STRUCTURAL_DISPLACEMENT).shape[0]]
    );
    assert_eq!(
        field(&result, &fea_thermo_mechanical_thermal_strain_field_id(0)).field_id,
        fea_thermo_mechanical_thermal_strain_field_id(0)
    );
    assert_eq!(
        field(&result, &fea_thermo_mechanical_thermal_stress_field_id(0)).field_id,
        fea_thermo_mechanical_thermal_stress_field_id(0)
    );
    assert_eq!(
        field(&result, &fea_thermo_mechanical_displacement_field_id(0)).field_id,
        fea_thermo_mechanical_displacement_field_id(0)
    );
    assert_eq!(
        field(&result, &fea_thermo_mechanical_von_mises_field_id(0)).field_id,
        fea_thermo_mechanical_von_mises_field_id(0)
    );
    assert_eq!(
        field(
            &result,
            &fea_thermo_mechanical_coupling_residual_field_id(0)
        )
        .field_id,
        fea_thermo_mechanical_coupling_residual_field_id(0)
    );
    assert_thermo_mechanical_consistency_diagnostic(&result.diagnostics);
    assert_thermo_mechanical_temperature_field_diagnostic(&result.diagnostics);
}

#[test]
fn convergence_diagnostics_are_emitted() {
    let model = fixture_model(FixtureId::CantileverLinearStatic);
    let result =
        crate::run_linear_static(&model, ComputeBackend::Cpu).expect("solve should succeed");
    assert!(result
        .diagnostics
        .iter()
        .any(|diag| diag.code == "FEA_CONVERGENCE"));
    assert!(result
        .diagnostics
        .iter()
        .any(|diag| diag.code == "FEA_STRUCTURAL_RESIDUAL"));
    assert!(result
        .diagnostics
        .iter()
        .any(|diag| diag.code == "FEA_STRUCTURAL_ENERGY"));
    assert!(result.diagnostics.iter().any(|diag| {
        diag.code == "FEA_STRUCTURAL_LINEAR_KNOWN_ANSWER"
            && diag.message.contains("work_energy_ratio=")
            && diag.message.contains("known_answer_coverage_ratio=")
    }));
    assert!(result.diagnostics.iter().any(|diag| {
        diag.code == "FEA_STRUCTURAL_FIELD_RECOVERY"
            && diag.message.contains("basis=operator_connectivity")
            && diag.message.contains("active_stiffness_edge_count=")
    }));
}

#[test]
fn prepared_structural_recovery_uses_prep_connectivity_edges() {
    let model = fixture_model(FixtureId::CantileverLinearStatic);
    let result = crate::run_linear_static_with_options(
        &model,
        ComputeBackend::Cpu,
        LinearStaticSolveOptions {
            prep_context: Some(FeaPrepContext {
                prepared_mesh_count: 1,
                prepared_node_count: 12,
                prepared_element_count: 18,
                mapped_region_count: 2,
                min_scaled_jacobian: 0.82,
                mean_aspect_ratio: 1.6,
                inverted_element_count: 0,
                mapped_load_count: 1,
                mapped_bc_count: 1,
                layout_seed: 17,
                topology_dof_multiplier: 1.4,
                topology_bandwidth_estimate: 3,
                mapped_region_participation_ratio: 0.8,
                topology_surface_patch_ratio: 0.25,
                topology_volume_core_ratio: 0.65,
                topology_mixed_family_ratio: 0.05,
                topology_region_span_mean: 4.0,
                topology_region_block_count: 2,
                topology_region_mesh_mean: 3.0,
                topology_region_mesh_variance: 0.4,
                topology_triangle_family_ratio: 0.2,
                topology_quad_family_ratio: 0.3,
                topology_tet_family_ratio: 0.3,
                topology_hex_family_ratio: 0.2,
                coordinate_span_x_m: 2.4,
                coordinate_span_y_m: 0.6,
                coordinate_span_z_m: 0.4,
                coordinate_active_dimension_count: 3,
                coordinate_characteristic_length_m: 0.2,
                element_geometry_node_count: 4,
                element_geometry_edge_count: 5,
                mean_element_edge_length_m: 0.2,
                mean_element_area_m2: 0.04,
                element_geometry_coverage_ratio: 1.0,
                reference_element_coordinates_m: [
                    [0.0, 0.0, 0.0],
                    [0.4, 0.0, 0.0],
                    [0.0, 0.2, 0.0],
                ],
                reference_element_area_m2: 0.04,
                element_topology_sample_element_count: 2,
                element_topology_sample_edge_count: 5,
                element_topology_sample_edge_nodes: [
                    [0, 1],
                    [1, 2],
                    [0, 2],
                    [2, 3],
                    [0, 3],
                    [0, 0],
                    [0, 0],
                    [0, 0],
                ],
                element_topology_sample_node_coordinates_m: [
                    [0.0, 0.0, 0.0],
                    [0.4, 0.0, 0.0],
                    [0.0, 0.2, 0.0],
                    [0.4, 0.2, 0.0],
                    [0.0, 0.0, 0.0],
                    [0.0, 0.0, 0.0],
                    [0.0, 0.0, 0.0],
                    [0.0, 0.0, 0.0],
                ],
                element_topology_sample_element_edges: [[0, 1, 2], [2, 3, 4], [0, 0, 0], [0, 0, 0]],
                element_topology_sample_element_orientations: [
                    [1, 1, -1],
                    [1, 1, -1],
                    [0, 0, 0],
                    [0, 0, 0],
                ],
                element_topology_sample_element_areas_m2: [0.04, 0.04, 0.0, 0.0],
                element_topology_node_coordinates_m: vec![
                    [0.0, 0.0, 0.0],
                    [0.4, 0.0, 0.0],
                    [0.0, 0.2, 0.0],
                    [0.4, 0.2, 0.0],
                ],
                element_topology_edge_nodes: vec![[0, 1], [1, 2], [0, 2], [2, 3], [0, 3]],
                element_topology_element_edges: vec![[0, 1, 2], [2, 3, 4]],
                element_topology_element_orientations: vec![[1, 1, -1], [1, 1, -1]],
                element_topology_element_areas_m2: vec![0.04, 0.04],
                calibration_profile_override: None,
            }),
            ..LinearStaticSolveOptions::default()
        },
    )
    .expect("prepared linear static solve should succeed");

    assert!(result.diagnostics.iter().any(|diag| {
        diag.code == "FEA_PREP_CONTEXT"
            && diag.message.contains("element_geometry_node_count=4")
            && diag.message.contains("element_geometry_edge_count=5")
            && diag.message.contains("mean_element_edge_length_m=0.2")
            && diag.message.contains("mean_element_area_m2=0.04")
            && diag.message.contains("element_geometry_coverage_ratio=1")
            && diag.message.contains("reference_element_area_m2=0.04")
    }));
    assert!(result.diagnostics.iter().any(|diag| {
        diag.code == "FEA_STRUCTURAL_FIELD_RECOVERY"
            && diag.message.contains("basis=prep_constant_strain_b_matrix")
            && diag.message.contains("prep_recovery_edge_count=")
            && diag.message.contains("recovery_element_count=2")
            && diag.message.contains("mean_edge_length_m=")
            && diag.message.contains("max_edge_strain_norm=")
            && diag.message.contains("strain_component_coverage_ratio=")
            && diag.message.contains("element_geometry_node_count=4")
            && diag.message.contains("element_geometry_edge_count=5")
            && diag.message.contains("element_geometry_coverage_ratio=1")
    }));
}

#[test]
fn deterministic_replay_for_fixture_is_stable() {
    let model = fixture_model(FixtureId::CantileverLinearStatic);
    let first =
        crate::run_linear_static(&model, ComputeBackend::Cpu).expect("first run should succeed");
    let second =
        crate::run_linear_static(&model, ComputeBackend::Cpu).expect("second run should succeed");

    assert_eq!(first.fields, second.fields);
    assert_eq!(first.diagnostics, second.diagnostics);
}

#[test]
fn cpu_gpu_parity_respects_tolerance_policy() {
    let model = fixture_model(FixtureId::CantileverLinearStatic);
    let cpu =
        crate::run_linear_static(&model, ComputeBackend::Cpu).expect("cpu run should succeed");
    let gpu =
        crate::run_linear_static(&model, ComputeBackend::Gpu).expect("gpu run should succeed");

    let tol = ParityTolerance::strict();
    let cpu_displacement = host_field(&cpu, FEA_FIELD_STRUCTURAL_DISPLACEMENT);
    let gpu_displacement = host_field(&gpu, FEA_FIELD_STRUCTURAL_DISPLACEMENT);
    assert_vectors_within_tolerance(cpu_displacement, gpu_displacement, tol);

    let cpu_stress = host_field(&cpu, FEA_FIELD_STRUCTURAL_VON_MISES);
    let gpu_stress = host_field(&gpu, FEA_FIELD_STRUCTURAL_VON_MISES);
    assert_vectors_within_tolerance(cpu_stress, gpu_stress, tol);

    let cpu_tensor_stress = host_field(&cpu, FEA_FIELD_STRUCTURAL_STRESS);
    let gpu_tensor_stress = host_field(&gpu, FEA_FIELD_STRUCTURAL_STRESS);
    assert_vectors_within_tolerance(cpu_tensor_stress, gpu_tensor_stress, tol);

    let cpu_reactions = host_field(&cpu, FEA_FIELD_STRUCTURAL_REACTION_FORCE);
    let gpu_reactions = host_field(&gpu, FEA_FIELD_STRUCTURAL_REACTION_FORCE);
    assert_vectors_within_tolerance(cpu_reactions, gpu_reactions, tol);

    let cpu_residual = host_field(&cpu, FEA_FIELD_STRUCTURAL_RESIDUAL_NORM);
    let gpu_residual = host_field(&gpu, FEA_FIELD_STRUCTURAL_RESIDUAL_NORM);
    assert_vectors_within_tolerance(cpu_residual, gpu_residual, tol);
}

#[test]
fn fixture_missing_materials_is_rejected() {
    let model = fixture_model(FixtureId::MissingMaterials);
    let err = crate::run_linear_static(&model, ComputeBackend::Cpu)
        .expect_err("fixture should fail validation");
    assert!(err
        .to_string()
        .contains("ANALYSIS_VALIDATION_MISSING_MATERIALS"));
}

#[test]
fn fixture_missing_loads_is_rejected() {
    let model = fixture_model(FixtureId::MissingLoads);
    let err = crate::run_linear_static(&model, ComputeBackend::Cpu)
        .expect_err("fixture should fail validation");
    assert!(err
        .to_string()
        .contains("ANALYSIS_VALIDATION_MISSING_LOADS"));
}

#[test]
fn modal_solver_emits_modes_for_modal_step_fixture() {
    let mut model = fixture_model(FixtureId::CantileverLinearStatic);
    model.steps = vec![runmat_analysis_core::AnalysisStep {
        step_id: "modal_1".to_string(),
        kind: runmat_analysis_core::AnalysisStepKind::Modal,
    }];
    let result = crate::run_modal(&model, ComputeBackend::Cpu).expect("modal solve should succeed");

    assert!(!result.eigenvalues_hz.is_empty());
    assert_eq!(result.eigenvalues_hz.len(), result.mode_shapes.len());
    assert_eq!(result.mode_shapes[0].shape.len(), 2);
    assert_eq!(result.mode_shapes[0].shape[1], 3);
    assert_eq!(
        result.mode_shapes[0].element_count(),
        result.mode_shapes[0].shape[0] * 3
    );
    assert_eq!(
        field(&result.run, FEA_FIELD_STRUCTURAL_DISPLACEMENT).shape,
        result.mode_shapes[0].shape
    );
    assert_eq!(
        field(&result.run, FEA_FIELD_MODAL_FREQUENCY_HZ).element_count(),
        result.eigenvalues_hz.len()
    );
    assert_eq!(
        field(&result.run, FEA_FIELD_MODAL_EIGENVALUE).element_count(),
        result.eigenvalues_hz.len()
    );
    assert_eq!(
        field(&result.run, FEA_FIELD_MODAL_MODAL_MASS).element_count(),
        result.eigenvalues_hz.len()
    );
    assert_eq!(
        field(&result.run, FEA_FIELD_MODAL_MODAL_STIFFNESS).element_count(),
        result.eigenvalues_hz.len()
    );
    assert_eq!(
        field(&result.run, FEA_FIELD_MODAL_PARTICIPATION_FACTOR).element_count(),
        result.eigenvalues_hz.len()
    );
    assert_eq!(
        field(&result.run, FEA_FIELD_MODAL_RESIDUAL_NORM).element_count(),
        result.eigenvalues_hz.len()
    );
    assert_eq!(
        field(&result.run, FEA_FIELD_MODAL_RELATIVE_FREQUENCY_SEPARATION).element_count(),
        result.eigenvalues_hz.len()
    );
    let orthogonality = field(&result.run, FEA_FIELD_MODAL_M_ORTHOGONALITY);
    assert_eq!(
        orthogonality.element_count(),
        result.eigenvalues_hz.len() * result.eigenvalues_hz.len()
    );
    assert_eq!(
        orthogonality.shape,
        vec![result.eigenvalues_hz.len(), result.eigenvalues_hz.len()]
    );
    let residual_norms = host_field(&result.run, FEA_FIELD_MODAL_RESIDUAL_NORM);
    assert_eq!(residual_norms, result.residual_norms.as_slice());
    assert!(result
        .run
        .diagnostics
        .iter()
        .any(|diag| diag.code == "FEA_MODAL_CONVERGENCE"));
}

#[test]
fn transient_solver_emits_time_snapshots_for_transient_step_fixture() {
    let mut model = fixture_model(FixtureId::CantileverLinearStatic);
    model.steps = vec![runmat_analysis_core::AnalysisStep {
        step_id: "transient_1".to_string(),
        kind: runmat_analysis_core::AnalysisStepKind::Transient,
    }];
    let result =
        crate::run_transient(&model, ComputeBackend::Cpu).expect("transient solve should succeed");

    assert!(!result.time_points_s.is_empty());
    assert_eq!(
        field(&result.run, FEA_FIELD_STRUCTURAL_DISPLACEMENT).shape,
        result.displacement_snapshots[0].shape
    );
    assert_eq!(
        result.time_points_s.len(),
        result.displacement_snapshots.len()
    );
    assert_eq!(result.time_points_s.len(), result.velocity_snapshots.len());
    assert_eq!(
        result.time_points_s.len(),
        result.acceleration_snapshots.len()
    );
    assert_eq!(result.time_points_s.len(), result.von_mises_snapshots.len());
    assert_eq!(
        result.time_points_s.len(),
        result.kinetic_energy_snapshots.len()
    );
    assert_eq!(
        result.time_points_s.len(),
        result.strain_energy_snapshots.len()
    );
    assert_eq!(
        result.time_points_s.len(),
        result.residual_norm_snapshots.len()
    );
    assert_eq!(
        result.velocity_snapshots[1].field_id,
        fea_transient_velocity_field_id(1)
    );
    assert_eq!(
        result.acceleration_snapshots[1].field_id,
        fea_transient_acceleration_field_id(1)
    );
    assert_eq!(
        result.von_mises_snapshots[1].field_id,
        fea_transient_von_mises_field_id(1)
    );
    assert_eq!(
        result.kinetic_energy_snapshots[1].field_id,
        fea_transient_kinetic_energy_field_id(1)
    );
    assert_eq!(
        result.strain_energy_snapshots[1].field_id,
        fea_transient_strain_energy_field_id(1)
    );
    assert_eq!(
        result.residual_norm_snapshots[1].field_id,
        fea_transient_residual_norm_field_id(1)
    );
    assert_eq!(
        result.residual_norm_snapshots[1]
            .as_host_f64()
            .expect("residual field should be host-backed")[0],
        result.residual_norms[0]
    );
    assert!(!result.residual_norms.is_empty());
    assert!(result
        .run
        .diagnostics
        .iter()
        .any(|diag| diag.code == "FEA_TRANSIENT_CONVERGENCE"));
    let energy_diag = result
        .run
        .diagnostics
        .iter()
        .find(|diag| diag.code == "FEA_TRANSIENT_ENERGY_BALANCE")
        .expect("transient energy balance diagnostic should be emitted");
    assert!(energy_diag.message.contains("max_total_energy="));
    assert!(energy_diag.message.contains("energy_growth_ratio="));
}

#[test]
fn thermal_solver_emits_heat_transfer_fields() {
    let model = fixture_model(FixtureId::CantileverLinearStatic);
    let result = crate::run_thermal_with_options(
        &model,
        ComputeBackend::Cpu,
        ThermalSolveOptions {
            step_count: 4,
            thermo_mechanical_context: Some(FeaThermoMechanicalContext {
                enabled: true,
                reference_temperature_k: 293.15,
                applied_temperature_delta_k: 45.0,
                thermal_expansion_coefficient: 1.2e-5,
                field_source: None,
                region_temperature_deltas: Vec::new(),
                time_profile: Vec::new(),
            }),
            ..ThermalSolveOptions::default()
        },
    )
    .expect("thermal solve should succeed");

    assert_eq!(result.time_points_s.len(), 4);
    assert_eq!(
        result.time_points_s.len(),
        result.temperature_snapshots.len()
    );
    assert_eq!(
        result.time_points_s.len(),
        result.temperature_gradient_snapshots.len()
    );
    assert_eq!(result.time_points_s.len(), result.heat_flux_snapshots.len());
    assert_eq!(
        result.time_points_s.len(),
        result.heat_source_snapshots.len()
    );
    assert_eq!(
        result.time_points_s.len(),
        result.boundary_heat_flux_snapshots.len()
    );
    assert_eq!(
        result.temperature_gradient_snapshots[1].field_id,
        fea_thermal_temperature_gradient_field_id(1)
    );
    assert_eq!(
        result.heat_flux_snapshots[1].field_id,
        fea_thermal_heat_flux_field_id(1)
    );
    assert_eq!(
        result.heat_source_snapshots[1].field_id,
        fea_thermal_heat_source_field_id(1)
    );
    assert_eq!(result.temperature_gradient_snapshots[1].shape.len(), 2);
    assert_eq!(result.temperature_gradient_snapshots[1].shape[1], 3);
    assert_eq!(
        result.heat_flux_snapshots[1].shape, result.temperature_gradient_snapshots[1].shape,
        "thermal heat flux should share the element-vector shape"
    );
    assert_eq!(
        result.heat_source_snapshots[1].shape,
        vec![result.temperature_gradient_snapshots[1].shape[0]],
        "thermal heat source should be an element scalar field"
    );
    assert_eq!(
        result.boundary_heat_flux_snapshots[1].field_id,
        fea_thermal_boundary_heat_flux_field_id(1)
    );
    assert_eq!(
        result.boundary_heat_flux_snapshots[1].shape,
        vec![6],
        "thermal boundary heat flux should report six domain faces"
    );
    assert!(result
        .run
        .diagnostics
        .iter()
        .any(|diag| diag.code == "FEA_THERMAL_FIELD_RECOVERY"
            && diag.message.contains("recovery_dimensions=")
            && diag.message.contains("boundary_face_count=6")));
}

#[test]
fn prepared_thermal_recovery_uses_prep_element_topology() {
    let model = fixture_model(FixtureId::CantileverLinearStatic);
    let result = crate::run_thermal_with_options(
        &model,
        ComputeBackend::Cpu,
        ThermalSolveOptions {
            step_count: 4,
            prep_context: Some(FeaPrepContext {
                prepared_mesh_count: 1,
                prepared_node_count: 14,
                prepared_element_count: 24,
                mapped_region_count: 2,
                min_scaled_jacobian: 0.86,
                mean_aspect_ratio: 1.4,
                inverted_element_count: 0,
                mapped_load_count: 1,
                mapped_bc_count: 1,
                layout_seed: 23,
                topology_dof_multiplier: 1.5,
                topology_bandwidth_estimate: 3,
                mapped_region_participation_ratio: 0.85,
                topology_surface_patch_ratio: 0.2,
                topology_volume_core_ratio: 0.7,
                topology_mixed_family_ratio: 0.05,
                topology_region_span_mean: 4.0,
                topology_region_block_count: 2,
                topology_region_mesh_mean: 3.0,
                topology_region_mesh_variance: 0.4,
                topology_triangle_family_ratio: 0.2,
                topology_quad_family_ratio: 0.25,
                topology_tet_family_ratio: 0.35,
                topology_hex_family_ratio: 0.2,
                coordinate_span_x_m: 2.0,
                coordinate_span_y_m: 0.5,
                coordinate_span_z_m: 0.4,
                coordinate_active_dimension_count: 3,
                coordinate_characteristic_length_m: 0.2,
                element_geometry_node_count: 4,
                element_geometry_edge_count: 5,
                mean_element_edge_length_m: 0.2,
                mean_element_area_m2: 0.04,
                element_geometry_coverage_ratio: 1.0,
                reference_element_coordinates_m: [
                    [0.0, 0.0, 0.0],
                    [0.4, 0.0, 0.0],
                    [0.0, 0.2, 0.0],
                ],
                reference_element_area_m2: 0.04,
                element_topology_sample_element_count: 2,
                element_topology_sample_edge_count: 5,
                element_topology_sample_edge_nodes: [
                    [0, 1],
                    [1, 2],
                    [0, 2],
                    [2, 3],
                    [0, 3],
                    [0, 0],
                    [0, 0],
                    [0, 0],
                ],
                element_topology_sample_node_coordinates_m: [
                    [0.0, 0.0, 0.0],
                    [0.4, 0.0, 0.0],
                    [0.0, 0.2, 0.0],
                    [0.4, 0.2, 0.0],
                    [0.0, 0.0, 0.0],
                    [0.0, 0.0, 0.0],
                    [0.0, 0.0, 0.0],
                    [0.0, 0.0, 0.0],
                ],
                element_topology_sample_element_edges: [[0, 1, 2], [2, 3, 4], [0, 0, 0], [0, 0, 0]],
                element_topology_sample_element_orientations: [
                    [1, 1, -1],
                    [1, 1, -1],
                    [0, 0, 0],
                    [0, 0, 0],
                ],
                element_topology_sample_element_areas_m2: [0.04, 0.04, 0.0, 0.0],
                element_topology_node_coordinates_m: vec![
                    [0.0, 0.0, 0.0],
                    [0.4, 0.0, 0.0],
                    [0.0, 0.2, 0.0],
                    [0.4, 0.2, 0.0],
                ],
                element_topology_edge_nodes: vec![[0, 1], [1, 2], [0, 2], [2, 3], [0, 3]],
                element_topology_element_edges: vec![[0, 1, 2], [2, 3, 4]],
                element_topology_element_orientations: vec![[1, 1, -1], [1, 1, -1]],
                element_topology_element_areas_m2: vec![0.04, 0.04],
                calibration_profile_override: None,
            }),
            thermo_mechanical_context: Some(FeaThermoMechanicalContext {
                enabled: true,
                reference_temperature_k: 293.15,
                applied_temperature_delta_k: 50.0,
                thermal_expansion_coefficient: 1.2e-5,
                field_source: None,
                region_temperature_deltas: Vec::new(),
                time_profile: Vec::new(),
            }),
            ..ThermalSolveOptions::default()
        },
    )
    .expect("prepared thermal solve should succeed");

    let recovery = result
        .run
        .diagnostics
        .iter()
        .find(|diag| diag.code == "FEA_THERMAL_FIELD_RECOVERY")
        .expect("thermal recovery diagnostic should be emitted");
    assert!(recovery
        .message
        .contains("basis=prep_element_triangle_topology"));
    assert!(recovery.message.contains("prep_recovery_edge_count="));
    assert!(recovery.message.contains("prep_triangle_element_count=2"));
    assert_eq!(result.temperature_gradient_snapshots[1].shape[0], 2);
    assert_eq!(result.temperature_gradient_snapshots[1].shape[1], 3);
    assert_eq!(
        result.heat_flux_snapshots[1].shape,
        result.temperature_gradient_snapshots[1].shape
    );
    assert_eq!(result.boundary_heat_flux_snapshots[1].shape, vec![6]);
}

#[test]
fn modal_large_fixture_emits_orthogonality_and_separation_diagnostics() {
    let model = fixture_model(FixtureId::ModalLarge);
    let result = crate::run_modal_with_options(
        &model,
        ComputeBackend::Cpu,
        ModalSolveOptions {
            mode_count: 8,
            prep_context: None,
            thermo_mechanical_context: None,
            electro_thermal_context: None,
        },
    )
    .expect("modal large fixture should solve");

    assert!(!result.eigenvalues_hz.is_empty());
    assert_eq!(result.eigenvalues_hz.len(), result.mode_shapes.len());
    assert!(result
        .run
        .diagnostics
        .iter()
        .any(|diag| diag.code == "FEA_MODAL_ORTHOGONALITY"));
    assert!(result
        .run
        .diagnostics
        .iter()
        .any(|diag| diag.code == "FEA_MODAL_SEPARATION"));
    assert!(result.run.diagnostics.iter().any(|diag| {
        diag.code == "FEA_MODAL_CLUSTER"
            && diag.message.contains("near_repeated_mode_pair_count=")
            && diag.message.contains("cluster_coverage_ratio=")
    }));
}

#[test]
fn transient_long_fixture_emits_stability_diagnostics() {
    let model = fixture_model(FixtureId::TransientLong);
    let result = crate::run_transient_with_options(
        &model,
        ComputeBackend::Cpu,
        TransientSolveOptions {
            step_count: 24,
            ..TransientSolveOptions::default()
        },
    )
    .expect("transient long fixture should solve");

    assert!(result.time_points_s.len() > 8);
    assert!(result
        .run
        .diagnostics
        .iter()
        .any(|diag| diag.code == "FEA_TRANSIENT_STABILITY"));
    assert!(result
        .run
        .diagnostics
        .iter()
        .any(|diag| diag.code == "FEA_TRANSIENT_ENERGY"));
}

#[test]
fn transient_shock_fixture_emits_adaptivity_and_physics_diagnostics() {
    let model = fixture_model(FixtureId::TransientShock);
    let result = crate::run_transient_with_options(
        &model,
        ComputeBackend::Cpu,
        TransientSolveOptions {
            step_count: 48,
            ..TransientSolveOptions::default()
        },
    )
    .expect("transient shock fixture should solve");

    assert!(result.time_points_s.len() > 24);
    assert!(result
        .run
        .diagnostics
        .iter()
        .any(|diag| diag.code == "FEA_TRANSIENT_ADAPTIVITY"));
    assert!(result
        .run
        .diagnostics
        .iter()
        .any(|diag| diag.code == "FEA_TRANSIENT_PHYSICS"));
}

#[test]
fn thermo_mechanical_transient_emits_coupled_solve_profile_diagnostic() {
    let model = fixture_model(FixtureId::ThermoMechanicalKickoff);
    let result = crate::run_transient_with_options(
        &model,
        ComputeBackend::Cpu,
        TransientSolveOptions {
            step_count: 24,
            thermo_mechanical_context: Some(FeaThermoMechanicalContext {
                enabled: true,
                reference_temperature_k: 293.15,
                applied_temperature_delta_k: 65.0,
                thermal_expansion_coefficient: 1.2e-5,
                field_source: None,
                region_temperature_deltas: Vec::new(),
                time_profile: Vec::new(),
            }),
            ..TransientSolveOptions::default()
        },
    )
    .expect("thermo-mechanical transient solve should succeed");

    assert_eq!(
        result.time_points_s.len(),
        result.thermo_mechanical_temperature_snapshots.len()
    );
    assert_eq!(
        result.time_points_s.len(),
        result.thermo_mechanical_thermal_strain_snapshots.len()
    );
    assert_eq!(
        result.time_points_s.len(),
        result.thermo_mechanical_thermal_stress_snapshots.len()
    );
    assert_eq!(
        result.time_points_s.len(),
        result.thermo_mechanical_displacement_snapshots.len()
    );
    assert_eq!(
        result.time_points_s.len(),
        result.thermo_mechanical_von_mises_snapshots.len()
    );
    assert_eq!(
        result.time_points_s.len(),
        result.thermo_mechanical_coupling_residual_snapshots.len()
    );
    assert_eq!(
        result.thermo_mechanical_temperature_snapshots[0].field_id,
        fea_thermo_mechanical_temperature_field_id(0)
    );
    assert_eq!(
        result.thermo_mechanical_temperature_snapshots[0].shape,
        vec![result.displacement_snapshots[0].shape[0]]
    );
    assert_eq!(
        result.thermo_mechanical_thermal_strain_snapshots[0].field_id,
        fea_thermo_mechanical_thermal_strain_field_id(0)
    );
    assert_eq!(
        result.thermo_mechanical_thermal_stress_snapshots[0].field_id,
        fea_thermo_mechanical_thermal_stress_field_id(0)
    );
    assert_eq!(
        result.thermo_mechanical_displacement_snapshots[0].field_id,
        fea_thermo_mechanical_displacement_field_id(0)
    );
    assert_eq!(
        result.thermo_mechanical_von_mises_snapshots[0].field_id,
        fea_thermo_mechanical_von_mises_field_id(0)
    );
    assert_eq!(
        result.thermo_mechanical_coupling_residual_snapshots[0].field_id,
        fea_thermo_mechanical_coupling_residual_field_id(0)
    );
    assert_thermo_mechanical_consistency_diagnostic(&result.run.diagnostics);
    assert_thermo_mechanical_temperature_field_diagnostic(&result.run.diagnostics);

    assert!(result
        .run
        .diagnostics
        .iter()
        .any(|diag| diag.code == "FEA_TM_COUPLING"));
    let coupling = result
        .run
        .diagnostics
        .iter()
        .find(|diag| diag.code == "FEA_TM_COUPLING")
        .expect("thermo coupling diagnostic should be present");
    assert!(coupling.message.contains("effective_modulus_scale="));
    assert!(coupling
        .message
        .contains("constitutive_material_spread_ratio="));
    assert!(coupling.message.contains("assignment_heterogeneity_index="));
    let profile = result
        .run
        .diagnostics
        .iter()
        .find(|diag| diag.code == "FEA_TM_TRANSIENT")
        .expect("thermo transient profile diagnostic should be present");
    assert!(profile.message.contains("effective_residual_target_peak="));
    assert!(profile.message.contains("growth_limit_min="));
}

#[test]
fn electro_thermal_transient_emits_coupled_fields() {
    let model = fixture_model(FixtureId::ElectroThermalJouleBenign);
    let result = crate::run_transient_with_options(
        &model,
        ComputeBackend::Cpu,
        TransientSolveOptions {
            step_count: 12,
            electro_thermal_context: Some(FeaElectroThermalContext {
                enabled: true,
                reference_temperature_k: 293.15,
                applied_voltage_v: 36.0,
                base_electrical_conductivity_s_per_m: 3.5e7,
                resistive_heating_coefficient: 4.0e-4,
                region_conductivity_scales: Vec::new(),
                time_profile: Vec::new(),
            }),
            ..TransientSolveOptions::default()
        },
    )
    .expect("electro-thermal transient solve should succeed");

    assert!(result
        .run
        .field(FEA_FIELD_ELECTRO_THERMAL_ELECTRIC_POTENTIAL)
        .is_some());
    assert!(result
        .run
        .field(FEA_FIELD_ELECTRO_THERMAL_ELECTRIC_FIELD)
        .is_some());
    assert!(result
        .run
        .field(FEA_FIELD_ELECTRO_THERMAL_CURRENT_DENSITY)
        .is_some());
    assert!(result
        .run
        .field(FEA_FIELD_ELECTRO_THERMAL_JOULE_HEAT)
        .is_some());
    let electric_field = field(&result.run, FEA_FIELD_ELECTRO_THERMAL_ELECTRIC_FIELD);
    let current_density = field(&result.run, FEA_FIELD_ELECTRO_THERMAL_CURRENT_DENSITY);
    let joule_heat = field(&result.run, FEA_FIELD_ELECTRO_THERMAL_JOULE_HEAT);
    assert_eq!(electric_field.shape.len(), 2);
    assert_eq!(electric_field.shape[1], 3);
    assert_eq!(current_density.shape, electric_field.shape);
    assert_eq!(joule_heat.shape, vec![electric_field.shape[0]]);
    assert_eq!(
        result.electro_thermal_temperature_snapshots.len(),
        result.time_points_s.len()
    );
    assert_eq!(
        result.electro_thermal_thermal_residual_snapshots.len(),
        result.time_points_s.len()
    );
    assert_eq!(
        result.electro_thermal_temperature_snapshots[0].field_id,
        fea_electro_thermal_temperature_field_id(0)
    );
    assert_eq!(
        result.electro_thermal_thermal_residual_snapshots[0].field_id,
        fea_electro_thermal_thermal_residual_field_id(0)
    );
    let coupling = result
        .run
        .diagnostics
        .iter()
        .find(|diag| diag.code == "FEA_ET_COUPLING")
        .expect("electro-thermal coupling diagnostic should be present");
    assert!(coupling.message.contains("electrical_power_in_w="));
    assert!(coupling.message.contains("integrated_joule_heat_w="));
    assert!(coupling.message.contains("power_balance_ratio="));
    assert!(coupling.message.contains("conservation_residual="));
    let conduction_conservation = result
        .run
        .diagnostics
        .iter()
        .find(|diag| diag.code == "FEA_ET_CONDUCTION_CONSERVATION")
        .expect("electro-thermal conduction conservation diagnostic should be present");
    assert!(conduction_conservation
        .message
        .contains("ohms_law_residual_ratio="));
    assert!(conduction_conservation
        .message
        .contains("joule_heat_balance_ratio="));
    assert!(conduction_conservation
        .message
        .contains("conduction_graph_coverage_ratio="));
}

#[test]
fn nonlinear_fixture_emits_incremental_payload_and_diagnostics() {
    let mut model = fixture_model(FixtureId::TransientShock);
    model.steps = vec![runmat_analysis_core::AnalysisStep {
        step_id: "nonlinear_1".to_string(),
        kind: runmat_analysis_core::AnalysisStepKind::Nonlinear,
    }];
    let result =
        crate::run_nonlinear(&model, ComputeBackend::Cpu).expect("nonlinear solve should succeed");

    assert!(!result.load_factors.is_empty());
    assert_eq!(result.load_factors.len(), result.residual_norms.len());
    assert_eq!(result.residual_norms.len(), result.increment_norms.len());
    assert_eq!(result.residual_norms.len(), result.iteration_counts.len());
    assert_eq!(result.load_factors.len(), result.von_mises_snapshots.len());
    assert_eq!(
        result.load_factors.len(),
        result.plastic_strain_snapshots.len()
    );
    assert_eq!(
        result.load_factors.len(),
        result.equivalent_plastic_strain_snapshots.len()
    );
    assert_eq!(
        result.load_factors.len(),
        result.contact_pressure_snapshots.len()
    );
    assert_eq!(
        result.load_factors.len(),
        result.contact_gap_snapshots.len()
    );
    assert_eq!(
        result.load_factors.len(),
        result.load_factor_snapshots.len()
    );
    assert_eq!(
        result.load_factors.len(),
        result.residual_norm_snapshots.len()
    );
    assert_eq!(
        result.von_mises_snapshots[0].field_id,
        fea_nonlinear_von_mises_field_id(0)
    );
    assert_eq!(
        result.plastic_strain_snapshots[0].field_id,
        fea_nonlinear_plastic_strain_field_id(0)
    );
    assert_eq!(
        result.equivalent_plastic_strain_snapshots[0].field_id,
        fea_nonlinear_equivalent_plastic_strain_field_id(0)
    );
    assert_eq!(
        result.contact_pressure_snapshots[0].field_id,
        fea_nonlinear_contact_pressure_field_id(0)
    );
    assert_eq!(
        result.contact_gap_snapshots[0].field_id,
        fea_nonlinear_contact_gap_field_id(0)
    );
    assert_eq!(
        result.load_factor_snapshots[0].field_id,
        fea_nonlinear_load_factor_field_id(0)
    );
    assert_eq!(
        result.residual_norm_snapshots[0].field_id,
        fea_nonlinear_residual_norm_field_id(0)
    );
    assert_eq!(
        result.load_factor_snapshots[0]
            .as_host_f64()
            .expect("load factor field should be host-backed")[0],
        result.load_factors[0]
    );
    assert_eq!(
        result.residual_norm_snapshots[0]
            .as_host_f64()
            .expect("residual field should be host-backed")[0],
        result.residual_norms[0]
    );
    assert!(result
        .run
        .diagnostics
        .iter()
        .any(|diag| diag.code == "FEA_NONLINEAR_CONVERGENCE"));
    assert!(result
        .run
        .diagnostics
        .iter()
        .any(|diag| diag.code == "FEA_NONLINEAR_COST"));
    let state = result
        .run
        .diagnostics
        .iter()
        .find(|diag| diag.code == "FEA_NONLINEAR_STATE")
        .expect("nonlinear state diagnostic should be present");
    assert!(state.message.contains("plastic_active_element_count="));
    assert!(state.message.contains("max_equivalent_plastic_strain="));
    assert!(state.message.contains("contact_active_entity_count="));
    assert!(state.message.contains("max_contact_pressure="));
    let topology = result
        .run
        .diagnostics
        .iter()
        .find(|diag| diag.code == "FEA_NONLINEAR_STATE_TOPOLOGY")
        .expect("nonlinear state topology diagnostic should be present");
    assert!(topology.message.contains("basis=dof_adjacency_fallback"));
    assert!(topology.message.contains("active_recovery_edge_count="));
    let convergence = result
        .run
        .diagnostics
        .iter()
        .find(|diag| diag.code == "FEA_NONLINEAR_CONVERGENCE")
        .expect("nonlinear convergence diagnostic should be present");
    assert!(convergence.message.contains("iteration_spike_count="));
    assert!(convergence.message.contains("convergence_stall_count="));
    assert!(convergence.message.contains("backtrack_burst_count="));
}

#[test]
fn nonlinear_contact_reference_enforces_pressure_gap_complementarity() {
    let options = NonlinearSolveOptions {
        contact_context: Some(FeaContactInterfaceContext {
            enabled: true,
            penalty_stiffness_scale: 2.0,
            max_penetration_ratio: 0.01,
            friction_coefficient: 0.0,
        }),
        ..NonlinearSolveOptions::default()
    };
    let model = fixture_model(FixtureId::NonlinearContactFrictionlessReference);
    let result = crate::run_nonlinear_with_options(&model, ComputeBackend::Cpu, options)
        .expect("frictionless contact reference solve should succeed");

    let final_pressure = result
        .contact_pressure_snapshots
        .last()
        .expect("contact pressure snapshot should be present")
        .as_host_f64()
        .expect("contact pressure should be host-backed");
    let final_gap = result
        .contact_gap_snapshots
        .last()
        .expect("contact gap snapshot should be present")
        .as_host_f64()
        .expect("contact gap should be host-backed");
    assert!(
        final_pressure.iter().any(|pressure| *pressure > 0.0),
        "reference contact should close and develop pressure"
    );
    assert!(
        final_gap.iter().all(|gap| *gap >= 0.0),
        "projected contact gap should be nonpenetrating"
    );
    for (pressure, gap) in final_pressure.iter().zip(final_gap.iter()) {
        assert!(
            *pressure == 0.0 || *gap <= 1.0e-12,
            "active contact pressure must only occur on closed gaps"
        );
    }

    let known_answer = result
        .run
        .diagnostics
        .iter()
        .find(|diag| diag.code == "FEA_CONTACT_KNOWN_ANSWER")
        .expect("contact known-answer diagnostic should be present");
    assert!(known_answer
        .message
        .contains("open_gap_pressure_residual=0"));
    assert!(known_answer
        .message
        .contains("pressure_gap_complementarity_residual=0"));
    assert!(known_answer
        .message
        .contains("closed_entity_coverage_ratio=1"));
}

#[test]
fn nonlinear_prepared_state_recovery_uses_prep_connectivity_edges() {
    let model = fixture_model(FixtureId::NonlinearLoadPathMix);
    let result = crate::run_nonlinear_with_options(
        &model,
        ComputeBackend::Cpu,
        NonlinearSolveOptions {
            prep_context: Some(FeaPrepContext {
                prepared_mesh_count: 1,
                prepared_node_count: 14,
                prepared_element_count: 22,
                mapped_region_count: 3,
                min_scaled_jacobian: 0.84,
                mean_aspect_ratio: 1.7,
                inverted_element_count: 0,
                mapped_load_count: 2,
                mapped_bc_count: 1,
                layout_seed: 29,
                topology_dof_multiplier: 1.6,
                topology_bandwidth_estimate: 4,
                mapped_region_participation_ratio: 0.85,
                topology_surface_patch_ratio: 0.2,
                topology_volume_core_ratio: 0.7,
                topology_mixed_family_ratio: 0.08,
                topology_region_span_mean: 5.0,
                topology_region_block_count: 3,
                topology_region_mesh_mean: 3.5,
                topology_region_mesh_variance: 0.5,
                topology_triangle_family_ratio: 0.2,
                topology_quad_family_ratio: 0.25,
                topology_tet_family_ratio: 0.35,
                topology_hex_family_ratio: 0.2,
                coordinate_span_x_m: 3.0,
                coordinate_span_y_m: 0.8,
                coordinate_span_z_m: 0.5,
                coordinate_active_dimension_count: 3,
                coordinate_characteristic_length_m: 0.25,
                element_geometry_node_count: 4,
                element_geometry_edge_count: 5,
                mean_element_edge_length_m: 0.25,
                mean_element_area_m2: 0.0625,
                element_geometry_coverage_ratio: 1.0,
                reference_element_coordinates_m: [
                    [0.0, 0.0, 0.0],
                    [0.5, 0.0, 0.0],
                    [0.0, 0.25, 0.0],
                ],
                reference_element_area_m2: 0.0625,
                element_topology_sample_element_count: 2,
                element_topology_sample_edge_count: 5,
                element_topology_sample_edge_nodes: [
                    [0, 1],
                    [1, 2],
                    [0, 2],
                    [2, 3],
                    [0, 3],
                    [0, 0],
                    [0, 0],
                    [0, 0],
                ],
                element_topology_sample_node_coordinates_m: [
                    [0.0, 0.0, 0.0],
                    [0.25, 0.0, 0.0],
                    [0.0, 0.25, 0.0],
                    [0.25, 0.25, 0.0],
                    [0.0, 0.0, 0.0],
                    [0.0, 0.0, 0.0],
                    [0.0, 0.0, 0.0],
                    [0.0, 0.0, 0.0],
                ],
                element_topology_sample_element_edges: [[0, 1, 2], [2, 3, 4], [0, 0, 0], [0, 0, 0]],
                element_topology_sample_element_orientations: [
                    [1, 1, -1],
                    [1, 1, -1],
                    [0, 0, 0],
                    [0, 0, 0],
                ],
                element_topology_sample_element_areas_m2: [0.0625, 0.0625, 0.0, 0.0],
                element_topology_node_coordinates_m: vec![
                    [0.0, 0.0, 0.0],
                    [0.25, 0.0, 0.0],
                    [0.0, 0.25, 0.0],
                    [0.25, 0.25, 0.0],
                ],
                element_topology_edge_nodes: vec![[0, 1], [1, 2], [0, 2], [2, 3], [0, 3]],
                element_topology_element_edges: vec![[0, 1, 2], [2, 3, 4]],
                element_topology_element_orientations: vec![[1, 1, -1], [1, 1, -1]],
                element_topology_element_areas_m2: vec![0.0625, 0.0625],
                calibration_profile_override: None,
            }),
            ..NonlinearSolveOptions::default()
        },
    )
    .expect("prepared nonlinear solve should succeed");

    let topology = result
        .run
        .diagnostics
        .iter()
        .find(|diag| diag.code == "FEA_NONLINEAR_STATE_TOPOLOGY")
        .expect("nonlinear state topology diagnostic should be present");
    assert!(topology
        .message
        .contains("basis=prep_constant_strain_b_matrix"));
    assert!(topology.message.contains("element_count=2"));
    assert!(topology.message.contains("prep_recovery_edge_count="));
    assert!(topology.message.contains("mean_edge_length_m="));
    assert!(topology.message.contains("max_edge_strain_norm="));
    assert!(topology
        .message
        .contains("strain_component_coverage_ratio="));
}

#[test]
fn thermo_mechanical_nonlinear_emits_coupled_convergence_profile_diagnostic() {
    let model = fixture_model(FixtureId::NonlinearLoadPathMix);
    let result = crate::run_nonlinear_with_options(
        &model,
        ComputeBackend::Cpu,
        NonlinearSolveOptions {
            thermo_mechanical_context: Some(FeaThermoMechanicalContext {
                enabled: true,
                reference_temperature_k: 293.15,
                applied_temperature_delta_k: 90.0,
                thermal_expansion_coefficient: 1.4e-5,
                field_source: None,
                region_temperature_deltas: Vec::new(),
                time_profile: Vec::new(),
            }),
            ..NonlinearSolveOptions::default()
        },
    )
    .expect("thermo-mechanical nonlinear solve should succeed");

    assert_eq!(
        result.load_factors.len(),
        result.thermo_mechanical_temperature_snapshots.len()
    );
    assert_eq!(
        result.load_factors.len(),
        result.thermo_mechanical_thermal_strain_snapshots.len()
    );
    assert_eq!(
        result.load_factors.len(),
        result.thermo_mechanical_thermal_stress_snapshots.len()
    );
    assert_eq!(
        result.load_factors.len(),
        result.thermo_mechanical_displacement_snapshots.len()
    );
    assert_eq!(
        result.load_factors.len(),
        result.thermo_mechanical_von_mises_snapshots.len()
    );
    assert_eq!(
        result.load_factors.len(),
        result.thermo_mechanical_coupling_residual_snapshots.len()
    );
    assert_eq!(
        result.thermo_mechanical_temperature_snapshots[0].field_id,
        fea_thermo_mechanical_temperature_field_id(0)
    );
    assert_eq!(
        result.thermo_mechanical_temperature_snapshots[0].shape,
        vec![result.displacement_snapshots[0].shape[0]]
    );
    assert_eq!(
        result.thermo_mechanical_thermal_strain_snapshots[0].field_id,
        fea_thermo_mechanical_thermal_strain_field_id(0)
    );
    assert_eq!(
        result.thermo_mechanical_thermal_stress_snapshots[0].field_id,
        fea_thermo_mechanical_thermal_stress_field_id(0)
    );
    assert_eq!(
        result.thermo_mechanical_displacement_snapshots[0].field_id,
        fea_thermo_mechanical_displacement_field_id(0)
    );
    assert_eq!(
        result.thermo_mechanical_von_mises_snapshots[0].field_id,
        fea_thermo_mechanical_von_mises_field_id(0)
    );
    assert_eq!(
        result.thermo_mechanical_coupling_residual_snapshots[0].field_id,
        fea_thermo_mechanical_coupling_residual_field_id(0)
    );

    let profile = result
        .run
        .diagnostics
        .iter()
        .find(|diag| diag.code == "FEA_TM_NONLINEAR")
        .expect("thermo nonlinear profile diagnostic should be present");
    assert!(profile
        .message
        .contains("convergence_residual_target_peak="));
    assert!(profile
        .message
        .contains("convergence_increment_target_peak="));
    assert_thermo_mechanical_consistency_diagnostic(&result.run.diagnostics);
    assert_thermo_mechanical_temperature_field_diagnostic(&result.run.diagnostics);
}

#[test]
fn nonlinear_harder_fixtures_emit_difficulty_profile_signals() {
    for fixture in [
        FixtureId::NonlinearSofteningBenchmark,
        FixtureId::NonlinearLoadPathMix,
    ] {
        let model = fixture_model(fixture);
        let result = crate::run_nonlinear(&model, ComputeBackend::Cpu)
            .expect("hard nonlinear fixture solves");
        assert!(!result.load_factors.is_empty());
        assert!(result.backtrack_burst_count > 0);
        assert!(result.iteration_spike_count > 0);
        assert!(result.max_line_search_backtracks_per_increment > 0);
        assert!(result
            .run
            .diagnostics
            .iter()
            .any(|diag| diag.code == "FEA_NONLINEAR_CONVERGENCE"));
    }
}

#[test]
fn load_sweep_fixture_uses_operator_solver_path() {
    let baseline = crate::run_linear_static(
        &fixture_model(FixtureId::CantileverLinearStatic),
        ComputeBackend::Cpu,
    )
    .expect("baseline solve should succeed");
    let model = fixture_model(FixtureId::CantileverLoadSweep);
    let result =
        crate::run_linear_static(&model, ComputeBackend::Cpu).expect("solve should succeed");

    let convergence = result
        .diagnostics
        .iter()
        .find(|diag| diag.code == "FEA_CONVERGENCE")
        .expect("convergence diagnostic should be present");
    assert!(convergence.message.contains("residual_norm="));
    assert!(result
        .diagnostics
        .iter()
        .any(|diag| diag.code == "FEA_SOLVER_METHOD"));
    assert!(field(&result, FEA_FIELD_STRUCTURAL_DISPLACEMENT).element_count() >= 384);

    let baseline_max = baseline
        .field(FEA_FIELD_STRUCTURAL_DISPLACEMENT)
        .expect("baseline displacement field should be present")
        .as_host_f64()
        .expect("baseline displacement should be host-backed")
        .iter()
        .map(|value| value.abs())
        .fold(0.0_f64, f64::max);
    let sweep_max = result
        .field(FEA_FIELD_STRUCTURAL_DISPLACEMENT)
        .expect("sweep displacement field should be present")
        .as_host_f64()
        .expect("sweep displacement should be host-backed")
        .iter()
        .map(|value| value.abs())
        .fold(0.0_f64, f64::max);
    assert!(sweep_max > baseline_max);
}

#[test]
fn large_load_sweep_fixture_scales_dof_count() {
    let model = fixture_model(FixtureId::CantileverLargeLoadSweep);
    let result =
        crate::run_linear_static(&model, ComputeBackend::Cpu).expect("solve should succeed");

    assert!(field(&result, FEA_FIELD_STRUCTURAL_DISPLACEMENT).element_count() >= 1536);
    assert!(result
        .diagnostics
        .iter()
        .any(|diag| diag.code == "FEA_SOLVER_METHOD"));
}

#[test]
fn structural_bar_and_beam_reference_fixtures_emit_known_answer_checks() {
    let axial = crate::run_linear_static(
        &fixture_model(FixtureId::StructuralAxialBarReference),
        ComputeBackend::Cpu,
    )
    .expect("axial bar reference should solve");
    let bending = crate::run_linear_static(
        &fixture_model(FixtureId::StructuralBeamBendingReference),
        ComputeBackend::Cpu,
    )
    .expect("beam bending reference should solve");

    assert!(axial.diagnostics.iter().any(|diag| {
        diag.code == "FEA_STRUCTURAL_LINEAR_KNOWN_ANSWER"
            && diag.message.contains("known_answer_coverage_ratio=1")
    }));
    assert!(axial.diagnostics.iter().any(|diag| {
        diag.code == "FEA_STRUCTURAL_REFERENCE_KINEMATICS"
            && diag.message.contains("case=axial_bar_tension")
            && diag.message.contains("primary_component=x")
            && diag
                .message
                .contains("transverse_displacement_leakage_ratio=")
            && diag.message.contains("primary_stress_component_ratio=")
            && diag
                .message
                .contains("directional_reference_coverage_ratio=1")
            && diag
                .message
                .contains("closed_form_displacement_error_ratio=")
            && diag.message.contains("closed_form_stress_error_ratio=")
            && diag
                .message
                .contains("closed_form_reference_coverage_ratio=1")
    }));
    assert!(bending.diagnostics.iter().any(|diag| {
        diag.code == "FEA_STRUCTURAL_LINEAR_KNOWN_ANSWER"
            && diag.message.contains("known_answer_coverage_ratio=1")
    }));
    assert!(bending.diagnostics.iter().any(|diag| {
        diag.code == "FEA_STRUCTURAL_REFERENCE_KINEMATICS"
            && diag.message.contains("case=beam_transverse_bending")
            && diag.message.contains("primary_component=y")
            && diag
                .message
                .contains("transverse_displacement_leakage_ratio=")
            && diag.message.contains("primary_stress_component_ratio=")
            && diag
                .message
                .contains("directional_reference_coverage_ratio=1")
            && diag
                .message
                .contains("closed_form_displacement_error_ratio=")
            && diag.message.contains("closed_form_stress_error_ratio=")
            && diag
                .message
                .contains("closed_form_reference_coverage_ratio=1")
    }));

    let axial_displacement = field(&axial, FEA_FIELD_STRUCTURAL_DISPLACEMENT)
        .as_host_f64()
        .expect("axial displacement should be host-backed");
    let bending_displacement = field(&bending, FEA_FIELD_STRUCTURAL_DISPLACEMENT)
        .as_host_f64()
        .expect("bending displacement should be host-backed");
    let axial_x = component_peak(axial_displacement, 0);
    let axial_y = component_peak(axial_displacement, 1);
    let bending_x = component_peak(bending_displacement, 0);
    let bending_y = component_peak(bending_displacement, 1);

    assert!(axial_x > axial_y, "axial reference should be x dominated");
    assert!(
        bending_y > bending_x,
        "bending reference should be transverse-load dominated"
    );
}

#[test]
fn multi_material_fixture_has_distinct_response_profile() {
    let baseline = crate::run_linear_static(
        &fixture_model(FixtureId::CantileverLinearStatic),
        ComputeBackend::Cpu,
    )
    .expect("baseline solve should succeed");
    let multi_material = crate::run_linear_static(
        &fixture_model(FixtureId::MultiMaterialAssembly),
        ComputeBackend::Cpu,
    )
    .expect("multi-material solve should succeed");

    assert!(field(&multi_material, FEA_FIELD_STRUCTURAL_DISPLACEMENT).element_count() >= 9);
    assert!(multi_material
        .diagnostics
        .iter()
        .any(|diag| diag.code == "FEA_SOLVER_METHOD"));

    let baseline_peak = baseline
        .field(FEA_FIELD_STRUCTURAL_DISPLACEMENT)
        .expect("baseline displacement field should be present")
        .as_host_f64()
        .expect("baseline displacement should be host-backed")
        .iter()
        .map(|v| v.abs())
        .fold(0.0_f64, f64::max);
    let multi_peak = multi_material
        .field(FEA_FIELD_STRUCTURAL_DISPLACEMENT)
        .expect("multi displacement field should be present")
        .as_host_f64()
        .expect("multi displacement should be host-backed")
        .iter()
        .map(|v| v.abs())
        .fold(0.0_f64, f64::max);
    assert!(multi_peak > baseline_peak);

    assert!(multi_material
        .diagnostics
        .iter()
        .any(|diag| diag.code == "ANALYSIS_MATERIAL_ASSIGNMENT_CONFLICT_INFERRED"));
}

fn component_peak(values: &[f64], component: usize) -> f64 {
    values
        .chunks_exact(3)
        .map(|components| components.get(component).copied().unwrap_or(0.0).abs())
        .fold(0.0_f64, f64::max)
}
