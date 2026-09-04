use crate::{
    fea_nonlinear_contact_gap_field_id, fea_nonlinear_contact_pressure_field_id,
    fea_nonlinear_equivalent_plastic_strain_field_id, fea_nonlinear_load_factor_field_id,
    fea_nonlinear_plastic_strain_field_id, fea_nonlinear_residual_norm_field_id,
    fea_nonlinear_rotation_field_id, fea_nonlinear_von_mises_field_id,
    fea_thermal_axis_boundary_heat_flux_summary_field_id, fea_thermal_energy_storage_rate_field_id,
    fea_thermal_heat_flux_field_id, fea_thermal_temperature_gradient_field_id,
    fea_thermo_mechanical_coupling_residual_field_id, fea_thermo_mechanical_displacement_field_id,
    fea_thermo_mechanical_temperature_field_id, fea_thermo_mechanical_thermal_strain_field_id,
    fea_thermo_mechanical_thermal_stress_field_id, fea_thermo_mechanical_von_mises_field_id,
    fea_transient_acceleration_field_id, fea_transient_angular_acceleration_field_id,
    fea_transient_angular_velocity_field_id, fea_transient_kinetic_energy_field_id,
    fea_transient_residual_norm_field_id, fea_transient_rotation_field_id,
    fea_transient_strain_energy_field_id, fea_transient_velocity_field_id,
    fea_transient_von_mises_field_id,
    fixtures::{
        fixture_model, replace_fixture_excitations, replace_fixture_with_electric_current,
        FixtureExcitation, FixtureExcitationKind, FixtureId,
    },
    parity::{assert_vectors_within_tolerance, ParityTolerance},
    solve::{nonlinear::NonlinearSolveOptions, transient::TransientSolveOptions},
    ComputeBackend, ElectromagneticSolveOptions, FeaContactInterfaceContext,
    FeaElectroThermalContext, FeaRunResult, FeaThermoMechanicalContext, LinearStaticSolveOptions,
    ModalSolveOptions, ThermalSolveOptions, FEA_FIELD_MODAL_EIGENVALUE,
    FEA_FIELD_MODAL_FREQUENCY_HZ, FEA_FIELD_MODAL_MODAL_MASS, FEA_FIELD_MODAL_MODAL_STIFFNESS,
    FEA_FIELD_MODAL_M_ORTHOGONALITY, FEA_FIELD_MODAL_PARTICIPATION_FACTOR,
    FEA_FIELD_MODAL_RELATIVE_FREQUENCY_SEPARATION, FEA_FIELD_MODAL_RESIDUAL_NORM,
    FEA_FIELD_STRUCTURAL_BEAM_AXIAL_FORCE, FEA_FIELD_STRUCTURAL_BEAM_BENDING_MOMENT,
    FEA_FIELD_STRUCTURAL_BEAM_BENDING_STRESS, FEA_FIELD_STRUCTURAL_BEAM_SHEAR_FORCE,
    FEA_FIELD_STRUCTURAL_BEAM_TORSION_MOMENT, FEA_FIELD_STRUCTURAL_BEAM_TORSION_STRESS,
    FEA_FIELD_STRUCTURAL_DISPLACEMENT, FEA_FIELD_STRUCTURAL_EQUATION_SCALE,
    FEA_FIELD_STRUCTURAL_REACTION_FORCE, FEA_FIELD_STRUCTURAL_REACTION_MOMENT,
    FEA_FIELD_STRUCTURAL_RESIDUAL_NORM, FEA_FIELD_STRUCTURAL_ROTATION,
    FEA_FIELD_STRUCTURAL_SHELL_BENDING_MOMENT, FEA_FIELD_STRUCTURAL_SHELL_MEMBRANE_FORCE,
    FEA_FIELD_STRUCTURAL_SHELL_TRANSVERSE_SHEAR, FEA_FIELD_STRUCTURAL_SHELL_VON_MISES,
    FEA_FIELD_STRUCTURAL_STRAIN, FEA_FIELD_STRUCTURAL_STRAIN_ENERGY_DENSITY,
    FEA_FIELD_STRUCTURAL_STRESS, FEA_FIELD_STRUCTURAL_TOTAL_STRAIN_ENERGY,
    FEA_FIELD_STRUCTURAL_VON_MISES,
};

fn fixed_constraint(id: &str, region: &str) -> runmat_analysis_core::AnalysisConstraint {
    runmat_analysis_core::AnalysisConstraint::Structural(
        runmat_analysis_core::StructuralConstraint {
            constraint_id: id.to_string(),
            frame: runmat_analysis_core::CoordinateFrame::Global,
            kind: runmat_analysis_core::StructuralConstraintKind::Fixed {
                target: runmat_analysis_core::RegionTarget::authored(region),
            },
        },
    )
}
use runmat_analysis_core::{ElectromagneticDomain, MaterialElectricalModel, ProcedureStep};
use runmat_meshing_core::{
    ElementOrder, PersistentEntityId, PersistentEntityKind, SolverMeshArtifact,
};

fn experimental_linear_static_options() -> LinearStaticSolveOptions {
    LinearStaticSolveOptions {
        formulation_requirement: runmat_analysis_core::FormulationRequirement::Experimental,
        ..LinearStaticSolveOptions::default()
    }
}

fn experimental_modal_options() -> ModalSolveOptions {
    ModalSolveOptions {
        formulation_requirement: runmat_analysis_core::FormulationRequirement::Experimental,
        ..ModalSolveOptions::default()
    }
}

fn experimental_thermal_options() -> ThermalSolveOptions {
    ThermalSolveOptions {
        formulation_requirement: runmat_analysis_core::FormulationRequirement::Experimental,
        ..ThermalSolveOptions::default()
    }
}

fn experimental_electromagnetic_options() -> ElectromagneticSolveOptions {
    ElectromagneticSolveOptions {
        formulation_requirement: runmat_analysis_core::FormulationRequirement::Experimental,
        ..ElectromagneticSolveOptions::default()
    }
}

fn experimental_transient_options() -> TransientSolveOptions {
    TransientSolveOptions {
        formulation_requirement: runmat_analysis_core::FormulationRequirement::Experimental,
        ..TransientSolveOptions::default()
    }
}

fn experimental_nonlinear_options() -> NonlinearSolveOptions {
    NonlinearSolveOptions {
        formulation_requirement: runmat_analysis_core::FormulationRequirement::Experimental,
        ..NonlinearSolveOptions::default()
    }
}

#[test]
fn production_requirement_rejects_before_structural_solver_progress() {
    use std::sync::{Arc, Mutex};

    let model = fixture_model(FixtureId::CantileverLinearStatic);
    let events = Arc::new(Mutex::new(Vec::new()));
    let captured = Arc::clone(&events);
    let _guard = crate::replace_fea_progress_context(
        Some(Arc::new(move |event| {
            captured.lock().expect("progress lock").push(event);
        })),
        None,
    );
    let error = crate::run_linear_static_with_options(
        &model,
        ComputeBackend::Cpu,
        LinearStaticSolveOptions {
            solver_mesh: Some(single_tetrahedron_solver_mesh_for_model(&model)),
            solver_mesh_artifact_path: Some("solver_mesh.cbor".to_string()),
            ..LinearStaticSolveOptions::default()
        },
    )
    .expect_err("no current structural realization is production-qualified");

    assert!(matches!(
        error,
        crate::FeaRunError::FormulationAvailability(ref rejection)
            if rejection.reason
                == runmat_analysis_core::FormulationAdmissionRejectionReason::ProductionQualificationMissing
    ));
    assert!(events.lock().expect("progress lock").is_empty());
}

fn field<'a>(result: &'a FeaRunResult, field_id: &str) -> &'a runmat_analysis_core::AnalysisField {
    result.field(field_id).expect("field should be present")
}

fn host_field<'a>(result: &'a FeaRunResult, field_id: &str) -> &'a [f64] {
    field(result, field_id)
        .as_host_f64()
        .expect("field should be host-backed")
}

fn run_linear_static_with_single_tetrahedron_mesh(
    model: &runmat_analysis_core::AnalysisModel,
    backend: ComputeBackend,
) -> Result<FeaRunResult, crate::FeaRunError> {
    crate::run_linear_static_with_options(
        model,
        backend,
        LinearStaticSolveOptions {
            solver_mesh: Some(single_tetrahedron_solver_mesh_for_model(model)),
            solver_mesh_artifact_path: Some("solver_mesh.cbor".to_string()),
            ..experimental_linear_static_options()
        },
    )
}

fn assert_close(actual: f64, expected: f64, tolerance: f64) {
    assert!(
        (actual - expected).abs() <= tolerance,
        "actual={actual} expected={expected} tolerance={tolerance}",
    );
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
    let result = run_linear_static_with_single_tetrahedron_mesh(&model, ComputeBackend::Cpu)
        .expect("solve should succeed");

    assert_eq!(
        result.identity.qualification(),
        runmat_analysis_core::FormulationMaturity::Experimental
    );
    assert_eq!(
        result.identity.family(),
        runmat_analysis_core::PhysicsFamily::Structural
    );
    assert_eq!(
        result.identity.realization().components[0].topologies,
        vec![runmat_analysis_core::TopologyImplementation::CanonicalSolverMeshTet4]
    );

    assert_eq!(
        field(&result, FEA_FIELD_STRUCTURAL_DISPLACEMENT).element_count(),
        12
    );
    assert_eq!(
        field(&result, FEA_FIELD_STRUCTURAL_VON_MISES).element_count(),
        1
    );
    assert_eq!(
        field(&result, FEA_FIELD_STRUCTURAL_DISPLACEMENT).shape,
        vec![4, 3]
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
        field(&result, FEA_FIELD_STRUCTURAL_STRAIN_ENERGY_DENSITY).shape,
        vec![1]
    );
    assert_eq!(
        field(&result, FEA_FIELD_STRUCTURAL_REACTION_FORCE).shape,
        vec![3, 3]
    );
    assert_eq!(
        field(&result, FEA_FIELD_STRUCTURAL_ROTATION).shape,
        vec![0, 3]
    );
    assert_eq!(
        field(&result, FEA_FIELD_STRUCTURAL_REACTION_MOMENT).shape,
        vec![0, 3]
    );
    let displacement = host_field(&result, FEA_FIELD_STRUCTURAL_DISPLACEMENT);
    assert!(displacement.iter().all(|value| value.is_finite()));
    assert!(displacement.iter().any(|value| value.abs() > 0.0));

    let stress = host_field(&result, FEA_FIELD_STRUCTURAL_VON_MISES);
    assert!(stress[0].is_finite() && stress[0] > 0.0);
    assert!(host_field(&result, FEA_FIELD_STRUCTURAL_STRAIN)
        .iter()
        .any(|value| value.abs() > 0.0));
    let stress_tensor = host_field(&result, FEA_FIELD_STRUCTURAL_STRESS);
    assert!(stress_tensor.iter().any(|value| value.abs() > 0.0));
    assert!(stress_tensor.iter().all(|value| value.is_finite()));
    assert!(host_field(&result, FEA_FIELD_STRUCTURAL_STRAIN_ENERGY_DENSITY)[0] > 0.0);
    assert!(host_field(&result, FEA_FIELD_STRUCTURAL_REACTION_FORCE)
        .iter()
        .all(|value| value.is_finite()));
    assert!(host_field(&result, FEA_FIELD_STRUCTURAL_TOTAL_STRAIN_ENERGY)[0] > 0.0);
    assert!(host_field(&result, FEA_FIELD_STRUCTURAL_RESIDUAL_NORM)[0] <= 1.0e-6);
    assert!(host_field(&result, FEA_FIELD_STRUCTURAL_EQUATION_SCALE)[0] >= 1.0);
}

#[test]
fn solid_physics_pipelines_assemble_from_the_canonical_solver_mesh() {
    let model = fixture_model(FixtureId::CantileverLinearStatic);
    let mesh = single_tetrahedron_solver_mesh_for_model(&model);

    let mut modal_model = model.clone();
    modal_model.sequence.steps = vec![ProcedureStep::structural_modal("modal")];
    let modal = crate::run_modal_with_options(
        &modal_model,
        ComputeBackend::Cpu,
        ModalSolveOptions {
            mode_count: 2,
            solver_mesh: Some(mesh.clone()),
            ..experimental_modal_options()
        },
    )
    .expect("modal solve should consume the canonical mesh");
    assert_eq!(modal.mode_shapes[0].shape, vec![4, 3]);

    let thermal = crate::run_thermal_with_options(
        &model,
        ComputeBackend::Cpu,
        ThermalSolveOptions {
            step_count: 2,
            solver_mesh: Some(mesh.clone()),
            thermo_mechanical_context: Some(FeaThermoMechanicalContext {
                enabled: true,
                reference_temperature_k: 293.15,
                applied_temperature_delta_k: 45.0,
                thermal_expansion_coefficient: 1.2e-5,
                field_source: None,
                region_temperature_deltas: Vec::new(),
                sampled_time_profile: Vec::new(),
            }),
            ..experimental_thermal_options()
        },
    )
    .expect("thermal solve should consume the canonical mesh");
    assert_eq!(thermal.temperature_snapshots[0].shape, vec![4]);

    let mut transient_model = model.clone();
    transient_model.sequence.steps = vec![ProcedureStep::structural_transient("transient")];
    let transient = crate::run_transient_with_options(
        &transient_model,
        ComputeBackend::Cpu,
        TransientSolveOptions {
            step_count: 2,
            solver_mesh: Some(mesh.clone()),
            ..experimental_transient_options()
        },
    )
    .expect("transient solve should consume the canonical mesh");
    assert_eq!(transient.displacement_snapshots[0].shape, vec![4, 3]);

    let mut nonlinear_model = model.clone();
    nonlinear_model.sequence.steps = vec![ProcedureStep::structural_nonlinear("nonlinear")];
    let nonlinear = crate::run_nonlinear_with_options(
        &nonlinear_model,
        ComputeBackend::Cpu,
        NonlinearSolveOptions {
            increment_count: 2,
            solver_mesh: Some(mesh.clone()),
            ..experimental_nonlinear_options()
        },
    )
    .expect("nonlinear solve should consume the canonical mesh");
    assert_eq!(nonlinear.displacement_snapshots[0].shape, vec![4, 3]);

    let mut electromagnetic_model = model.clone();
    electromagnetic_model.electromagnetic = Some(ElectromagneticDomain {
        enabled: true,
        reference_frequency_hz: 60.0,
        applied_current_a: 120.0,
    });
    for material in &mut electromagnetic_model.materials {
        material.electrical = Some(MaterialElectricalModel::default());
    }
    let electromagnetic = crate::run_electromagnetic_with_options(
        &electromagnetic_model,
        ComputeBackend::Cpu,
        ElectromagneticSolveOptions {
            solver_mesh: Some(mesh),
            ..experimental_electromagnetic_options()
        },
    )
    .expect("electromagnetic solve should consume the canonical mesh");
    let topology_diagnostic = electromagnetic
        .run
        .diagnostics
        .iter()
        .find(|diagnostic| diagnostic.code == "FEA_EM_MAXWELL_EDGE_TOPOLOGY")
        .expect("electromagnetic solve should report its mesh topology");
    assert!(
        topology_diagnostic
            .message
            .contains("full_topology_edge_count=6"),
        "{}",
        topology_diagnostic.message
    );
    assert!(
        topology_diagnostic
            .message
            .contains("full_topology_element_count=1"),
        "{}",
        topology_diagnostic.message
    );
}

#[test]
fn moment_loads_require_rotational_dofs() {
    let mut model = fixture_model(FixtureId::CantileverLinearStatic);
    replace_fixture_excitations(
        &mut model,
        vec![FixtureExcitation {
            id: "tip_load".into(),
            region_id: "tip".into(),
            kind: FixtureExcitationKind::Moment([0.0, 0.0, 125.0]),
        }],
    );

    let err = crate::run_linear_static_with_options(
        &model,
        ComputeBackend::Cpu,
        experimental_linear_static_options(),
    )
    .expect_err("moment loads should require rotational DOFs");
    let message = err.to_string();
    assert!(message.contains("moment loads require rotational-DOF structural elements"));
    assert!(message.contains("use a wrench load"));
    assert!(message.contains("load_id=tip_load"));
}

#[test]
fn prescribed_rotations_require_rotational_dofs() {
    let mut model = fixture_model(FixtureId::CantileverLinearStatic);
    model.constraints = vec![runmat_analysis_core::AnalysisConstraint::Structural(
        runmat_analysis_core::StructuralConstraint {
            constraint_id: "tip_rotation".to_string(),
            frame: runmat_analysis_core::CoordinateFrame::Global,
            kind: runmat_analysis_core::StructuralConstraintKind::Prescribed(
                runmat_analysis_core::PrescribedStructuralDofs {
                    target: runmat_analysis_core::RegionTarget::boundary("tip"),
                    translations: Vec::new(),
                    rotations: vec![runmat_analysis_core::RotationDofValue {
                        dof: runmat_analysis_core::StructuralRotationDof::Rx,
                        value: runmat_analysis_core::PrescribedValue::Constant(
                            runmat_analysis_core::Rotation::new(0.1),
                        ),
                    }],
                },
            ),
        },
    )];

    let err = crate::run_linear_static_with_options(
        &model,
        ComputeBackend::Cpu,
        experimental_linear_static_options(),
    )
    .expect_err("prescribed rotations should require rotational DOFs");
    let message = err.to_string();
    assert!(message.contains("prescribed rotations require rotational-DOF structural elements"));
    assert!(message.contains("constraint_id=tip_rotation"));
}

#[test]
fn canonical_constant_signal_drives_prescribed_translation_assembly() {
    let inline_model = fixture_model(FixtureId::TransientShock);
    let inline = crate::assembly::assemble_structural_system(
        &inline_model,
        None,
        None,
        None,
        crate::assembly::StructuralFormulationRequirement::ExperimentalSyntheticOperator,
    )
    .expect("inline prescribed value should assemble");

    let mut signal_model = inline_model.clone();
    signal_model
        .signals
        .push(runmat_analysis_core::PhysicalSignal::Displacement(
            runmat_analysis_core::Signal {
                signal_id: runmat_analysis_core::SignalId("zero_displacement".into()),
                shape: runmat_analysis_core::SignalShape::Scalar,
                frame: None,
                representation: runmat_analysis_core::SignalRepresentation::Constant(
                    runmat_analysis_core::SignalValue::Scalar(
                        runmat_analysis_core::Displacement::new(0.0),
                    ),
                ),
                precision: runmat_analysis_core::SignalPrecision::Real64,
                evidence: runmat_analysis_core::SignalApproximationEvidence {
                    max_absolute_error: None,
                    max_relative_error: None,
                    derivative_max_absolute_error: None,
                    sample_count: 1,
                },
                provenance: runmat_analysis_core::SignalProvenance::Authored,
            },
        ));
    for constraint in &mut signal_model.constraints {
        let runmat_analysis_core::AnalysisConstraint::Structural(constraint) = constraint else {
            continue;
        };
        let runmat_analysis_core::StructuralConstraintKind::Prescribed(values) =
            &mut constraint.kind
        else {
            continue;
        };
        for prescribed in &mut values.translations {
            prescribed.value = runmat_analysis_core::PrescribedValue::Signal(
                runmat_analysis_core::SignalId("zero_displacement".into()),
            );
        }
    }

    let referenced = crate::assembly::assemble_structural_system(
        &signal_model,
        None,
        None,
        None,
        crate::assembly::StructuralFormulationRequirement::ExperimentalSyntheticOperator,
    )
    .expect("signal-backed prescribed value should assemble");
    assert_eq!(referenced.operator.constrained, inline.operator.constrained);
    assert_eq!(referenced.operator.rhs, inline.operator.rhs);
}

#[test]
fn assembly_summary_reports_structural_dof_layout_metrics() {
    let model = fixture_model(FixtureId::StructuralBeamCantileverEndMomentReference);
    let summary = crate::assembly::assemble_structural_system(
        &model,
        None,
        None,
        None,
        crate::assembly::StructuralFormulationRequirement::PhysicalDiscretization,
    )
    .expect("beam system should assemble");

    assert_eq!(summary.dof_count, 12);
    assert_eq!(summary.structural_node_count, 2);
    assert_eq!(summary.structural_translational_dof_count, 6);
    assert_eq!(summary.structural_rotational_dof_count, 6);
    assert_eq!(summary.structural_rotation_node_count, 2);
    assert_eq!(summary.structural_moment_load_count, 1);
    assert_eq!(summary.structural_direct_rotational_moment_load_count, 1);
    assert_eq!(summary.structural_beam_element_count, 1);
    assert_eq!(summary.structural_shell_element_count, 0);
    assert_eq!(summary.structural_solid_element_count, 0);
}

#[test]
fn beam_assembly_uses_lumped_rotational_inertia() {
    let model = fixture_model(FixtureId::StructuralBeamCantileverEndMomentReference);
    let summary = crate::assembly::assemble_structural_system(
        &model,
        None,
        None,
        None,
        crate::assembly::StructuralFormulationRequirement::PhysicalDiscretization,
    )
    .expect("beam operator should assemble");
    let layout = &summary.structural_dof_layout;
    let node = 0usize;
    let mass = &summary.operator.mass_diag;

    let translational_mass = 7850.0 * 2.0e-4 / 2.0;
    let torsion_inertia = 7850.0 * 2.4e-9 / 2.0;
    let bending_y_inertia = 7850.0 * 1.6e-9 / 2.0;
    let bending_z_inertia = 7850.0 * 6.4e-9 / 2.0;

    let dof = |kind| {
        layout
            .index(node, kind)
            .expect("beam node should expose requested DOF")
    };
    assert_close(
        mass[dof(crate::assembly::dofs::StructuralDofKind::Ux)],
        translational_mass,
        1.0e-12,
    );
    assert_close(
        mass[dof(crate::assembly::dofs::StructuralDofKind::Uy)],
        translational_mass,
        1.0e-12,
    );
    assert_close(
        mass[dof(crate::assembly::dofs::StructuralDofKind::Uz)],
        translational_mass,
        1.0e-12,
    );
    assert_close(
        mass[dof(crate::assembly::dofs::StructuralDofKind::Rx)],
        torsion_inertia,
        1.0e-15,
    );
    assert_close(
        mass[dof(crate::assembly::dofs::StructuralDofKind::Ry)],
        bending_y_inertia,
        1.0e-15,
    );
    assert_close(
        mass[dof(crate::assembly::dofs::StructuralDofKind::Rz)],
        bending_z_inertia,
        1.0e-15,
    );
}

#[test]
fn shell_moment_solves_rotation_and_reaction_moment() {
    let model = fixture_model(FixtureId::StructuralShellPlateMomentReference);

    let result = crate::run_linear_static_with_options(
        &model,
        ComputeBackend::Cpu,
        experimental_linear_static_options(),
    )
    .expect("shell moment solve should run");
    assert_eq!(
        field(&result, FEA_FIELD_STRUCTURAL_DISPLACEMENT).shape,
        vec![3, 3]
    );
    assert_eq!(
        field(&result, FEA_FIELD_STRUCTURAL_ROTATION).shape,
        vec![3, 3]
    );
    assert_eq!(
        field(&result, FEA_FIELD_STRUCTURAL_REACTION_MOMENT).shape,
        vec![3, 3]
    );
    assert_eq!(
        field(&result, FEA_FIELD_STRUCTURAL_SHELL_MEMBRANE_FORCE).shape,
        vec![1, 3]
    );
    assert_eq!(
        field(&result, FEA_FIELD_STRUCTURAL_SHELL_BENDING_MOMENT).shape,
        vec![1, 3]
    );
    assert_eq!(
        field(&result, FEA_FIELD_STRUCTURAL_SHELL_TRANSVERSE_SHEAR).shape,
        vec![1, 2]
    );
    assert_eq!(
        field(&result, FEA_FIELD_STRUCTURAL_SHELL_VON_MISES).shape,
        vec![1]
    );
    let rotation = host_field(&result, FEA_FIELD_STRUCTURAL_ROTATION);
    let reaction_moment = host_field(&result, FEA_FIELD_STRUCTURAL_REACTION_MOMENT);
    let shell_bending = host_field(&result, FEA_FIELD_STRUCTURAL_SHELL_BENDING_MOMENT);
    let shell_von_mises = host_field(&result, FEA_FIELD_STRUCTURAL_SHELL_VON_MISES);
    assert!(rotation[7].abs() > 0.0);
    assert!(reaction_moment[1].abs() > 0.0);
    assert!(shell_bending.iter().any(|value| value.abs() > 0.0));
    assert!(shell_von_mises[0].is_finite() && shell_von_mises[0] > 0.0);
    assert!(result.diagnostics.iter().any(|diag| {
        diag.code == "FEA_STRUCTURAL_ROTATIONAL_DOF"
            && diag.message.contains("structural_rotational_dof_count=9")
            && diag.message.contains("structural_shell_element_count=1")
            && diag
                .message
                .contains("structural_direct_rotational_moment_load_count=1")
    }));
}

#[test]
fn beam_end_moment_solves_rotation_and_reaction_moment() {
    let moment_n_m = 1.25;
    let length_m = 1.0;
    let youngs_modulus_pa = 200.0e9;
    let iz_m4 = 6.4e-9;
    let mut model = fixture_model(FixtureId::CantileverLinearStatic);
    model.model_id = runmat_analysis_core::AnalysisModelId("beam_end_moment".to_string());
    model.structural = Some(runmat_analysis_core::StructuralModel {
        nodes: vec![
            runmat_analysis_core::StructuralNode {
                node_id: 1,
                coordinates_m: [0.0, 0.0, 0.0],
            },
            runmat_analysis_core::StructuralNode {
                node_id: 2,
                coordinates_m: [length_m, 0.0, 0.0],
            },
        ],
        elements: vec![runmat_analysis_core::StructuralElement {
            element_id: "beam_1".to_string(),
            region_id: "beam_span".to_string(),
            kind: runmat_analysis_core::StructuralElementKind::Beam(
                runmat_analysis_core::BeamElementModel {
                    node_ids: [1, 2],
                    section_id: "rect".to_string(),
                    reference_axis: [0.0, 1.0, 0.0],
                },
            ),
        }],
        beam_sections: vec![runmat_analysis_core::BeamSectionModel {
            section_id: "rect".to_string(),
            area_m2: 2.0e-4,
            iy_m4: 1.6e-9,
            iz_m4,
            torsion_j_m4: 2.4e-9,
            outer_fiber_y_m: 0.01,
            outer_fiber_z_m: 0.005,
            torsion_outer_radius_m: 0.011_180_339_887_498_949,
        }],
        shell_sections: Vec::new(),
    });
    model.constraints = vec![fixed_constraint("fixed_root", "node:1")];
    replace_fixture_excitations(
        &mut model,
        vec![FixtureExcitation {
            id: "tip_moment".into(),
            region_id: "node:2".into(),
            kind: FixtureExcitationKind::Moment([0.0, 0.0, moment_n_m]),
        }],
    );

    let result = crate::run_linear_static_with_options(
        &model,
        ComputeBackend::Cpu,
        experimental_linear_static_options(),
    )
    .expect("beam moment solve should run");
    assert_eq!(
        field(&result, FEA_FIELD_STRUCTURAL_DISPLACEMENT).shape,
        vec![2, 3]
    );
    assert_eq!(
        field(&result, FEA_FIELD_STRUCTURAL_ROTATION).shape,
        vec![2, 3]
    );
    assert_eq!(
        field(&result, FEA_FIELD_STRUCTURAL_REACTION_MOMENT).shape,
        vec![2, 3]
    );
    assert_eq!(
        field(&result, FEA_FIELD_STRUCTURAL_BEAM_AXIAL_FORCE).shape,
        vec![1]
    );
    assert_eq!(
        field(&result, FEA_FIELD_STRUCTURAL_BEAM_SHEAR_FORCE).shape,
        vec![1, 2]
    );
    assert_eq!(
        field(&result, FEA_FIELD_STRUCTURAL_BEAM_TORSION_MOMENT).shape,
        vec![1]
    );
    assert_eq!(
        field(&result, FEA_FIELD_STRUCTURAL_BEAM_BENDING_MOMENT).shape,
        vec![1, 2]
    );
    assert_eq!(
        field(&result, FEA_FIELD_STRUCTURAL_BEAM_BENDING_STRESS).shape,
        vec![1, 2]
    );
    assert_eq!(
        field(&result, FEA_FIELD_STRUCTURAL_BEAM_TORSION_STRESS).shape,
        vec![1]
    );

    let expected_rotation = moment_n_m * length_m / (youngs_modulus_pa * iz_m4);
    let expected_displacement = moment_n_m * length_m.powi(2) / (2.0 * youngs_modulus_pa * iz_m4);
    let expected_bending_stress = moment_n_m * 0.01 / iz_m4;
    let displacement = host_field(&result, FEA_FIELD_STRUCTURAL_DISPLACEMENT);
    let rotation = host_field(&result, FEA_FIELD_STRUCTURAL_ROTATION);
    let reaction_moment = host_field(&result, FEA_FIELD_STRUCTURAL_REACTION_MOMENT);
    let beam_bending_moment = host_field(&result, FEA_FIELD_STRUCTURAL_BEAM_BENDING_MOMENT);
    let beam_bending_stress = host_field(&result, FEA_FIELD_STRUCTURAL_BEAM_BENDING_STRESS);
    let beam_torsion = host_field(&result, FEA_FIELD_STRUCTURAL_BEAM_TORSION_MOMENT);

    assert!(
        (rotation[5] - expected_rotation).abs() <= expected_rotation.abs() * 1.0e-6,
        "rotation={} expected={}",
        rotation[5],
        expected_rotation
    );
    assert!(
        (displacement[4] - expected_displacement).abs() <= expected_displacement.abs() * 1.0e-6,
        "displacement={} expected={}",
        displacement[4],
        expected_displacement
    );
    assert!(
        (reaction_moment[2] + moment_n_m).abs() <= moment_n_m.abs() * 1.0e-6,
        "reaction_moment={} applied={}",
        reaction_moment[2],
        moment_n_m
    );
    assert!(
        (beam_bending_moment[1].abs() - moment_n_m).abs() <= moment_n_m.abs() * 1.0e-6,
        "beam_bending_moment_z={} applied={}",
        beam_bending_moment[1],
        moment_n_m
    );
    assert!(
        (beam_bending_stress[1].abs() - expected_bending_stress).abs()
            <= expected_bending_stress.abs() * 1.0e-6,
        "beam_bending_stress_z={} expected={}",
        beam_bending_stress[1],
        expected_bending_stress
    );
    assert!(beam_torsion[0].abs() <= 1.0e-8);
    assert!(result.diagnostics.iter().any(|diag| {
        diag.code == "FEA_STRUCTURAL_ROTATIONAL_DOF"
            && diag.message.contains("structural_rotational_dof_count=6")
            && diag.message.contains("structural_rotation_node_count=2")
            && diag.message.contains("structural_moment_load_count=1")
            && diag
                .message
                .contains("structural_direct_rotational_moment_load_count=1")
            && diag.message.contains("structural_beam_element_count=1")
            && diag
                .message
                .contains("structural_beam_local_frame_coverage_ratio=1")
            && diag
                .message
                .contains("structural_beam_stiffness_matrix_symmetry_residual=0")
    }));
    assert!(result.diagnostics.iter().any(|diag| {
        diag.code == "FEA_STRUCTURAL_MOMENT_BALANCE"
            && diag
                .message
                .contains("structural_reaction_moment_norm_n_m=1.25")
            && diag
                .message
                .contains("structural_moment_requested_norm_n_m=1.25")
            && diag
                .message
                .contains("structural_moment_realized_norm_n_m=1.25")
            && diag
                .message
                .contains("structural_moment_realization_ratio=1")
            && diag
                .message
                .contains("structural_moment_balance_residual_ratio=")
    }));
}

#[test]
fn thermo_mechanical_linear_static_emits_coupled_fields() {
    let mut model = fixture_model(FixtureId::ThermoMechanicalKickoff);
    model.sequence.steps = vec![ProcedureStep::coupled(
        "thermomechanical_static",
        runmat_analysis_core::CoupledPhysicsFamily::Thermomechanical,
        runmat_analysis_core::AnalysisProcedure::Static,
    )];
    let result = crate::run_linear_static_with_options(
        &model,
        ComputeBackend::Cpu,
        LinearStaticSolveOptions {
            solver_mesh: Some(single_tetrahedron_solver_mesh_for_model(&model)),
            solver_mesh_artifact_path: Some("solver_mesh.cbor".to_string()),
            thermo_mechanical_context: Some(FeaThermoMechanicalContext {
                enabled: true,
                reference_temperature_k: 293.15,
                applied_temperature_delta_k: 65.0,
                thermal_expansion_coefficient: 1.2e-5,
                field_source: None,
                region_temperature_deltas: Vec::new(),
                sampled_time_profile: Vec::new(),
            }),
            ..experimental_linear_static_options()
        },
    )
    .expect("thermo-mechanical linear static solve should succeed");

    assert_eq!(
        result.identity.family(),
        runmat_analysis_core::PhysicsFamily::Coupled(
            runmat_analysis_core::CoupledPhysicsFamily::Thermomechanical,
        )
    );
    assert_eq!(
        result.identity.realization().coupling,
        runmat_analysis_core::CouplingImplementation::OnePassPrescribedThermomechanicalProfile
    );
    assert_eq!(
        result.identity.realization().temporals,
        vec![
            runmat_analysis_core::TemporalImplementation::StructuralStaticLinearSolve,
            runmat_analysis_core::TemporalImplementation::PrescribedThermomechanicalProfileApplication,
        ]
    );
    assert_eq!(
        result.identity.realization().components[1].spatial_operator,
        runmat_analysis_core::SpatialOperatorImplementation::PrescribedThermalCouplingProfile
    );
    assert_eq!(
        result.identity.realization().components[1].sources,
        vec![runmat_analysis_core::SourceImplementation::PrescribedTemperatureDeltaProfile]
    );
    assert!(!result
        .identity
        .realization()
        .boundary_transfers
        .contains(&runmat_analysis_core::BoundaryTransferImplementation::NotApplicable));
    assert_eq!(
        result.identity.family(),
        runmat_analysis_core::PhysicsFamily::Coupled(
            runmat_analysis_core::CoupledPhysicsFamily::Thermomechanical,
        )
    );
    assert_eq!(
        result.identity.procedure(),
        runmat_analysis_core::AnalysisProcedure::Static
    );

    assert_eq!(
        field(&result, &fea_thermo_mechanical_temperature_field_id(0)).field_id,
        fea_thermo_mechanical_temperature_field_id(0)
    );
    assert_eq!(
        field(&result, &fea_thermo_mechanical_temperature_field_id(0)).shape,
        vec![240]
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
fn linear_static_rejects_structural_step_with_thermomechanical_assembly() {
    let model = fixture_model(FixtureId::CantileverLinearStatic);
    let mesh = single_tetrahedron_solver_mesh_for_model(&model);
    let error = crate::run_linear_static_with_options(
        &model,
        ComputeBackend::Cpu,
        LinearStaticSolveOptions {
            solver_mesh: Some(mesh),
            thermo_mechanical_context: Some(FeaThermoMechanicalContext {
                enabled: true,
                reference_temperature_k: 293.15,
                applied_temperature_delta_k: 65.0,
                thermal_expansion_coefficient: 1.2e-5,
                field_source: None,
                region_temperature_deltas: Vec::new(),
                sampled_time_profile: Vec::new(),
            }),
            ..experimental_linear_static_options()
        },
    )
    .expect_err("a structural step must not silently execute coupled assembly math");

    let message = error.to_string();
    assert!(message.contains("requests Structural"));
    assert!(message.contains("realizes Coupled(Thermomechanical)"));
}

#[test]
fn transient_rejects_thermomechanical_step_without_coupled_assembly() {
    let mut model = fixture_model(FixtureId::TransientLong);
    model.sequence.steps = vec![ProcedureStep::coupled(
        "thermomechanical_transient_without_context",
        runmat_analysis_core::CoupledPhysicsFamily::Thermomechanical,
        runmat_analysis_core::AnalysisProcedure::TransientResponse,
    )];
    let error = crate::run_transient_with_options(
        &model,
        ComputeBackend::Cpu,
        experimental_transient_options(),
    )
    .expect_err("a thermomechanical step must realize coupled assembly math");

    let message = error.to_string();
    assert!(message.contains("requests Coupled(Thermomechanical)"));
    assert!(message.contains("realizes Structural"));
}

#[test]
fn nonlinear_rejects_explicit_connectivity_that_drops_thermomechanical_coupling() {
    let mut model = fixture_model(FixtureId::StructuralBeamCantileverEndMomentReference);
    model.sequence.steps = vec![ProcedureStep::coupled(
        "thermomechanical_explicit_nonlinear",
        runmat_analysis_core::CoupledPhysicsFamily::Thermomechanical,
        runmat_analysis_core::AnalysisProcedure::Nonlinear,
    )];
    let error = crate::run_nonlinear_with_options(
        &model,
        ComputeBackend::Cpu,
        NonlinearSolveOptions {
            thermo_mechanical_context: Some(FeaThermoMechanicalContext {
                enabled: true,
                reference_temperature_k: 293.15,
                applied_temperature_delta_k: 65.0,
                thermal_expansion_coefficient: 1.2e-5,
                field_source: None,
                region_temperature_deltas: Vec::new(),
                sampled_time_profile: Vec::new(),
            }),
            ..experimental_nonlinear_options()
        },
    )
    .expect_err("explicit connectivity must not silently drop thermomechanical coupling");

    let message = error.to_string();
    assert!(message.contains("requests Coupled(Thermomechanical)"));
    assert!(message.contains("realizes Structural"));
}

#[test]
fn structural_pipeline_rejects_ambiguous_procedure_steps() {
    let mut model = fixture_model(FixtureId::CantileverLinearStatic);
    model
        .sequence
        .steps
        .push(ProcedureStep::structural_static("static_duplicate"));
    let error = crate::run_linear_static_with_options(
        &model,
        ComputeBackend::Cpu,
        experimental_linear_static_options(),
    )
    .expect_err("a structural pipeline must resolve one canonical procedure step");

    assert!(error.to_string().contains("multiple steps were found"));
}

#[test]
fn convergence_diagnostics_are_emitted() {
    let model = fixture_model(FixtureId::CantileverLinearStatic);
    let result = run_linear_static_with_single_tetrahedron_mesh(&model, ComputeBackend::Cpu)
        .expect("solve should succeed");
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
            && diag
                .message
                .contains("basis=solid_tetrahedron4_constant_strain")
            && diag.message.contains("solver_mesh_element_count=1")
    }));
}

#[test]
fn solid_continuum_requires_solver_mesh_when_requested() {
    let model = fixture_model(FixtureId::CantileverLinearStatic);
    let err = crate::run_linear_static_with_options(
        &model,
        ComputeBackend::Cpu,
        LinearStaticSolveOptions {
            require_solver_mesh_for_solid: true,
            ..experimental_linear_static_options()
        },
    )
    .expect_err("solid continuum solve should require a solver mesh");

    assert!(matches!(err, crate::FeaRunError::InvalidModel(_)));
    assert!(err.to_string().contains("require a canonical solver mesh"));
}

#[test]
fn explicit_beam_topology_does_not_require_solver_mesh() {
    let model = fixture_model(FixtureId::StructuralBeamCantileverEndMomentReference);
    let result = crate::run_linear_static_with_options(
        &model,
        ComputeBackend::Cpu,
        LinearStaticSolveOptions {
            require_solver_mesh_for_solid: true,
            ..experimental_linear_static_options()
        },
    )
    .expect("explicit beam topology should not require a solid solver mesh");

    assert!(result.diagnostics.iter().any(|diag| {
        diag.code == "FEA_STRUCTURAL_ROTATIONAL_DOF"
            && diag.message.contains("structural_beam_element_count=1")
    }));
}

#[test]
fn solver_mesh_linear_static_reports_solid_assembly_basis() {
    let model = fixture_model(FixtureId::CantileverLinearStatic);
    let result = crate::run_linear_static_with_options(
        &model,
        ComputeBackend::Cpu,
        LinearStaticSolveOptions {
            solver_mesh: Some(single_tetrahedron_solver_mesh()),
            solver_mesh_artifact_path: Some("solver_mesh.cbor".to_string()),
            ..experimental_linear_static_options()
        },
    )
    .expect("analysis-mesh-backed linear static solve should succeed");

    assert!(result.diagnostics.iter().any(|diag| {
        diag.code == "FEA_STRUCTURAL_SOLID_ASSEMBLY"
            && diag.message.contains("basis=solid_tetrahedron4_stiffness")
            && diag.message.contains("storage=sparse_csr")
            && diag.message.contains("solid_node_count=4")
            && diag.message.contains("solid_element_count=1")
            && diag.message.contains("stiffness_present=true")
    }));
    assert!(result.diagnostics.iter().any(|diag| {
        diag.code == "FEA_STRUCTURAL_FIELD_RECOVERY"
            && diag
                .message
                .contains("basis=solid_tetrahedron4_constant_strain")
            && diag.message.contains("solver_mesh_node_count=4")
            && diag.message.contains("solver_mesh_element_count=1")
    }));
}

fn single_tetrahedron_solver_mesh() -> SolverMeshArtifact {
    let mut mesh = crate::assembly::solver_solid::tests::artifact(ElementOrder::Tet4);
    add_face_region(&mut mesh, 0, "root");
    add_face_region(&mut mesh, 1, "tip");
    reseal_solver_mesh(&mut mesh);
    mesh
}

fn single_tetrahedron_solver_mesh_for_model(
    model: &runmat_analysis_core::AnalysisModel,
) -> SolverMeshArtifact {
    let mut mesh = single_tetrahedron_solver_mesh();
    if let Some(assignment) = model.material_assignments.first() {
        let region = PersistentEntityId {
            kind: PersistentEntityKind::Region,
            source_topology_id: assignment.region_id.clone(),
            assembly_path: vec!["root".to_owned()],
        };
        mesh.topology.volume_elements[0].region_id = region.clone();
        mesh.topology.regions[0].region_id = region;
    }
    let mut root_regions = std::collections::BTreeSet::from(["root".to_string()]);
    root_regions.extend(
        model
            .constraints
            .iter()
            .filter_map(|condition| condition.target().map(|target| target.region_id.clone())),
    );
    let mut loaded_regions = std::collections::BTreeSet::from(["tip".to_string()]);
    loaded_regions.extend(model.excitations.iter().filter_map(|action| {
        match &action.excitation().target {
            runmat_analysis_core::ExcitationTarget::Region(target) => {
                Some(target.region_id.clone())
            }
            runmat_analysis_core::ExcitationTarget::ReferencePoint(point) => {
                Some(point.reference_point_id.clone())
            }
            _ => None,
        }
    }));
    loaded_regions.extend(
        model
            .material_assignments
            .iter()
            .map(|assignment| assignment.region_id.clone()),
    );
    loaded_regions.extend(
        model
            .material_assignments
            .iter()
            .map(|assignment| assignment.assigned_material_id.clone()),
    );
    for region in root_regions {
        add_face_region(&mut mesh, 0, &region);
    }
    for region in loaded_regions {
        add_face_region(&mut mesh, 1, &region);
    }
    reseal_solver_mesh(&mut mesh);
    mesh
}

fn add_face_region(mesh: &mut SolverMeshArtifact, face_index: usize, region_id: &str) {
    let provenance = &mut mesh.topology.boundary_faces[face_index].provenance;
    let entity = PersistentEntityId {
        kind: PersistentEntityKind::Face,
        source_topology_id: region_id.to_owned(),
        assembly_path: vec!["root".to_owned()],
    };
    if !provenance.contains(&entity) {
        provenance.push(entity);
        provenance.sort();
    }
}

fn reseal_solver_mesh(mesh: &mut SolverMeshArtifact) {
    mesh.canonical_digest = runmat_meshing_core::StableDigest::ZERO;
    mesh.seal_canonical_digest()
        .expect("test solver mesh should remain canonical");
}

#[test]
fn deterministic_replay_for_fixture_is_stable() {
    let model = fixture_model(FixtureId::CantileverLinearStatic);
    let first = run_linear_static_with_single_tetrahedron_mesh(&model, ComputeBackend::Cpu)
        .expect("first run should succeed");
    let second = run_linear_static_with_single_tetrahedron_mesh(&model, ComputeBackend::Cpu)
        .expect("second run should succeed");

    assert_eq!(first.fields, second.fields);
    assert_eq!(first.diagnostics, second.diagnostics);
}

#[test]
fn cpu_gpu_parity_respects_tolerance_policy() {
    let model = fixture_model(FixtureId::CantileverLinearStatic);
    let cpu = run_linear_static_with_single_tetrahedron_mesh(&model, ComputeBackend::Cpu)
        .expect("cpu run should succeed");
    let gpu = run_linear_static_with_single_tetrahedron_mesh(&model, ComputeBackend::Gpu)
        .expect("gpu run should succeed");

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
    let err = crate::run_linear_static_with_options(
        &model,
        ComputeBackend::Cpu,
        experimental_linear_static_options(),
    )
    .expect_err("fixture should fail validation");
    assert!(err
        .to_string()
        .contains("ANALYSIS_VALIDATION_MISSING_MATERIALS"));
}

#[test]
fn thermal_pipeline_rejects_moment_loads_before_assembly() {
    let model = fixture_model(FixtureId::StructuralBeamCantileverEndMomentReference);
    let err = crate::run_thermal_with_options(
        &model,
        ComputeBackend::Cpu,
        experimental_thermal_options(),
    )
    .expect_err("thermal solve should reject structural moment loads");
    let message = err.to_string();

    assert!(message.contains("moment loads are structural loads"));
    assert!(message.contains("thermal"));
    assert!(message.contains("load_id=tip_moment_z"));
}

#[test]
fn electromagnetic_pipeline_rejects_moment_loads_before_domain_validation() {
    let model = fixture_model(FixtureId::StructuralBeamCantileverEndMomentReference);
    let err = crate::run_electromagnetic_with_options(
        &model,
        ComputeBackend::Cpu,
        experimental_electromagnetic_options(),
    )
    .expect_err("electromagnetic solve should reject structural moment loads");
    let message = err.to_string();

    assert!(message.contains("moment loads are structural loads"));
    assert!(message.contains("electromagnetic"));
    assert!(message.contains("load_id=tip_moment_z"));
}

#[test]
fn electromagnetic_pipeline_consumes_electric_current_without_structural_admission() {
    let mut model = fixture_model(FixtureId::StructuralBeamCantileverEndMomentReference);
    replace_fixture_with_electric_current(&mut model, "coil_current", "beam_span", 4.0);
    model.electromagnetic = Some(ElectromagneticDomain {
        enabled: true,
        reference_frequency_hz: 60.0,
        applied_current_a: 4.0,
    });
    for material in &mut model.materials {
        material.electrical = Some(MaterialElectricalModel::default());
    }

    let result = crate::run_electromagnetic_with_options(
        &model,
        ComputeBackend::Cpu,
        experimental_electromagnetic_options(),
    )
    .expect("electromagnetic admission must own the electric-current source");

    assert_eq!(
        result.run.identity.family(),
        runmat_analysis_core::PhysicsFamily::Electromagnetic
    );
    assert!(result
        .run
        .identity
        .realization()
        .components
        .iter()
        .any(|component| component
            .sources
            .contains(&runmat_analysis_core::SourceImplementation::OrdinalDistribution)));
}

#[test]
fn modal_solver_emits_modes_for_modal_step_fixture() {
    let mut model = fixture_model(FixtureId::CantileverLinearStatic);
    model.sequence.steps = vec![ProcedureStep::structural_modal("modal_1".to_string())];
    let result =
        crate::run_modal_with_options(&model, ComputeBackend::Cpu, experimental_modal_options())
            .expect("modal solve should succeed");

    assert!(!result.frequencies_hz.is_empty());
    assert_eq!(result.frequencies_hz.len(), result.mode_shapes.len());
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
        result.frequencies_hz.len()
    );
    assert_eq!(
        field(&result.run, FEA_FIELD_MODAL_EIGENVALUE).element_count(),
        result.frequencies_hz.len()
    );
    assert_eq!(
        field(&result.run, FEA_FIELD_MODAL_MODAL_MASS).element_count(),
        result.frequencies_hz.len()
    );
    assert_eq!(
        field(&result.run, FEA_FIELD_MODAL_MODAL_STIFFNESS).element_count(),
        result.frequencies_hz.len()
    );
    assert_eq!(
        field(&result.run, FEA_FIELD_MODAL_PARTICIPATION_FACTOR).element_count(),
        result.frequencies_hz.len()
    );
    assert_eq!(
        field(&result.run, FEA_FIELD_MODAL_RESIDUAL_NORM).element_count(),
        result.frequencies_hz.len()
    );
    assert_eq!(
        field(&result.run, FEA_FIELD_MODAL_RELATIVE_FREQUENCY_SEPARATION).element_count(),
        result.frequencies_hz.len()
    );
    let orthogonality = field(&result.run, FEA_FIELD_MODAL_M_ORTHOGONALITY);
    assert_eq!(
        orthogonality.element_count(),
        result.frequencies_hz.len() * result.frequencies_hz.len()
    );
    assert_eq!(
        orthogonality.shape,
        vec![result.frequencies_hz.len(), result.frequencies_hz.len()]
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
    model.sequence.steps = vec![ProcedureStep::structural_transient(
        "transient_1".to_string(),
    )];
    let result = crate::run_transient_with_options(
        &model,
        ComputeBackend::Cpu,
        experimental_transient_options(),
    )
    .expect("transient solve should succeed");

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
fn transient_beam_moment_emits_rotational_snapshots() {
    let mut model = fixture_model(FixtureId::StructuralBeamCantileverEndMomentReference);
    model.sequence.steps = vec![ProcedureStep::structural_transient(
        "beam_moment_transient".to_string(),
    )];
    let result = crate::run_transient_with_options(
        &model,
        ComputeBackend::Cpu,
        experimental_transient_options(),
    )
    .expect("transient beam solve should run");

    assert_eq!(result.time_points_s.len(), result.rotation_snapshots.len());
    assert_eq!(
        result.time_points_s.len(),
        result.angular_velocity_snapshots.len()
    );
    assert_eq!(
        result.time_points_s.len(),
        result.angular_acceleration_snapshots.len()
    );
    assert_eq!(
        result.rotation_snapshots[1].field_id,
        fea_transient_rotation_field_id(1)
    );
    assert_eq!(
        result.angular_velocity_snapshots[1].field_id,
        fea_transient_angular_velocity_field_id(1)
    );
    assert_eq!(
        result.angular_acceleration_snapshots[1].field_id,
        fea_transient_angular_acceleration_field_id(1)
    );
    assert_eq!(result.rotation_snapshots[1].shape, vec![2, 3]);
    assert_eq!(result.angular_velocity_snapshots[1].shape, vec![2, 3]);
    assert_eq!(result.angular_acceleration_snapshots[1].shape, vec![2, 3]);

    let final_rotation = result
        .rotation_snapshots
        .last()
        .and_then(runmat_analysis_core::AnalysisField::as_host_f64)
        .expect("rotation snapshot should be host-backed");
    let max_angular_velocity = result
        .angular_velocity_snapshots
        .iter()
        .filter_map(runmat_analysis_core::AnalysisField::as_host_f64)
        .flat_map(|values| values.iter())
        .map(|value| value.abs())
        .fold(0.0_f64, f64::max);

    assert!(final_rotation.iter().any(|value| value.abs() > 0.0));
    assert!(max_angular_velocity > 0.0);
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
                sampled_time_profile: Vec::new(),
            }),
            ..experimental_thermal_options()
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
        result.energy_storage_rate_snapshots.len()
    );
    assert_eq!(
        result.time_points_s.len(),
        result.axis_boundary_heat_flux_summary_snapshots.len()
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
        result.energy_storage_rate_snapshots[1].field_id,
        fea_thermal_energy_storage_rate_field_id(1)
    );
    assert_eq!(result.temperature_gradient_snapshots[1].shape.len(), 2);
    assert_eq!(result.temperature_gradient_snapshots[1].shape[1], 3);
    assert_eq!(
        result.heat_flux_snapshots[1].shape, result.temperature_gradient_snapshots[1].shape,
        "thermal heat flux should share the element-vector shape"
    );
    assert_eq!(
        result.energy_storage_rate_snapshots[1].shape,
        vec![result.temperature_gradient_snapshots[1].shape[0]],
        "thermal energy storage rate should be an element scalar field"
    );
    assert_eq!(
        result.axis_boundary_heat_flux_summary_snapshots[1].field_id,
        fea_thermal_axis_boundary_heat_flux_summary_field_id(1)
    );
    assert_eq!(
        result.axis_boundary_heat_flux_summary_snapshots[1].shape,
        vec![6],
        "thermal axis-boundary heat-flux summary should report six signed axis-face averages"
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
fn modal_large_fixture_emits_orthogonality_and_separation_diagnostics() {
    let model = fixture_model(FixtureId::ModalLarge);
    let result = crate::run_modal_with_options(
        &model,
        ComputeBackend::Cpu,
        ModalSolveOptions {
            formulation_requirement: runmat_analysis_core::FormulationRequirement::Experimental,
            mode_count: 8,
            solver_mesh: None,
            thermo_mechanical_context: None,
            electro_thermal_context: None,
        },
    )
    .expect("modal large fixture should solve");

    assert!(!result.frequencies_hz.is_empty());
    assert_eq!(result.frequencies_hz.len(), result.mode_shapes.len());
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
            ..experimental_transient_options()
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
            ..experimental_transient_options()
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
    let mut model = fixture_model(FixtureId::ThermoMechanicalKickoff);
    model.sequence.steps = vec![ProcedureStep::coupled(
        "thermomechanical_transient",
        runmat_analysis_core::CoupledPhysicsFamily::Thermomechanical,
        runmat_analysis_core::AnalysisProcedure::TransientResponse,
    )];
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
                sampled_time_profile: Vec::new(),
            }),
            ..experimental_transient_options()
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
fn structural_transient_rejects_unimplemented_electrothermal_formulation() {
    let model = fixture_model(FixtureId::ElectroThermalJouleBenign);
    let error = crate::run_transient_with_options(
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
                sampled_time_profile: Vec::new(),
            }),
            ..experimental_transient_options()
        },
    )
    .expect_err("structural transient must not relabel its profile math as electrothermal");
    assert!(error
        .to_string()
        .contains("do not implement an electrothermal formulation"));
}

#[test]
fn nonlinear_fixture_emits_incremental_payload_and_diagnostics() {
    let mut model = fixture_model(FixtureId::TransientShock);
    model.sequence.steps = vec![ProcedureStep::structural_nonlinear(
        "nonlinear_1".to_string(),
    )];
    let result = crate::run_nonlinear_with_options(
        &model,
        ComputeBackend::Cpu,
        experimental_nonlinear_options(),
    )
    .expect("nonlinear solve should succeed");

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
fn nonlinear_beam_moment_emits_rotation_snapshots() {
    let mut model = fixture_model(FixtureId::StructuralBeamCantileverEndMomentReference);
    model.sequence.steps = vec![ProcedureStep::structural_nonlinear(
        "beam_moment_nonlinear".to_string(),
    )];
    let result = crate::run_nonlinear_with_options(
        &model,
        ComputeBackend::Cpu,
        experimental_nonlinear_options(),
    )
    .expect("nonlinear beam moment solve should succeed");

    assert_eq!(result.load_factors.len(), result.rotation_snapshots.len());
    assert_eq!(
        result.rotation_snapshots[0].field_id,
        fea_nonlinear_rotation_field_id(0)
    );
    assert_eq!(result.rotation_snapshots[0].shape, vec![2, 3]);
    let first_rotation = result.rotation_snapshots[0]
        .as_host_f64()
        .expect("rotation snapshot should be host-backed");
    let final_rotation = result
        .rotation_snapshots
        .last()
        .and_then(runmat_analysis_core::AnalysisField::as_host_f64)
        .expect("final rotation snapshot should be host-backed");

    assert!(final_rotation.iter().any(|value| value.abs() > 0.0));
    assert!(
        final_rotation
            .iter()
            .map(|value| value.abs())
            .fold(0.0_f64, f64::max)
            >= first_rotation
                .iter()
                .map(|value| value.abs())
                .fold(0.0_f64, f64::max)
    );
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
        ..experimental_nonlinear_options()
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
fn thermo_mechanical_nonlinear_emits_coupled_convergence_profile_diagnostic() {
    let mut model = fixture_model(FixtureId::NonlinearLoadPathMix);
    model.sequence.steps = vec![ProcedureStep::coupled(
        "thermomechanical_nonlinear",
        runmat_analysis_core::CoupledPhysicsFamily::Thermomechanical,
        runmat_analysis_core::AnalysisProcedure::Nonlinear,
    )];
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
                sampled_time_profile: Vec::new(),
            }),
            ..experimental_nonlinear_options()
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
        let result = crate::run_nonlinear_with_options(
            &model,
            ComputeBackend::Cpu,
            experimental_nonlinear_options(),
        )
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
fn load_sweep_fixture_uses_solver_mesh_gate() {
    let baseline = run_linear_static_with_single_tetrahedron_mesh(
        &fixture_model(FixtureId::CantileverLinearStatic),
        ComputeBackend::Cpu,
    )
    .expect("baseline solve should succeed");
    let mut model = fixture_model(FixtureId::CantileverLoadSweep);
    for action in &mut model.excitations {
        if let runmat_analysis_core::ExcitationTarget::Region(target) =
            &mut action.excitation_mut().target
        {
            target.region_id = "tip".to_owned();
        }
    }
    let result = run_linear_static_with_single_tetrahedron_mesh(&model, ComputeBackend::Cpu)
        .expect("solve should succeed");

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
    let mut model = fixture_model(FixtureId::CantileverLargeLoadSweep);
    for action in &mut model.excitations {
        if let runmat_analysis_core::ExcitationTarget::Region(target) =
            &mut action.excitation_mut().target
        {
            target.region_id = "tip".to_owned();
        }
    }
    let result = run_linear_static_with_single_tetrahedron_mesh(&model, ComputeBackend::Cpu)
        .expect("solve should succeed");

    assert!(field(&result, FEA_FIELD_STRUCTURAL_DISPLACEMENT).element_count() >= 1536);
    assert!(result
        .diagnostics
        .iter()
        .any(|diag| diag.code == "FEA_SOLVER_METHOD"));
}

#[test]
fn structural_bar_and_beam_reference_fixtures_emit_known_answer_checks() {
    let axial = run_linear_static_with_single_tetrahedron_mesh(
        &fixture_model(FixtureId::StructuralAxialBarReference),
        ComputeBackend::Cpu,
    )
    .expect("axial bar reference should solve");
    let bending = run_linear_static_with_single_tetrahedron_mesh(
        &fixture_model(FixtureId::StructuralBeamBendingReference),
        ComputeBackend::Cpu,
    )
    .expect("beam bending reference should solve");

    assert!(axial.diagnostics.iter().any(|diag| {
        diag.code == "FEA_STRUCTURAL_LINEAR_KNOWN_ANSWER"
            && diag.message.contains("known_answer_coverage_ratio=1")
    }));
    assert!(bending.diagnostics.iter().any(|diag| {
        diag.code == "FEA_STRUCTURAL_LINEAR_KNOWN_ANSWER"
            && diag.message.contains("known_answer_coverage_ratio=1")
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
fn structural_beam_moment_reference_fixtures_emit_closed_form_checks() {
    for (fixture, case) in [
        (
            FixtureId::StructuralBeamCantileverEndMomentReference,
            "cantilever_end_moment",
        ),
        (
            FixtureId::StructuralBeamTorsionReference,
            "cantilever_torsion",
        ),
        (
            FixtureId::StructuralBeamForceAndMomentReference,
            "cantilever_force_and_moment",
        ),
    ] {
        let result = crate::run_linear_static_with_options(
            &fixture_model(fixture),
            ComputeBackend::Cpu,
            experimental_linear_static_options(),
        )
        .expect("beam moment reference should solve");
        assert_eq!(
            field(&result, FEA_FIELD_STRUCTURAL_DISPLACEMENT).shape,
            vec![2, 3]
        );
        assert_eq!(
            field(&result, FEA_FIELD_STRUCTURAL_ROTATION).shape,
            vec![2, 3]
        );
        assert_eq!(
            field(&result, FEA_FIELD_STRUCTURAL_REACTION_MOMENT).shape,
            vec![2, 3]
        );
        assert!(result.diagnostics.iter().any(|diag| {
            diag.code == "FEA_STRUCTURAL_ROTATIONAL_DOF"
                && diag.message.contains("structural_rotational_dof_count=6")
                && diag.message.contains("structural_beam_element_count=1")
        }));
        assert!(result.diagnostics.iter().any(|diag| {
            diag.code == "FEA_STRUCTURAL_MOMENT_BALANCE"
                && diag
                    .message
                    .contains("structural_moment_balance_residual_ratio=")
        }));
        assert!(result.diagnostics.iter().any(|diag| {
            diag.code == "FEA_STRUCTURAL_BEAM_CLOSED_FORM"
                && diag.message.contains(&format!("case={case}"))
                && diag
                    .message
                    .contains("structural_beam_closed_form_rotation_error_ratio=0")
                && diag
                    .message
                    .contains("structural_beam_closed_form_displacement_error_ratio=0")
                && diag
                    .message
                    .contains("structural_beam_closed_form_coverage_ratio=1")
        }));
    }
}

#[test]
fn structural_invalid_moment_without_rotational_dofs_fixture_is_typed() {
    let err = crate::run_linear_static_with_options(
        &fixture_model(FixtureId::StructuralInvalidMomentWithoutRotationalDofs),
        ComputeBackend::Cpu,
        experimental_linear_static_options(),
    )
    .expect_err("moment on displacement-only fixture should fail");
    assert!(err
        .to_string()
        .contains("moment loads require rotational-DOF structural elements"));
}

#[test]
fn moment_load_region_must_resolve_to_rotational_dofs() {
    let mut model = fixture_model(FixtureId::StructuralBeamCantileverEndMomentReference);
    if let runmat_analysis_core::ExcitationTarget::Region(target) =
        &mut model.excitations[0].excitation_mut().target
    {
        target.region_id = "node:99".to_string();
    }

    let err = crate::run_linear_static_with_options(
        &model,
        ComputeBackend::Cpu,
        experimental_linear_static_options(),
    )
    .expect_err("unresolved beam moment target should fail");
    assert!(err
        .to_string()
        .contains("does not resolve to the selected discretization"));
    assert!(err.to_string().contains("region_id=node:99"));
    assert!(err.to_string().contains("load_kind=moment"));
}

#[test]
fn multi_material_fixture_has_distinct_response_profile() {
    let baseline = run_linear_static_with_single_tetrahedron_mesh(
        &fixture_model(FixtureId::CantileverLinearStatic),
        ComputeBackend::Cpu,
    )
    .expect("baseline solve should succeed");
    let multi_material = run_linear_static_with_single_tetrahedron_mesh(
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
