mod thermal_schedule_admission;
mod thermal_storage_history;

use crate::{
    fea_thermal_energy_storage_rate_field_id, fea_thermal_heat_flux_field_id,
    fea_thermal_temperature_gradient_field_id, fea_thermo_mechanical_coupling_residual_field_id,
    fea_thermo_mechanical_displacement_field_id, fea_thermo_mechanical_temperature_field_id,
    fea_thermo_mechanical_thermal_strain_field_id, fea_thermo_mechanical_thermal_stress_field_id,
    fea_thermo_mechanical_von_mises_field_id, fea_transient_acceleration_field_id,
    fea_transient_angular_acceleration_field_id, fea_transient_angular_velocity_field_id,
    fea_transient_damping_dissipation_field_id, fea_transient_energy_balance_residual_field_id,
    fea_transient_external_work_field_id, fea_transient_kinetic_energy_field_id,
    fea_transient_reaction_force_field_id, fea_transient_relative_energy_balance_error_field_id,
    fea_transient_residual_norm_field_id, fea_transient_rotation_field_id,
    fea_transient_strain_energy_field_id, fea_transient_velocity_field_id,
    fixtures::{
        append_fixture_temperature_field, append_fixture_terminal_voltage, fixture_model,
        mixed_solid_beam_model, mixed_solid_beam_shell_model, replace_fixture_excitations,
        FixtureExcitation, FixtureExcitationKind, FixtureId,
    },
    parity::{assert_vectors_within_tolerance, ParityTolerance},
    solve::{
        nonlinear::NonlinearSolveOptions,
        transient::{StructuralTimeStepPolicy, TransientSolveOptions},
    },
    ComputeBackend, FeaContactEnforcement, FeaRunResult, LinearStaticSolveOptions,
    ModalSolveOptions, ModalTruncationPolicy, ThermalProcedureControl, ThermalSolveOptions,
    FEA_FIELD_MODAL_CUMULATIVE_EFFECTIVE_MASS, FEA_FIELD_MODAL_CUMULATIVE_MASS_RATIO,
    FEA_FIELD_MODAL_EFFECTIVE_MASS, FEA_FIELD_MODAL_EIGENVALUE, FEA_FIELD_MODAL_FREQUENCY_HZ,
    FEA_FIELD_MODAL_MODAL_MASS, FEA_FIELD_MODAL_MODAL_STIFFNESS, FEA_FIELD_MODAL_M_ORTHOGONALITY,
    FEA_FIELD_MODAL_PARTICIPATION_FACTOR, FEA_FIELD_MODAL_RELATIVE_FREQUENCY_SEPARATION,
    FEA_FIELD_MODAL_RESIDUAL_NORM, FEA_FIELD_MODAL_TOTAL_FREE_MASS,
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
                target: runmat_analysis_core::RegionTarget::authored(
                    crate::fixtures::fixture_target_selection(region),
                ),
            },
        },
    )
}

fn static_excitations(
    model: &runmat_analysis_core::AnalysisModel,
) -> crate::excitation::RealizedExcitations {
    crate::excitation::realize_step_excitations(
        model,
        &model.sequence.steps[0],
        crate::excitation::ExcitationSamplePoint::Static,
    )
    .expect("test model excitations should realize at the static sample")
}

use runmat_analysis_core::{AnalysisProcedure, ProcedureStep};
use runmat_meshing_core::{
    ElementOrder, PersistentEntityId, PersistentEntityKind, SolverMeshArtifact,
};

mod column_buckling;
mod connector_qualification;
mod prestressed_frequency;
mod random_response_portable;
mod random_response_qualification;
mod response_spectrum_portable;
mod solid_static_curved_pressure;
mod solid_static_patch;
mod solid_static_pressure_patch;
mod structural_static_surface;
mod transient_physical;

fn baseline_linear_static_options() -> LinearStaticSolveOptions {
    LinearStaticSolveOptions {
        ..LinearStaticSolveOptions::default()
    }
}

fn baseline_modal_options() -> ModalSolveOptions {
    ModalSolveOptions {
        ..ModalSolveOptions::default()
    }
}

fn baseline_thermal_options() -> ThermalSolveOptions {
    ThermalSolveOptions {
        ..ThermalSolveOptions::default()
    }
}

fn baseline_transient_options() -> TransientSolveOptions {
    TransientSolveOptions {
        ..TransientSolveOptions::default()
    }
}

fn configure_thermo_mechanical_excitation(
    model: &mut runmat_analysis_core::AnalysisModel,
    temperature_delta_k: f64,
    expansion_coefficient_per_k: f64,
) {
    let reference_temperature_k = 293.15;
    model.thermo_mechanical = Some(runmat_analysis_core::ThermoMechanicalDomain {
        reference_temperature: runmat_analysis_core::Temperature::new(reference_temperature_k),
    });
    for material in &mut model.materials {
        let thermal = material
            .thermal
            .as_mut()
            .expect("thermo-mechanical fixture declares thermal capability");
        thermal.reference_temperature_k =
            runmat_types::physical::Temperature::new(reference_temperature_k);
        thermal.expansion_coefficient_per_k =
            runmat_types::physical::ThermalExpansion::new(expansion_coefficient_per_k);
    }
    append_fixture_temperature_field(
        model,
        "coupled_temperature",
        reference_temperature_k + temperature_delta_k,
        runmat_analysis_core::ExcitationTarget::WholeModel,
    );
}

fn configure_electro_thermal_excitation(model: &mut runmat_analysis_core::AnalysisModel) {
    let reference_temperature_k = 293.15;
    model.electro_thermal = Some(runmat_analysis_core::ElectroThermalDomain {
        reference_temperature: runmat_analysis_core::Temperature::new(reference_temperature_k),
    });
    for material in &mut model.materials {
        material.electrical = Some(runmat_analysis_core::MaterialElectricalModel {
            reference_temperature_k: runmat_types::physical::Temperature::new(
                reference_temperature_k,
            ),
            conductivity_s_per_m: runmat_types::physical::ElectricalConductivity::new(3.5e7),
            resistive_heating_coefficient: runmat_types::physical::InverseTemperature::new(4.0e-4),
            relative_permittivity: runmat_types::physical::RelativePermittivity::new(1.0),
            relative_permeability: runmat_types::physical::RelativePermeability::new(1.0),
            conductivity_frequency_response: Vec::new(),
        });
    }
    append_fixture_terminal_voltage(
        model,
        "coupled_terminal_voltage",
        36.0,
        runmat_analysis_core::ExcitationTarget::Region(
            runmat_analysis_core::RegionTarget::boundary(
                runmat_meshing_core::fixtures::fixture_face("tip"),
            ),
        ),
    );
}

fn baseline_nonlinear_options() -> NonlinearSolveOptions {
    NonlinearSolveOptions {
        ..NonlinearSolveOptions::default()
    }
}

fn nodal_history_observation(
    id: &str,
    quantity: runmat_analysis_core::ObservationQuantity,
    sampling: runmat_analysis_core::SamplingPolicy,
) -> runmat_analysis_core::ObservationRequest {
    history_observation(
        id,
        quantity,
        sampling,
        runmat_analysis_core::FieldLocation::Node,
    )
}

fn history_observation(
    id: &str,
    quantity: runmat_analysis_core::ObservationQuantity,
    sampling: runmat_analysis_core::SamplingPolicy,
    location: runmat_analysis_core::FieldLocation,
) -> runmat_analysis_core::ObservationRequest {
    runmat_analysis_core::ObservationRequest::History(runmat_analysis_core::ObservationSpec {
        observation_id: runmat_analysis_core::ObservationId(id.to_owned()),
        quantity,
        value_shape: quantity.canonical_value_shape(),
        target: runmat_analysis_core::ObservationTarget::WholeModel,
        source_location: location,
        output_location: location,
        recovery: runmat_analysis_core::RecoveryPolicy::Native,
        component: runmat_analysis_core::ObservationComponent::All,
        frame: runmat_analysis_core::CoordinateFrame::Global,
        sign_convention: runmat_analysis_core::SignConvention::Native,
        complex_representation: runmat_analysis_core::ComplexRepresentation::Real,
        sampling,
        reduction: None,
        storage: runmat_analysis_core::ObservationStoragePolicy {
            precision: runmat_analysis_core::SignalPrecision::Real64,
            compression: runmat_analysis_core::CompressionPolicy::Lossless,
            retention: runmat_analysis_core::RetentionPolicy::Canonical,
            partial_access: runmat_analysis_core::PartialAccessPolicy::ManifestAndChunks,
        },
        purpose: runmat_analysis_core::ObservationPurpose::Publication,
        cost: runmat_analysis_core::ObservationCostPolicy::UpperBound(
            runmat_analysis_core::ObservationCostEstimate {
                operation_count: 1_000_000_000,
                peak_resident_bytes: 1_000_000_000,
                stored_bytes: 1_000_000_000,
                transfer_bytes: 1_000_000_000,
                collective_count: 0,
            },
        ),
    })
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
            ..baseline_linear_static_options()
        },
    )
}

#[test]
fn direct_physical_static_rejects_accuracy_memory_ceiling_before_assembly() {
    let mut model = fixture_model(FixtureId::CantileverLinearStatic);
    model.sequence.steps[0]
        .accuracy
        .resources
        .maximum_resident_bytes = Some(1);
    let error = run_linear_static_with_single_tetrahedron_mesh(&model, ComputeBackend::Cpu)
        .expect_err("direct physical solve must admit its resource plan before assembly");
    assert!(matches!(
        error,
        crate::FeaRunError::Resource(
            crate::planning::resource::ResourcePlanningError::AccuracyCeiling {
                resource: "resident memory bytes",
                maximum: 1,
                ..
            }
        )
    ));
}

#[test]
fn direct_connector_static_without_connector_observation_preflights_memory() {
    let mut model = fixture_model(FixtureId::StructuralBeamCantileverEndMomentReference);
    model
        .connectors
        .push(runmat_analysis_core::AnalysisConnector {
            connector_id: "tip_support_spring".into(),
            kind: runmat_analysis_core::ConnectorKind::IsotropicGroundSpring {
                target: runmat_analysis_core::RegionTarget::point("node:2"),
                stiffness: runmat_analysis_core::Stiffness::new(1.0e6),
            },
        });
    model.sequence.steps[0].observations.clear();
    let admitted_limit = model.sequence.steps[0]
        .accuracy
        .resources
        .maximum_resident_bytes;
    model.sequence.steps[0]
        .accuracy
        .resources
        .maximum_resident_bytes = Some(1);
    let error = crate::run_linear_static_with_options(
        &model,
        ComputeBackend::Cpu,
        baseline_linear_static_options(),
    )
    .expect_err("connector-bearing direct solve must preflight before assembly");
    assert!(matches!(
        error,
        crate::FeaRunError::Resource(
            crate::planning::resource::ResourcePlanningError::AccuracyCeiling {
                resource: "resident memory bytes",
                maximum: 1,
                ..
            }
        )
    ));
    model.sequence.steps[0]
        .accuracy
        .resources
        .maximum_resident_bytes = admitted_limit;
    let result = crate::run_linear_static_with_options(
        &model,
        ComputeBackend::Cpu,
        baseline_linear_static_options(),
    )
    .expect("the admitted connector-bearing direct solve must complete");
    assert_eq!(
        result
            .structural_connector_result
            .as_ref()
            .expect("connector response")
            .responses
            .len(),
        1,
    );
}

#[test]
fn direct_solid_connector_static_preflights_memory() {
    let mut model = fixture_model(FixtureId::CantileverLinearStatic);
    let mesh = single_tetrahedron_solver_mesh_for_model(&model);
    let target_node = mesh.topology.nodes[1].node_id;
    model
        .connectors
        .push(runmat_analysis_core::AnalysisConnector {
            connector_id: "solid_tip_spring".into(),
            kind: runmat_analysis_core::ConnectorKind::IsotropicGroundSpring {
                target: runmat_analysis_core::RegionTarget::point(format!("node:{target_node}")),
                stiffness: runmat_analysis_core::Stiffness::new(1.0e6),
            },
        });
    model.sequence.steps[0].observations.clear();
    let admitted_limit = model.sequence.steps[0]
        .accuracy
        .resources
        .maximum_resident_bytes;
    model.sequence.steps[0]
        .accuracy
        .resources
        .maximum_resident_bytes = Some(1);
    let error = run_linear_static_with_single_tetrahedron_mesh(&model, ComputeBackend::Cpu)
        .expect_err("solid connector must preflight before assembly");
    assert!(matches!(
        error,
        crate::FeaRunError::Resource(
            crate::planning::resource::ResourcePlanningError::AccuracyCeiling {
                resource: "resident memory bytes",
                maximum: 1,
                ..
            }
        )
    ));
    model.sequence.steps[0]
        .accuracy
        .resources
        .maximum_resident_bytes = admitted_limit;
    let result = run_linear_static_with_single_tetrahedron_mesh(&model, ComputeBackend::Cpu)
        .expect("admitted solid connector must solve");
    assert_eq!(
        result
            .structural_connector_result
            .as_ref()
            .expect("connector response")
            .plan,
        crate::assembly::explicit_connectors::compile_model_connector_plan(&model, Some(&mesh), 0)
            .unwrap(),
    );
    assert_eq!(
        result
            .structural_connector_result
            .as_ref()
            .expect("connector response")
            .responses
            .len(),
        1,
    );
}

fn assert_close(actual: f64, expected: f64, tolerance: f64) {
    assert!(
        (actual - expected).abs() <= tolerance,
        "actual={actual} expected={expected} tolerance={tolerance}",
    );
}

#[test]
fn canonical_cantilever_benchmark_runs() {
    let model = fixture_model(FixtureId::CantileverLinearStatic);
    let result = run_linear_static_with_single_tetrahedron_mesh(&model, ComputeBackend::Cpu)
        .expect("solve should succeed");

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
        vec![4, 3]
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
            truncation: ModalTruncationPolicy::requested(2),
            solver_mesh: Some(mesh.clone()),
            ..baseline_modal_options()
        },
    )
    .expect("modal solve should consume the canonical mesh");
    assert_eq!(modal.mode_shapes[0].shape, vec![4, 3]);

    let mut thermal_model = model.clone();
    thermal_model.thermal = Some(runmat_analysis_core::ThermalDomain {
        reference_temperature: runmat_analysis_core::Temperature::new(293.15),
    });
    thermal_model.sequence.steps = vec![ProcedureStep::thermal_transient("thermal")];
    let thermal = crate::run_thermal_with_options(
        &thermal_model,
        ComputeBackend::Cpu,
        ThermalSolveOptions {
            procedure: ThermalProcedureControl::BackwardEuler { snapshot_count: 2 },
            solver_mesh: Some(mesh.clone()),
            ..baseline_thermal_options()
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
            time_step_policy: StructuralTimeStepPolicy::Fixed { interval_count: 2 },
            solver_mesh: Some(mesh.clone()),
            ..baseline_transient_options()
        },
    )
    .expect("transient solve should consume the canonical mesh");
    assert_eq!(transient.displacement_snapshots[0].shape, vec![4, 3]);
    assert_eq!(
        transient.run.spatial_projection_evidence.len(),
        transient.time_points_s.len()
    );
    for (evidence, time_s) in transient
        .run
        .spatial_projection_evidence
        .iter()
        .zip(&transient.time_points_s)
    {
        assert_eq!(
            evidence.sample,
            crate::excitation::ProjectionSample::TimeSeconds {
                value: *time_s,
                approach: runmat_analysis_core::SignalSampleApproach::At,
            }
        );
        assert!(evidence.accepted);
    }

    let mut nonlinear_model = model.clone();
    nonlinear_model.sequence.steps = vec![ProcedureStep::structural_nonlinear("nonlinear")];
    let nonlinear = crate::run_nonlinear_with_options(
        &nonlinear_model,
        ComputeBackend::Cpu,
        NonlinearSolveOptions {
            increment_count: 2,
            solver_mesh: Some(mesh.clone()),
            ..baseline_nonlinear_options()
        },
    )
    .expect("nonlinear solve should consume the canonical mesh");
    assert_eq!(nonlinear.displacement_snapshots[0].shape, vec![4, 3]);
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
        baseline_linear_static_options(),
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
                    target: runmat_analysis_core::RegionTarget::boundary(
                        runmat_meshing_core::fixtures::fixture_face("tip"),
                    ),
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
        baseline_linear_static_options(),
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
        &static_excitations(&inline_model),
        None,
        crate::assembly::StructuralFormulationRequirement::SyntheticOperator,
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
        &static_excitations(&signal_model),
        None,
        crate::assembly::StructuralFormulationRequirement::SyntheticOperator,
    )
    .expect("signal-backed prescribed value should assemble");
    assert_eq!(
        referenced.operator.constrained_dofs(),
        inline.operator.constrained_dofs()
    );
    assert_eq!(referenced.operator.load(), inline.operator.load());
}

#[test]
fn assembly_summary_reports_structural_dof_layout_metrics() {
    let model = fixture_model(FixtureId::StructuralBeamCantileverEndMomentReference);
    let summary = crate::assembly::assemble_structural_system(
        &model,
        &static_excitations(&model),
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
        &static_excitations(&model),
        None,
        crate::assembly::StructuralFormulationRequirement::PhysicalDiscretization,
    )
    .expect("beam operator should assemble");
    let layout = &summary.structural_dof_layout;
    let node = 0usize;
    let mass = &summary.operator.operators().mass().diagonal_values();

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
        baseline_linear_static_options(),
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

// witness: FEA-STRUCTURAL-BEAM-001/physical_beam_mixed_dofs @
//   linear_structural/beam|shell/numerical_kernel; coverage: partial;
//   boundary: A mixed beam and shell model solves and recovers each formulation.
#[test]
fn mixed_beam_shell_model_solves_and_recovers_each_formulation() {
    let mut model = fixture_model(FixtureId::StructuralBeamCantileverEndMomentReference);
    let shell_model = fixture_model(FixtureId::StructuralShellPlateMomentReference);
    model
        .material_assignments
        .extend(shell_model.material_assignments.clone());
    let shell = shell_model.structural.expect("shell topology");
    let structural = model.structural.as_mut().expect("beam topology");
    structural.nodes.push(shell.nodes[2].clone());
    structural.elements.push(shell.elements[0].clone());
    structural.shell_sections = shell.shell_sections;

    let result = crate::run_linear_static_with_options(
        &model,
        ComputeBackend::Cpu,
        baseline_linear_static_options(),
    )
    .expect("the shared-node beam-shell model must complete one admitted solve");

    assert_eq!(
        field(&result, FEA_FIELD_STRUCTURAL_BEAM_AXIAL_FORCE).shape,
        vec![1]
    );
    assert_eq!(
        field(&result, FEA_FIELD_STRUCTURAL_SHELL_BENDING_MOMENT).shape,
        vec![1, 3]
    );
    assert!(result.field(FEA_FIELD_STRUCTURAL_STRESS).is_none());
}

#[test]
fn mixed_solid_beam_pipeline_is_deterministic_and_recovers_both_formulations() {
    let mut mesh = single_tetrahedron_solver_mesh();
    let attached_coordinates = mesh.topology.nodes[0].coordinates_m.canonical_value();
    mesh.topology.nodes[0]
        .provenance
        .push(PersistentEntityId::new(
            PersistentEntityKind::Vertex,
            "beam_root".into(),
            vec!["root".into()],
        ));
    mesh.topology.nodes[0].provenance.sort();
    add_face_region(&mut mesh, 1, "solid_load");
    reseal_solver_mesh(&mut mesh);
    let model = mixed_solid_beam_model(attached_coordinates);
    let options = LinearStaticSolveOptions {
        solver_mesh: Some(mesh),
        solver_mesh_artifact_path: Some("mixed-solid-beam.rmmesh".into()),
        ..baseline_linear_static_options()
    };

    let first = crate::run_linear_static_with_options(&model, ComputeBackend::Cpu, options.clone())
        .expect("mixed solid-beam pipeline");
    let second = crate::run_linear_static_with_options(&model, ComputeBackend::Cpu, options)
        .expect("deterministic replay");

    assert_eq!(first.fields, second.fields);
    assert_eq!(first.diagnostics, second.diagnostics);
    assert_eq!(
        field(&first, FEA_FIELD_STRUCTURAL_DISPLACEMENT).shape,
        vec![5, 3]
    );
    assert_eq!(
        field(&first, FEA_FIELD_STRUCTURAL_ROTATION).shape,
        vec![5, 3]
    );
    assert_eq!(field(&first, FEA_FIELD_STRUCTURAL_STRESS).shape, vec![1, 6]);
    assert_eq!(
        field(&first, FEA_FIELD_STRUCTURAL_BEAM_AXIAL_FORCE).shape,
        vec![1]
    );
    assert!(host_field(&first, FEA_FIELD_STRUCTURAL_STRESS)
        .iter()
        .all(|value| value.is_finite()));
    assert!(host_field(&first, FEA_FIELD_STRUCTURAL_BEAM_AXIAL_FORCE)[0].abs() > 0.0);
}

#[test]
fn conformal_solid_beam_shell_pipeline_recovers_every_formulation() {
    let mut mesh = single_tetrahedron_solver_mesh();
    let attached_coordinates = mesh.topology.nodes[0].coordinates_m.canonical_value();
    mesh.topology.nodes[0]
        .provenance
        .push(PersistentEntityId::new(
            PersistentEntityKind::Vertex,
            "beam_root".into(),
            vec!["root".into()],
        ));
    mesh.topology.nodes[0].provenance.sort();
    add_face_region(&mut mesh, 1, "solid_load");
    reseal_solver_mesh(&mut mesh);
    let model = mixed_solid_beam_shell_model(attached_coordinates);
    let options = LinearStaticSolveOptions {
        solver_mesh: Some(mesh),
        solver_mesh_artifact_path: Some("mixed-solid-beam-shell.rmmesh".into()),
        ..baseline_linear_static_options()
    };

    let result = crate::run_linear_static_with_options(&model, ComputeBackend::Cpu, options)
        .expect("conformal solid-beam-shell pipeline");

    assert_eq!(
        field(&result, FEA_FIELD_STRUCTURAL_DISPLACEMENT).shape,
        vec![7, 3]
    );
    assert_eq!(
        field(&result, FEA_FIELD_STRUCTURAL_ROTATION).shape,
        vec![7, 3]
    );
    assert_eq!(
        field(&result, FEA_FIELD_STRUCTURAL_STRESS).shape,
        vec![1, 6]
    );
    assert_eq!(
        field(&result, FEA_FIELD_STRUCTURAL_BEAM_AXIAL_FORCE).shape,
        vec![1]
    );
    assert_eq!(
        field(&result, FEA_FIELD_STRUCTURAL_SHELL_MEMBRANE_FORCE).shape,
        vec![1, 3]
    );
    assert_eq!(
        field(&result, FEA_FIELD_STRUCTURAL_SHELL_BENDING_MOMENT).shape,
        vec![1, 3]
    );
    assert!(host_field(&result, FEA_FIELD_STRUCTURAL_STRESS)
        .iter()
        .all(|value| value.is_finite()));
    assert!(host_field(&result, FEA_FIELD_STRUCTURAL_BEAM_AXIAL_FORCE)[0].is_finite());
    assert!(
        host_field(&result, FEA_FIELD_STRUCTURAL_SHELL_MEMBRANE_FORCE)
            .iter()
            .all(|value| value.is_finite())
    );
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
                coordinates_m: runmat_types::physical::Quantity::<
                    runmat_types::physical::kind::Position,
                    _,
                >::new([0.0, 0.0, 0.0]),
                solid_attachment: None,
            },
            runmat_analysis_core::StructuralNode {
                node_id: 2,
                coordinates_m: runmat_types::physical::Quantity::<
                    runmat_types::physical::kind::Position,
                    _,
                >::new([length_m, 0.0, 0.0]),
                solid_attachment: None,
            },
        ],
        elements: vec![runmat_analysis_core::StructuralElement {
            element_id: "beam_1".to_string(),
            region_id: runmat_meshing_core::fixtures::fixture_edge("beam_span"),
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
            area_m2: runmat_types::physical::Quantity::<runmat_types::physical::kind::Area, _>::new(
                2.0e-4,
            ),
            iy_m4: runmat_types::physical::Quantity::<
                runmat_types::physical::kind::SecondMomentOfArea,
                _,
            >::new(1.6e-9),
            iz_m4: runmat_types::physical::Quantity::<
                runmat_types::physical::kind::SecondMomentOfArea,
                _,
            >::new(iz_m4),
            torsion_j_m4: runmat_types::physical::Quantity::<
                runmat_types::physical::kind::SecondMomentOfArea,
                _,
            >::new(2.4e-9),
            outer_fiber_y_m: runmat_types::physical::Quantity::<
                runmat_types::physical::kind::Length,
                _,
            >::new(0.01),
            outer_fiber_z_m: runmat_types::physical::Quantity::<
                runmat_types::physical::kind::Length,
                _,
            >::new(0.005),
            torsion_outer_radius_m: runmat_types::physical::Quantity::<
                runmat_types::physical::kind::Length,
                _,
            >::new(0.011_180_339_887_498_949),
        }],
        shell_sections: Vec::new(),
    });
    model.material_assignments = vec![runmat_analysis_core::MaterialAssignment {
        target: runmat_analysis_core::PropertyRegionTarget::beam(
            runmat_meshing_core::fixtures::fixture_edge("beam_span"),
        ),
        expected_material_id: "mat_steel".into(),
        assigned_material_id: "mat_steel".into(),
        confidence: runmat_analysis_core::EvidenceConfidence::Verified,
    }];
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
        baseline_linear_static_options(),
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
    let balance = result
        .diagnostics
        .iter()
        .find(|diag| diag.code == "FEA_STRUCTURAL_MOMENT_BALANCE")
        .expect("moment balance diagnostic");
    let value = |key: &str| -> f64 {
        balance
            .message
            .split_whitespace()
            .find_map(|part| part.strip_prefix(key)?.strip_prefix('='))
            .unwrap_or_else(|| panic!("{key} missing from {}", balance.message))
            .parse()
            .unwrap()
    };
    for key in [
        "structural_reaction_moment_norm_n_m",
        "structural_moment_requested_norm_n_m",
        "structural_moment_realized_norm_n_m",
    ] {
        assert!((value(key) - 1.25).abs() <= 1.25e-12, "{}", balance.message);
    }
    assert!((value("structural_moment_realization_ratio") - 1.0).abs() <= 1.0e-12);
    assert!(value("structural_moment_balance_residual_ratio") <= 1.0e-10);
}

#[test]
fn grounded_explicit_spring_changes_the_completed_physical_solution() {
    let mut baseline_model = fixture_model(FixtureId::StructuralBeamCantileverEndMomentReference);
    replace_fixture_excitations(
        &mut baseline_model,
        vec![FixtureExcitation {
            id: "tip_force".into(),
            region_id: "node:2".into(),
            kind: FixtureExcitationKind::Force([0.0, -100.0, 0.0]),
        }],
    );
    let baseline = crate::run_linear_static_with_options(
        &baseline_model,
        ComputeBackend::Cpu,
        baseline_linear_static_options(),
    )
    .expect("baseline beam solve");

    let mut supported_model = baseline_model;
    supported_model.connectors = vec![runmat_analysis_core::AnalysisConnector {
        connector_id: "tip_support_spring".into(),
        kind: runmat_analysis_core::ConnectorKind::IsotropicGroundSpring {
            target: runmat_analysis_core::RegionTarget::point("node:2"),
            stiffness: runmat_analysis_core::Stiffness::new(1.0e6),
        },
    }];
    supported_model.sequence.steps[0].observations =
        vec![runmat_analysis_core::ObservationRequest::Probe(
            runmat_analysis_core::ObservationSpec {
                observation_id: runmat_analysis_core::ObservationId("tip_support_force".into()),
                quantity: runmat_analysis_core::ObservationQuantity::ReactionForce,
                value_shape: runmat_analysis_core::ObservationValueShape::Vector3,
                target: runmat_analysis_core::ObservationTarget::Connector {
                    connector_id: "tip_support_spring".into(),
                },
                source_location: runmat_analysis_core::FieldLocation::Connector,
                output_location: runmat_analysis_core::FieldLocation::Connector,
                recovery: runmat_analysis_core::RecoveryPolicy::Native,
                component: runmat_analysis_core::ObservationComponent::All,
                frame: runmat_analysis_core::CoordinateFrame::Global,
                sign_convention: runmat_analysis_core::SignConvention::Native,
                complex_representation: runmat_analysis_core::ComplexRepresentation::Real,
                sampling: runmat_analysis_core::SamplingPolicy::Final,
                reduction: None,
                storage: runmat_analysis_core::ObservationStoragePolicy {
                    precision: runmat_analysis_core::SignalPrecision::Real64,
                    compression: runmat_analysis_core::CompressionPolicy::Lossless,
                    retention: runmat_analysis_core::RetentionPolicy::Canonical,
                    partial_access: runmat_analysis_core::PartialAccessPolicy::ManifestAndChunks,
                },
                purpose: runmat_analysis_core::ObservationPurpose::Publication,
                cost: runmat_analysis_core::ObservationCostPolicy::Automatic,
            },
        )];
    let mut rejected_model = supported_model.clone();
    rejected_model.sequence.steps[0]
        .accuracy
        .resources
        .maximum_resident_bytes = Some(1);
    let error = crate::run_linear_static_with_options(
        &rejected_model,
        ComputeBackend::Cpu,
        baseline_linear_static_options(),
    )
    .expect_err("connector observation must be costed before assembly");
    assert!(matches!(
        error,
        crate::FeaRunError::Resource(
            crate::planning::resource::ResourcePlanningError::AccuracyCeiling {
                resource: "resident memory bytes",
                maximum: 1,
                ..
            }
        )
    ));
    let supported = crate::run_linear_static_with_options(
        &supported_model,
        ComputeBackend::Cpu,
        baseline_linear_static_options(),
    )
    .expect("spring-supported beam solve");

    let baseline_displacement = host_field(&baseline, FEA_FIELD_STRUCTURAL_DISPLACEMENT)[4].abs();
    let supported_displacement = host_field(&supported, FEA_FIELD_STRUCTURAL_DISPLACEMENT)[4].abs();
    assert!(
        supported_displacement < baseline_displacement,
        "supported={supported_displacement} baseline={baseline_displacement}"
    );
    let connector_result = supported
        .structural_connector_result
        .as_ref()
        .expect("the executed connector plan must publish physical response");
    assert_eq!(
        connector_result.plan,
        crate::assembly::explicit_connectors::compile_model_connector_plan(
            &supported_model,
            None,
            0,
        )
        .unwrap(),
    );
    let expected_force = connector_result.responses[0].resultant_force_newtons();
    let observation = supported
        .observations
        .first()
        .expect("the exact connector observation must be published");
    assert_eq!(observation.observation_id.0, "tip_support_force");
    assert_eq!(
        observation.field.as_host_f64().unwrap(),
        expected_force.as_slice()
    );
}

#[test]
fn thermo_mechanical_linear_static_emits_coupled_fields() {
    let mut model = fixture_model(FixtureId::ThermoMechanicalKickoff);
    model.material_assignments.truncate(1);
    configure_thermo_mechanical_excitation(&mut model, 65.0, 1.2e-5);
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
            ..baseline_linear_static_options()
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
        runmat_analysis_core::SpatialOperatorImplementation::IntegratedPrescribedSolidThermoelasticLoad
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
        vec![1]
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
    let temperature = field(&result, &fea_thermo_mechanical_temperature_field_id(0));
    assert!(temperature
        .as_host_f64()
        .unwrap()
        .iter()
        .all(|value| (*value - 358.15).abs() < 1.0e-12));
    let thermal_strain = field(&result, &fea_thermo_mechanical_thermal_strain_field_id(0));
    assert!(thermal_strain.as_host_f64().unwrap()[..3]
        .iter()
        .all(|value| (*value - 7.8e-4).abs() < 1.0e-15));
    let eigenstress = field(&result, &fea_thermo_mechanical_thermal_stress_field_id(0));
    for stress in &eigenstress.as_host_f64().unwrap()[..3] {
        assert!(
            (*stress - 3.9e8).abs() < 3.9e8 * 1.0e-12,
            "unexpected element thermal eigenstress {stress}"
        );
    }
    assert!(result.diagnostics.iter().any(|diagnostic| {
        diagnostic.code == "FEA_TM_CONSISTENCY"
            && diagnostic
                .message
                .contains("element_owned_thermal_eigenstress")
    }));
    assert!(result.diagnostics.iter().any(|diagnostic| {
        diagnostic.code == "FEA_TM_TEMPERATURE_FIELD"
            && diagnostic
                .message
                .contains("temperature_field_basis=element_constant")
    }));
    let density = host_field(&result, FEA_FIELD_STRUCTURAL_STRAIN_ENERGY_DENSITY)[0];
    let total_energy = host_field(&result, FEA_FIELD_STRUCTURAL_TOTAL_STRAIN_ENERGY)[0];
    assert!(density >= 0.0);
    assert_close(
        total_energy,
        density / 6.0,
        density.abs() * 1.0e-10 + 1.0e-9,
    );
}

#[test]
fn thermo_mechanical_static_region_temperature_overrides_global_field_on_exact_solid() {
    let mut model = fixture_model(FixtureId::ThermoMechanicalKickoff);
    model.material_assignments.truncate(1);
    configure_thermo_mechanical_excitation(&mut model, 65.0, 1.2e-5);
    let region_id = model.material_assignments[0]
        .selected_region_id()
        .unwrap()
        .to_owned();
    append_fixture_temperature_field(
        &mut model,
        "region_temperature",
        313.15,
        runmat_analysis_core::ExcitationTarget::Region(runmat_analysis_core::RegionTarget::volume(
            region_id,
        )),
    );
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
            ..baseline_linear_static_options()
        },
    )
    .expect("exact-region prescribed temperature should solve physically");
    assert_eq!(
        result.identity.realization().components[1].sources,
        vec![
            runmat_analysis_core::SourceImplementation::PrescribedTemperatureDeltaProfile,
            runmat_analysis_core::SourceImplementation::PersistentRegionElementTemperatureMap,
        ]
    );
    assert_eq!(
        field(&result, &fea_thermo_mechanical_temperature_field_id(0))
            .as_host_f64()
            .unwrap(),
        &[313.15]
    );
    for strain in &field(&result, &fea_thermo_mechanical_thermal_strain_field_id(0))
        .as_host_f64()
        .unwrap()[..3]
    {
        assert!((*strain - 2.4e-4).abs() < 1.0e-15);
    }
}

#[test]
fn linear_static_structural_step_leaves_declared_thermomechanical_domain_inactive() {
    let mut model = fixture_model(FixtureId::CantileverLinearStatic);
    let baseline = run_linear_static_with_single_tetrahedron_mesh(&model, ComputeBackend::Cpu)
        .expect("uncoupled structural baseline");
    // A domain declaration alone is inactive. A prescribed temperature field
    // is an authored excitation and must not be silently dropped by this step.
    model.thermo_mechanical = Some(runmat_analysis_core::ThermoMechanicalDomain {
        reference_temperature: runmat_analysis_core::Temperature::new(293.15),
    });
    let mesh = single_tetrahedron_solver_mesh_for_model(&model);
    let result = crate::run_linear_static_with_options(
        &model,
        ComputeBackend::Cpu,
        LinearStaticSolveOptions {
            solver_mesh: Some(mesh),
            ..baseline_linear_static_options()
        },
    )
    .expect("a declared coupling domain does not activate a structural-only step");
    assert_eq!(
        result.identity.family(),
        runmat_analysis_core::PhysicsFamily::Structural
    );
    assert_eq!(result.identity, baseline.identity);
    assert_eq!(
        field(&result, FEA_FIELD_STRUCTURAL_DISPLACEMENT),
        field(&baseline, FEA_FIELD_STRUCTURAL_DISPLACEMENT),
        "inactive thermal declarations must not change structural displacement"
    );

    configure_thermo_mechanical_excitation(&mut model, 65.0, 1.2e-5);
    let mesh = single_tetrahedron_solver_mesh_for_model(&model);
    let error = crate::run_linear_static_with_options(
        &model,
        ComputeBackend::Cpu,
        LinearStaticSolveOptions {
            solver_mesh: Some(mesh),
            ..baseline_linear_static_options()
        },
    )
    .expect_err("a structural-only step must not discard an active temperature field");
    assert!(matches!(error, crate::FeaRunError::InvalidModel(message)
            if message.contains("active temperature-field coupling (coupled_temperature)")
                && message.contains("realizes Coupled(Thermomechanical)")));
}

#[test]
fn linear_static_rejects_structural_step_that_would_drop_a_terminal_voltage() {
    let mut model = fixture_model(FixtureId::CantileverLinearStatic);
    append_fixture_terminal_voltage(
        &mut model,
        "uncoupled_terminal_voltage",
        36.0,
        runmat_analysis_core::ExcitationTarget::Region(
            runmat_analysis_core::RegionTarget::boundary(
                runmat_meshing_core::fixtures::fixture_face("tip"),
            ),
        ),
    );
    let mesh = single_tetrahedron_solver_mesh_for_model(&model);
    let error = crate::run_linear_static_with_options(
        &model,
        ComputeBackend::Cpu,
        LinearStaticSolveOptions {
            solver_mesh: Some(mesh),
            ..baseline_linear_static_options()
        },
    )
    .expect_err("a structural step must not silently drop an authored terminal voltage");

    let message = error.to_string();
    assert!(
        message.contains("terminal-voltage coupling (uncoupled_terminal_voltage)"),
        "{message}"
    );
    assert!(
        message.contains("realizes Coupled(Electrothermal)"),
        "{message}"
    );
}

#[test]
fn linear_static_rejects_thermomechanical_step_on_explicit_connectivity() {
    let mut model = fixture_model(FixtureId::StructuralBeamCantileverEndMomentReference);
    configure_thermo_mechanical_excitation(&mut model, 65.0, 1.2e-5);
    model.sequence.steps = vec![ProcedureStep::coupled(
        "thermomechanical_explicit_static",
        runmat_analysis_core::CoupledPhysicsFamily::Thermomechanical,
        runmat_analysis_core::AnalysisProcedure::Static,
    )];
    let error = crate::run_linear_static_with_options(
        &model,
        ComputeBackend::Cpu,
        baseline_linear_static_options(),
    )
    .expect_err("explicit connectivity has no temperature-load path to apply the profile");
    let message = error.to_string();
    assert!(
        message.contains("physical solid temperature-load assembly"),
        "{message}"
    );
}

#[test]
fn transient_rejects_thermomechanical_identity_before_assembly() {
    let mut model = fixture_model(FixtureId::TransientLong);
    model.sequence.steps = vec![ProcedureStep::coupled(
        "thermomechanical_transient_without_context",
        runmat_analysis_core::CoupledPhysicsFamily::Thermomechanical,
        runmat_analysis_core::AnalysisProcedure::TransientResponse,
    )];
    let error = crate::run_transient_with_options(
        &model,
        ComputeBackend::Cpu,
        baseline_transient_options(),
    )
    .expect_err("thermomechanical transient requires a solver-owned coupled thermal state");

    assert!(matches!(
        error,
        crate::FeaRunError::FormulationAvailability(rejection)
            if rejection.reason
                == runmat_analysis_core::FormulationAdmissionRejectionReason::UnsupportedFamilyProcedure
    ));
}

#[test]
fn nonlinear_rejects_explicit_connectivity_that_drops_thermomechanical_coupling() {
    let mut model = fixture_model(FixtureId::StructuralBeamCantileverEndMomentReference);
    configure_thermo_mechanical_excitation(&mut model, 65.0, 1.2e-5);
    model.sequence.steps = vec![ProcedureStep::coupled(
        "thermomechanical_explicit_nonlinear",
        runmat_analysis_core::CoupledPhysicsFamily::Thermomechanical,
        runmat_analysis_core::AnalysisProcedure::Nonlinear,
    )];
    let error = crate::run_nonlinear_with_options(
        &model,
        ComputeBackend::Cpu,
        baseline_nonlinear_options(),
    )
    .expect_err("explicit connectivity must not silently drop thermomechanical coupling");

    let message = error.to_string();
    assert!(
        matches!(
            error,
            crate::FeaRunError::FormulationAvailability(rejection)
                if rejection.reason == runmat_analysis_core::FormulationAdmissionRejectionReason::UnsupportedFamilyProcedure
        ),
        "{message}"
    );
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
        baseline_linear_static_options(),
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
            ..baseline_linear_static_options()
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
            ..baseline_linear_static_options()
        },
    )
    .expect("explicit beam topology should not require a solid solver mesh");

    assert!(result.diagnostics.iter().any(|diag| {
        diag.code == "FEA_STRUCTURAL_ROTATIONAL_DOF"
            && diag.message.contains("structural_beam_element_count=1")
    }));
}

#[test]
fn explicit_results_do_not_impersonate_continuum_stress_fields() {
    for fixture in [
        FixtureId::StructuralBeamCantileverEndMomentReference,
        FixtureId::StructuralShellPlateMomentReference,
    ] {
        let model = fixture_model(fixture);
        let result = crate::run_linear_static_with_options(
            &model,
            ComputeBackend::Cpu,
            baseline_linear_static_options(),
        )
        .expect("explicit structural solve");
        for field_id in [
            FEA_FIELD_STRUCTURAL_STRAIN,
            FEA_FIELD_STRUCTURAL_STRESS,
            FEA_FIELD_STRUCTURAL_VON_MISES,
            FEA_FIELD_STRUCTURAL_STRAIN_ENERGY_DENSITY,
        ] {
            assert!(
                result.field(field_id).is_none(),
                "explicit result must not publish continuum field {field_id}"
            );
        }
    }

    let model = fixture_model(FixtureId::StructuralBeamCantileverEndMomentReference);
    let result = crate::run_linear_static_with_options(
        &model,
        ComputeBackend::Cpu,
        baseline_linear_static_options(),
    )
    .expect("explicit beam solve");
    assert!(
        result
            .field(FEA_FIELD_STRUCTURAL_BEAM_BENDING_STRESS)
            .is_some(),
        "formulation-specific beam stress remains available"
    );
}

#[test]
fn solver_mesh_linear_static_reports_solid_assembly_basis() {
    let model = fixture_model(FixtureId::CantileverLinearStatic);
    let mesh = single_tetrahedron_solver_mesh();
    let projection_digest = crate::topology::AnalysisProjection::derive(&mesh)
        .expect("canonical solver mesh must project")
        .projection_digest;
    let result = crate::run_linear_static_with_options(
        &model,
        ComputeBackend::Cpu,
        LinearStaticSolveOptions {
            solver_mesh: Some(mesh),
            solver_mesh_artifact_path: Some("solver_mesh.cbor".to_string()),
            ..baseline_linear_static_options()
        },
    )
    .expect("analysis-mesh-backed linear static solve should succeed");

    assert!(result.diagnostics.iter().any(|diag| {
        diag.code == "FEA_STRUCTURAL_SOLID_ASSEMBLY"
            && diag.message.contains("basis=tet4_linear_elasticity")
            && diag.message.contains("stiffness_storage=sparse_csr")
            && diag.message.contains("mass_storage=sparse_csr")
            && diag.message.contains("solid_node_count=4")
            && diag.message.contains("solid_element_count=1")
            && diag.message.contains("stiffness_present=true")
            && diag.message.contains("mass_present=true")
    }));
    assert!(result.diagnostics.iter().any(|diag| {
        diag.code == "FEA_STRUCTURAL_FIELD_RECOVERY"
            && diag
                .message
                .contains("basis=solid_tetrahedron4_constant_strain")
            && diag.message.contains("solver_mesh_node_count=4")
            && diag.message.contains("solver_mesh_element_count=1")
    }));
    let evidence = result
        .spatial_projection_evidence
        .first()
        .expect("executed force projection must survive into the run result");
    assert_eq!(evidence.analysis_projection_digest, projection_digest);
    assert_eq!(
        evidence.quantity,
        runmat_analysis_core::PhysicalQuantityKind::Force
    );
    assert_eq!(evidence.signal_id, "tip_load.signal");
    assert_eq!(evidence.sample, crate::excitation::ProjectionSample::Static);
    assert!(evidence.accepted);
}

#[test]
fn linear_static_executes_authored_displacement_observation() {
    let mut model = fixture_model(FixtureId::CantileverLinearStatic);
    model.sequence.steps[0].observations = vec![runmat_analysis_core::ObservationRequest::Field(
        runmat_analysis_core::ObservationSpec {
            observation_id: runmat_analysis_core::ObservationId("whole_displacement".into()),
            quantity: runmat_analysis_core::ObservationQuantity::Displacement,
            value_shape: runmat_analysis_core::ObservationValueShape::Vector3,
            target: runmat_analysis_core::ObservationTarget::WholeModel,
            source_location: runmat_analysis_core::FieldLocation::Node,
            output_location: runmat_analysis_core::FieldLocation::Node,
            recovery: runmat_analysis_core::RecoveryPolicy::Native,
            component: runmat_analysis_core::ObservationComponent::All,
            frame: runmat_analysis_core::CoordinateFrame::Global,
            sign_convention: runmat_analysis_core::SignConvention::Native,
            complex_representation: runmat_analysis_core::ComplexRepresentation::Real,
            sampling: runmat_analysis_core::SamplingPolicy::Final,
            reduction: None,
            storage: runmat_analysis_core::ObservationStoragePolicy {
                precision: runmat_analysis_core::SignalPrecision::Real64,
                compression: runmat_analysis_core::CompressionPolicy::Lossless,
                retention: runmat_analysis_core::RetentionPolicy::Canonical,
                partial_access: runmat_analysis_core::PartialAccessPolicy::ManifestAndChunks,
            },
            purpose: runmat_analysis_core::ObservationPurpose::Publication,
            cost: runmat_analysis_core::ObservationCostPolicy::UpperBound(
                runmat_analysis_core::ObservationCostEstimate {
                    operation_count: 1_000,
                    peak_resident_bytes: 1_000_000,
                    stored_bytes: 1_000_000,
                    transfer_bytes: 1_000_000,
                    collective_count: 0,
                },
            ),
        },
    )];
    let mesh = single_tetrahedron_solver_mesh_for_model(&model);
    let result = crate::run_linear_static_with_options(
        &model,
        ComputeBackend::Cpu,
        LinearStaticSolveOptions {
            solver_mesh: Some(mesh),
            ..baseline_linear_static_options()
        },
    )
    .expect("authored observation should execute after physical field recovery");

    let observation = result
        .observations
        .first()
        .expect("the authored observation must be published");
    assert_eq!(
        observation.observation_id.0,
        "whole_displacement".to_string()
    );
    assert_eq!(
        observation.field.as_host_f64(),
        result
            .field(FEA_FIELD_STRUCTURAL_DISPLACEMENT)
            .and_then(runmat_analysis_core::AnalysisField::as_host_f64)
    );
    assert_eq!(observation.entities.len(), 4);
}

#[test]
fn linear_static_reuses_structural_primary_state_as_an_initial_guess() {
    let model = fixture_model(FixtureId::CantileverLinearStatic);
    let mesh = single_tetrahedron_solver_mesh_for_model(&model);
    let baseline = crate::run_linear_static_with_options(
        &model,
        ComputeBackend::Cpu,
        LinearStaticSolveOptions {
            solver_mesh: Some(mesh.clone()),
            solver_mesh_artifact_path: Some("solver_mesh.cbor".into()),
            ..baseline_linear_static_options()
        },
    )
    .expect("baseline structural solve should succeed");
    let initial_displacement = baseline
        .field(crate::FEA_FIELD_STRUCTURAL_DISPLACEMENT)
        .and_then(runmat_analysis_core::AnalysisField::as_host_f64)
        .expect("baseline must publish host displacement")
        .to_vec();

    let reused = crate::run_linear_static_with_options(
        &model,
        ComputeBackend::Cpu,
        LinearStaticSolveOptions {
            solver_mesh: Some(mesh),
            solver_mesh_artifact_path: Some("solver_mesh.cbor".into()),
            initial_displacement: Some(initial_displacement),
            ..baseline_linear_static_options()
        },
    )
    .expect("state-initialized structural solve should succeed");

    assert_eq!(
        baseline
            .field(crate::FEA_FIELD_STRUCTURAL_DISPLACEMENT)
            .unwrap(),
        reused
            .field(crate::FEA_FIELD_STRUCTURAL_DISPLACEMENT)
            .unwrap()
    );
    assert!(reused.diagnostics.iter().any(|diagnostic| {
        diagnostic.code == "FEA_CONVERGENCE" && diagnostic.message.contains("iterations=0 ")
    }));
}

fn single_tetrahedron_solver_mesh() -> SolverMeshArtifact {
    let mut mesh = crate::assembly::solver_solid::tests::artifact(ElementOrder::Tet4);
    add_face_region(&mut mesh, 0, "root");
    add_face_region(&mut mesh, 1, "tip");
    reseal_solver_mesh(&mut mesh);
    mesh
}

pub(crate) fn single_tetrahedron_solver_mesh_for_model(
    model: &runmat_analysis_core::AnalysisModel,
) -> SolverMeshArtifact {
    single_tetrahedron_solver_mesh_with_order_for_model(model, ElementOrder::Tet4)
}

fn single_tetrahedron_solver_mesh_with_order_for_model(
    model: &runmat_analysis_core::AnalysisModel,
    order: ElementOrder,
) -> SolverMeshArtifact {
    let mut mesh = crate::assembly::solver_solid::tests::artifact(order);
    add_face_region(&mut mesh, 0, "root");
    add_face_region(&mut mesh, 1, "tip");
    if let Some(assignment) = model.material_assignments.first() {
        let selection = assignment
            .selected_region_id()
            .expect("fixture uses selected material region");
        let region = runmat_meshing_core::fixtures::fixture_entity_for_selection(selection)
            .filter(|entity| entity.kind == PersistentEntityKind::Region)
            .expect("fixture material targets select a fixture region");
        mesh.topology.volume_elements[0].region_id = region.clone();
        mesh.topology.regions[0].region_id = region;
    }
    let mut root_regions = std::collections::BTreeSet::from(["root".to_string()]);
    root_regions.extend(
        model
            .constraints
            .iter()
            .filter_map(|condition| condition.target().map(|target| target.region_id.clone()))
            .filter(|region| {
                *region != crate::fixtures::fixture_target_selection("free_vertex_y_support")
                    && *region != crate::fixtures::fixture_target_selection("free_vertex_z_support")
            }),
    );
    let mut loaded_regions = std::collections::BTreeSet::from(["tip".to_string()]);
    loaded_regions.extend(
        model
            .excitations
            .iter()
            .filter_map(|action| match &action.excitation().target {
                runmat_analysis_core::ExcitationTarget::Region(target) => {
                    Some(target.region_id.clone())
                }
                runmat_analysis_core::ExcitationTarget::ReferencePoint(point) => {
                    Some(point.reference_point_id.clone())
                }
                _ => None,
            })
            .filter(|region| *region != crate::fixtures::fixture_target_selection("dynamic_force")),
    );
    loaded_regions.extend(model.material_assignments.iter().map(|assignment| {
        assignment
            .selected_region_id()
            .expect("fixture uses selected material region")
            .to_owned()
    }));
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
    if matches!(
        model.model_id.0.as_str(),
        "structural_transient_tet4_reference"
            | "structural_transient_tet4_support_motion_reference"
    ) {
        add_face_region(&mut mesh, 1, "free_vertex_y_support");
        add_face_region(&mut mesh, 2, "free_vertex_z_support");
        add_face_region(&mut mesh, 3, "dynamic_force");
    }
    reseal_solver_mesh(&mut mesh);
    mesh
}

#[test]
fn tet10_linear_static_admits_its_assembled_recovery_contract() {
    let model = fixture_model(FixtureId::CantileverLinearStatic);
    let mesh = single_tetrahedron_solver_mesh_with_order_for_model(&model, ElementOrder::Tet10);
    let result = crate::run_linear_static_with_options(
        &model,
        ComputeBackend::Cpu,
        LinearStaticSolveOptions {
            solver_mesh: Some(mesh.clone()),
            require_solver_mesh_for_solid: true,
            ..baseline_linear_static_options()
        },
    )
    .expect("a physical Tet10 solve must admit the recovery implementation it assembled");

    assert!(result.identity.realization().components.iter().any(|component| {
        component.spatial_operator
            == runmat_analysis_core::SpatialOperatorImplementation::Structural(
                runmat_analysis_core::StructuralOperatorImplementation::Tet10LinearElasticity,
            )
            && component.recovery
                == runmat_analysis_core::RecoveryImplementation::Structural(
                    runmat_analysis_core::StructuralRecoveryImplementation::Tet10IntegrationPointProjection,
                )
    }));
    assert!(result
        .field(FEA_FIELD_STRUCTURAL_STRAIN)
        .is_some_and(|field| field.shape == vec![1, 6]));
    assert!(result
        .field(FEA_FIELD_STRUCTURAL_STRESS)
        .is_some_and(|field| field.shape == vec![1, 6]));
    let integration_point_stress = result
        .field(crate::contracts::FEA_FIELD_STRUCTURAL_INTEGRATION_POINT_STRESS)
        .expect("Tet10 recovery must retain its four physical integration-point stresses");
    assert_eq!(integration_point_stress.shape, vec![4, 6]);
    let estimate = crate::adaptation::estimate_structural_recovery_error(
        &mesh,
        "elements",
        integration_point_stress,
        crate::adaptation::StructuralRecoveryEstimatorOptions::default(),
    )
    .expect("the executed Tet10 field must be accepted by the recovery estimator");
    assert_eq!(estimate.solver_artifact_digest, mesh.canonical_digest);
}

// witness: FEA-DYNAMICS-PRESTRESS-001/complete_base_operator @
//   structural_stability/tet4/numerical_kernel; coverage: partial;
//   boundary: A physical preload publishes a signed-load linear geometric stiffness on a Tet4
//   fixture.
#[test]
fn physical_preload_publishes_signed_load_linear_geometric_stiffness() {
    let mut model = fixture_model(FixtureId::StructuralTransientTet4Reference);
    model.sequence.steps = vec![ProcedureStep::structural_static("preload")];
    model.constraints.truncate(1);
    let mesh = single_tetrahedron_solver_mesh_for_model(&model);
    let plan = crate::constraints::CanonicalStructuralConstraintPlan::compile_for_model(
        &model,
        Some(&mesh),
    )
    .unwrap();
    let topology_digest = plan.topology.discretization_digest();
    let options = LinearStaticSolveOptions {
        solver_mesh: Some(mesh.clone()),
        require_solver_mesh_for_solid: true,
        ..LinearStaticSolveOptions::default()
    };
    let base = crate::run_linear_static_with_prepared_constraint_plan_and_linearization(
        &model,
        ComputeBackend::Cpu,
        options.clone(),
        &plan,
        topology_digest,
        None,
    )
    .unwrap();
    assert_eq!(base.linearization.topology_digest(), topology_digest);
    assert_eq!(base.linearization.initial_stress().len(), 1);
    assert_eq!(
        base.linearization.evidence().stress_quadrature_point_count,
        15
    );
    assert!(base
        .linearization
        .geometric_stiffness()
        .values()
        .iter()
        .any(|value| value.abs() > 0.0));

    replace_fixture_excitations(
        &mut model,
        vec![FixtureExcitation {
            id: "double_preload".into(),
            region_id: "dynamic_force".into(),
            kind: FixtureExcitationKind::Force([2.0, 0.0, 0.0]),
        }],
    );
    let doubled = crate::run_linear_static_with_prepared_constraint_plan_and_linearization(
        &model,
        ComputeBackend::Cpu,
        options,
        &plan,
        topology_digest,
        None,
    )
    .unwrap();
    for row in 0..base.linearization.geometric_stiffness().rows() {
        for column in 0..base.linearization.geometric_stiffness().columns() {
            let expected = 2.0 * base.linearization.geometric_stiffness().get(row, column);
            let actual = doubled.linearization.geometric_stiffness().get(row, column);
            assert!((actual - expected).abs() <= 1.0e-10 * expected.abs().max(1.0));
        }
    }

    let constraint_result =
        crate::constraints::CanonicalStructuralConstraintResult::recover_for_model(
            &plan,
            &model,
            Some(&mesh),
            &base.run.fields,
        )
        .unwrap();
    let portable = crate::execution::StructuralLinearWorkloadResult::with_linearization(
        base.run,
        constraint_result,
        base.linearization,
    )
    .unwrap();
    let decoded = crate::execution::StructuralLinearWorkloadResult::from_canonical_bytes(
        &portable.canonical_bytes().unwrap(),
    )
    .unwrap();
    assert_eq!(decoded, portable);
}

// witness: FEA-DYNAMICS-PRESTRESS-001/references @ structural_stability/tet4/numerical_kernel;
//   coverage: partial;
//   boundary: A single-tetrahedron compressive preload buckling problem has an exact residual and
//   load factors that scale inversely with load; column and frame buckling, mesh convergence and
//   sign cases are not covered.
// witness: FEA-DYNAMICS-GOVERNANCE-001/modal_preload_buckling @
//   linear_structural_dynamics/tet4/numerical_kernel; coverage: partial;
//   boundary: Single-tetrahedron preload buckling with exact residual and inverse load scaling.
#[test]
fn physical_compressive_preload_buckling_has_exact_residual_and_inverse_load_scaling() {
    let mut model = fixture_model(FixtureId::StructuralTransientTet4Reference);
    model.sequence.steps = vec![ProcedureStep::structural_static("preload")];
    model.constraints.truncate(1);
    let mesh = single_tetrahedron_solver_mesh_for_model(&model);
    let plan = crate::constraints::CanonicalStructuralConstraintPlan::compile_for_model(
        &model,
        Some(&mesh),
    )
    .unwrap();
    let topology_digest = plan.topology.discretization_digest();
    let options = || LinearStaticSolveOptions {
        solver_mesh: Some(mesh.clone()),
        require_solver_mesh_for_solid: true,
        ..LinearStaticSolveOptions::default()
    };
    let solve_at_load = |model: &runmat_analysis_core::AnalysisModel| {
        let preload = crate::run_linear_static_with_prepared_constraint_plan_and_linearization(
            model,
            ComputeBackend::Cpu,
            options(),
            &plan,
            topology_digest,
            None,
        )
        .unwrap();
        let excitations = static_excitations(model);
        let summary = crate::assembly::assemble_structural_system_with_constraint_plan(
            model,
            &excitations,
            Some(mesh.clone()),
            crate::assembly::StructuralFormulationRequirement::SyntheticOperator,
            &plan,
        )
        .unwrap();
        crate::solve::buckling::solve_eigenvalue_buckling(
            &summary,
            &preload.linearization,
            1,
            1.0e-10,
            8,
        )
        .unwrap()
    };

    replace_fixture_excitations(
        &mut model,
        vec![FixtureExcitation {
            id: "compressive_preload".into(),
            region_id: "tip".into(),
            kind: FixtureExcitationKind::Force([0.0, 0.0, -1.0]),
        }],
    );
    let unit = solve_at_load(&model);
    assert!(unit.converged);
    assert_eq!(unit.modes.len(), 1);
    assert!(unit.modes[0].load_factor > 0.0);
    assert!(unit.modes[0].residual_norm <= 1.0e-12);

    replace_fixture_excitations(
        &mut model,
        vec![FixtureExcitation {
            id: "double_compressive_preload".into(),
            region_id: "tip".into(),
            kind: FixtureExcitationKind::Force([0.0, 0.0, -2.0]),
        }],
    );
    let doubled = solve_at_load(&model);
    assert!(doubled.converged);
    assert!(
        (doubled.modes[0].load_factor - 0.5 * unit.modes[0].load_factor).abs()
            <= 1.0e-10 * unit.modes[0].load_factor
    );
}

#[test]
fn portable_buckling_pipeline_consumes_the_exact_published_preload_state() {
    use runmat_analysis_core::{
        ReferenceStatePolicy, SequenceDependency, SequenceDependencyReason, StateTransition,
        StateTransitionMethod, TopologyTransitionPolicy, TransitionId,
    };
    use runmat_execution::Digest;

    use crate::execution::{
        CanonicalSemanticState, CanonicalSemanticStateInput, CanonicalSemanticStatePayload,
        CanonicalStateTransition, CanonicalStateTransitionInput, SequenceIdentity,
        SolverResultIdentity, StepIdentity, StructuralBucklingSolverPolicy,
        StructuralBucklingTemplate,
    };

    let mut model = fixture_model(FixtureId::StructuralTransientTet4Reference);
    let preload_step = ProcedureStep::structural_preload("preload");
    let mut buckling_step =
        ProcedureStep::structural_buckling("buckling", "preload", "geometric_stiffness");
    buckling_step.independent_domain =
        runmat_analysis_core::IndependentDomain::Eigenvalue { requested_count: 1 };
    model.sequence.steps = vec![preload_step, buckling_step];
    model.sequence.dependencies = vec![SequenceDependency {
        predecessor: runmat_analysis_core::ProcedureStepId("preload".into()),
        successor: runmat_analysis_core::ProcedureStepId("buckling".into()),
        reason: SequenceDependencyReason::StateTransfer,
    }];
    model.sequence.transitions = vec![StateTransition {
        transition_id: TransitionId("preload-to-buckling".into()),
        producer_step: runmat_analysis_core::ProcedureStepId("preload".into()),
        consumer_step: runmat_analysis_core::ProcedureStepId("buckling".into()),
        state_id: runmat_analysis_core::StateId("geometric_stiffness".into()),
        method: StateTransitionMethod::Reuse,
        topology: TopologyTransitionPolicy::RequireIdentical,
        reference: ReferenceStatePolicy::Preserve,
    }];
    model.constraints.truncate(1);
    replace_fixture_excitations(
        &mut model,
        vec![FixtureExcitation {
            id: "compressive_preload".into(),
            region_id: "tip".into(),
            kind: FixtureExcitationKind::Force([0.0, 0.0, -1.0]),
        }],
    );
    let mesh = single_tetrahedron_solver_mesh_for_model(&model);
    let topology_digest = Digest::from_bytes(*mesh.canonical_digest.bytes());
    let constraint_plan = crate::constraints::CanonicalStructuralConstraintPlan::compile_for_model(
        &model,
        Some(&mesh),
    )
    .unwrap();
    let preload = crate::run_linear_static_with_prepared_constraint_plan_and_linearization(
        &model,
        ComputeBackend::Cpu,
        LinearStaticSolveOptions {
            solver_mesh: Some(mesh.clone()),
            require_solver_mesh_for_solid: true,
            ..LinearStaticSolveOptions::default()
        },
        &constraint_plan,
        topology_digest,
        None,
    )
    .unwrap();
    let sequence = SequenceIdentity::from_solver_contract(&model.sequence).unwrap();
    let producer = StepIdentity::from_solver_contract(sequence, &model.sequence.steps[0]).unwrap();
    let consumer = StepIdentity::from_solver_contract(sequence, &model.sequence.steps[1]).unwrap();
    let source_result =
        SolverResultIdentity::from_step_result(producer, preload.linearization.identity().digest())
            .unwrap();
    let state = CanonicalSemanticState::new(CanonicalSemanticStateInput {
        producer,
        declaration: model.sequence.steps[0].state_outputs[1].clone(),
        topology_digest,
        source_result,
        payload: CanonicalSemanticStatePayload::StructuralLinearization {
            linearization: Box::new(preload.linearization),
        },
    })
    .unwrap();
    let transition = CanonicalStateTransition::new(CanonicalStateTransitionInput {
        sequence,
        producer,
        consumer,
        authored: &model.sequence.transitions[0],
        source: &state,
        target_topology_digest: topology_digest,
    })
    .unwrap();
    let workload = StructuralBucklingTemplate {
        schema_version: 1,
        model,
        step_id: runmat_analysis_core::ProcedureStepId("buckling".into()),
        backend: ComputeBackend::Cpu,
        observation_execution: crate::results::observation::ObservationExecutionPolicy::scalable(),
        solver: StructuralBucklingSolverPolicy::new(1, 1.0e-10, 8).unwrap(),
    }
    .bind(transition)
    .unwrap();
    let result = crate::execute_structural_buckling_workload(&workload, &mesh, &state).unwrap();
    assert_eq!(result.modes().len(), 1);
    assert!(result.modes()[0].load_factor > 0.0);
    assert!(result.modes()[0].residual_norm <= 1.0e-12);
    let decoded = crate::execution::StructuralBucklingSolverResult::from_canonical_bytes(
        &result.canonical_bytes().unwrap(),
    )
    .unwrap();
    decoded.validate_for(&workload).unwrap();

    let mut substituted = state.clone();
    substituted.topology_digest = Digest::sha256(b"different-topology");
    assert!(crate::execute_structural_buckling_workload(&workload, &mesh, &substituted).is_err());
}

pub(crate) fn prepared_prestressed_modal_case(
    axial_force: f64,
) -> (
    crate::execution::StructuralModalWorkload,
    SolverMeshArtifact,
    crate::execution::CanonicalSemanticState,
) {
    use runmat_analysis_core::{
        AnalysisSequence, ReferenceStatePolicy, StateTransition, StateTransitionMethod,
        TopologyTransitionPolicy, TransitionId,
    };
    use runmat_execution::Digest;

    use crate::execution::{
        CanonicalSemanticState, CanonicalSemanticStateInput, CanonicalSemanticStatePayload,
        CanonicalStateTransition, SequenceIdentity, SolverResultIdentity, StepIdentity,
        StructuralPrestressedModalTemplate,
    };

    let mut model = fixture_model(FixtureId::StructuralTransientTet4Reference);
    let preload = ProcedureStep::structural_preload("preload");
    let mut modal = ProcedureStep::structural_prestressed_modal(
        "prestressed_modes",
        "preload",
        "geometric_stiffness",
    );
    modal.independent_domain =
        runmat_analysis_core::IndependentDomain::Eigenvalue { requested_count: 1 };
    let mut sequence =
        AnalysisSequence::from_steps("prestressed_modal", vec![preload, modal.clone()]);
    sequence.dependencies[0].reason = runmat_analysis_core::SequenceDependencyReason::StateTransfer;
    sequence.transitions.push(StateTransition {
        transition_id: TransitionId("preload-to-prestressed-modal".into()),
        producer_step: runmat_analysis_core::ProcedureStepId("preload".into()),
        consumer_step: modal.step_id.clone(),
        state_id: runmat_analysis_core::StateId("geometric_stiffness".into()),
        method: StateTransitionMethod::Reuse,
        topology: TopologyTransitionPolicy::RequireIdentical,
        reference: ReferenceStatePolicy::Preserve,
    });
    model.sequence = sequence;
    model.constraints.truncate(1);
    replace_fixture_excitations(
        &mut model,
        vec![FixtureExcitation {
            id: "axial_preload".into(),
            region_id: "tip".into(),
            kind: FixtureExcitationKind::Force([0.0, 0.0, axial_force]),
        }],
    );
    let mesh = single_tetrahedron_solver_mesh_for_model(&model);
    let topology_digest = Digest::from_bytes(*mesh.canonical_digest.bytes());
    let constraints = crate::constraints::CanonicalStructuralConstraintPlan::compile_for_model(
        &model,
        Some(&mesh),
    )
    .unwrap();
    let preload = crate::run_linear_static_with_prepared_constraint_plan_and_linearization(
        &model,
        ComputeBackend::Cpu,
        LinearStaticSolveOptions {
            solver_mesh: Some(mesh.clone()),
            require_solver_mesh_for_solid: true,
            ..LinearStaticSolveOptions::default()
        },
        &constraints,
        topology_digest,
        None,
    )
    .unwrap();
    let sequence = SequenceIdentity::from_solver_contract(&model.sequence).unwrap();
    let producer = StepIdentity::from_solver_contract(sequence, &model.sequence.steps[0]).unwrap();
    let source_result =
        SolverResultIdentity::from_step_result(producer, preload.linearization.identity().digest())
            .unwrap();
    let state = CanonicalSemanticState::new(CanonicalSemanticStateInput {
        producer,
        declaration: model.sequence.steps[0].state_outputs[1].clone(),
        topology_digest,
        source_result,
        payload: CanonicalSemanticStatePayload::StructuralLinearization {
            linearization: Box::new(preload.linearization),
        },
    })
    .unwrap();
    let transition = CanonicalStateTransition::bind_upstream(
        &model,
        &modal.step_id,
        &state,
        runmat_analysis_core::StateUse::Prestress,
        topology_digest,
    )
    .unwrap();
    let workload = StructuralPrestressedModalTemplate {
        schema_version: 1,
        model,
        step_id: modal.step_id,
        backend: ComputeBackend::Cpu,
        observation_execution: crate::results::observation::ObservationExecutionPolicy::scalable(),
        truncation: crate::ModalTruncationPolicy::requested(1),
        convergence: crate::ModalConvergencePolicy::exploratory(),
        residual_vector_directions: Vec::new(),
    }
    .bind(transition)
    .unwrap();
    (workload, mesh, state)
}

#[test]
fn prestressed_modal_pipeline_proves_tension_stiffening_and_compression_softening() {
    let (unit_compression, unit_mesh, unit_state) = prepared_prestressed_modal_case(-1.0);
    let unit_linearization = unit_state.payload.structural_linearization().unwrap();
    let unit_summary = crate::assembly::assemble_structural_system(
        unit_compression.model(),
        &static_excitations(unit_compression.model()),
        Some(unit_mesh),
        crate::assembly::StructuralFormulationRequirement::SyntheticOperator,
    )
    .unwrap();
    let critical = crate::solve::buckling::solve_eigenvalue_buckling(
        &unit_summary,
        unit_linearization,
        1,
        1.0e-10,
        16,
    )
    .unwrap()
    .modes[0]
        .load_factor;
    let qualified_force = 0.1 * critical;

    let solve = |force| {
        let (workload, mesh, state) = prepared_prestressed_modal_case(force);
        let result =
            crate::execute_structural_prestressed_modal_workload(&workload, &mesh, &state).unwrap();
        (workload, mesh, state, result)
    };
    let (_, _, _, unstressed) = solve(0.0);
    let (_, _, _, tension) = solve(qualified_force);
    let (compression_workload, compression_mesh, compression_state, compression) =
        solve(-qualified_force);
    let unstressed_frequency = unstressed.basis().modes()[0].frequency_hz();
    let tension_frequency = tension.basis().modes()[0].frequency_hz();
    let compression_frequency = compression.basis().modes()[0].frequency_hz();
    assert!(tension_frequency > unstressed_frequency);
    assert!(compression_frequency < unstressed_frequency);

    let round_trip = crate::execution::StructuralModalSolverResult::from_canonical_bytes(
        &compression.canonical_bytes().unwrap(),
    )
    .unwrap();
    assert_eq!(round_trip, compression);
    let source = crate::execution::AnalysisWorkloadIdentity::for_test(
        "prestressed-modal-physical-reference",
    )
    .unwrap();
    let receipt =
        crate::execution::StructuralModalNumericalReceipt::new(source, &compression).unwrap();
    let recovered = crate::recover_completed_structural_prestressed_modal_run(
        &compression_workload,
        &compression_mesh,
        &compression_state,
        &compression,
        &receipt,
    )
    .unwrap();
    assert_eq!(recovered.basis, *compression.basis());

    let mut substituted = compression_state;
    substituted.topology_digest = runmat_execution::Digest::sha256(b"different-topology");
    assert!(crate::execute_structural_prestressed_modal_workload(
        &compression_workload,
        &compression_mesh,
        &substituted,
    )
    .is_err());
}

#[test]
fn structural_frequency_entries_execute_independently_and_assemble_in_plan_order() {
    use runmat_analysis_core::IndependentDomain;

    use crate::{
        execution::{
            FieldRecoveryRequest, FieldRecoverySource, StructuralFrequencyEntryRequest,
            StructuralFrequencyResultAssemblyRequest, StructuralFrequencySolverPlan,
            StructuralFrequencyWorkload, StructuralFrequencyWorkloadInput,
        },
        frequency::FrequencyResponsePlan,
        pipeline::structural_frequency::{
            execute_structural_frequency_entry, prepare_structural_frequency,
            recover_structural_frequency_entry_fields,
        },
        results::field::{RecoveredFieldSample, StructuralRecoveredFieldTopology},
        solve::dynamics::DirectHarmonicPolicy,
    };

    let mut model = fixture_model(FixtureId::CantileverLinearStatic);
    let mut step = ProcedureStep::structural_harmonic("frequency_response");
    step.independent_domain = IndependentDomain::FrequencyHertz(vec![20.0, 40.0]);
    let mut connector_observation = history_observation(
        "dashpot_force",
        runmat_analysis_core::ObservationQuantity::ReactionForce,
        runmat_analysis_core::SamplingPolicy::FrequencyHertz(vec![20.0, 40.0]),
        runmat_analysis_core::FieldLocation::Connector,
    );
    let runmat_analysis_core::ObservationRequest::History(spec) = &mut connector_observation else {
        unreachable!()
    };
    spec.target = runmat_analysis_core::ObservationTarget::Connector {
        connector_id: "root_to_tip_dashpot".into(),
    };
    spec.complex_representation = runmat_analysis_core::ComplexRepresentation::Complex;
    spec.storage.precision = runmat_analysis_core::SignalPrecision::Complex128;
    step.observations = vec![connector_observation];
    model.sequence.steps = vec![step];
    const DASHPOT_COEFFICIENT: f64 = 75.0;
    model.connectors = vec![runmat_analysis_core::AnalysisConnector {
        connector_id: "root_to_tip_dashpot".into(),
        kind: runmat_analysis_core::ConnectorKind::AxialDashpot {
            primary: runmat_analysis_core::RegionTarget::point("node:2"),
            secondary: runmat_analysis_core::RegionTarget::point("node:4"),
            damping: runmat_analysis_core::Damping::new(DASHPOT_COEFFICIENT),
        },
    }];
    let mesh = single_tetrahedron_solver_mesh_for_model(&model);
    let plan = FrequencyResponsePlan::new(20.0, &[40.0], &[]).unwrap();
    let workload = StructuralFrequencyWorkload::new(StructuralFrequencyWorkloadInput {
        model,
        step_id: runmat_analysis_core::ProcedureStepId("frequency_response".into()),
        backend: ComputeBackend::Cpu,
        observation_execution: crate::results::observation::ObservationExecutionPolicy::scalable(),
        plan,
        solver: StructuralFrequencySolverPlan::Direct {
            policy: DirectHarmonicPolicy::default(),
        },
        prestress_transition: None,
    })
    .unwrap();
    assert_eq!(
        StructuralFrequencyWorkload::from_canonical_bytes(&workload.canonical_bytes().unwrap())
            .unwrap(),
        workload
    );
    let prepared = prepare_structural_frequency(&workload, &mesh).unwrap();
    let mut results = workload
        .plan()
        .entries()
        .unwrap()
        .into_iter()
        .map(|entry| {
            let request = StructuralFrequencyEntryRequest::new(
                &workload,
                prepared.study,
                prepared.step,
                prepared.execution_plan,
                prepared.scope.clone(),
                entry,
            )
            .unwrap();
            let decoded = StructuralFrequencyEntryRequest::from_canonical_bytes(
                &request.canonical_bytes().unwrap(),
            )
            .unwrap();
            assert_eq!(decoded, request);
            execute_structural_frequency_entry(&workload, &mesh, &request, None).unwrap()
        })
        .collect::<Vec<_>>();
    assert_eq!(results.len(), 2);
    assert!(results.iter().all(|result| result
        .sample()
        .balance()
        .equilibrium_residual()
        .real()
        .iter()
        .chain(result.sample().balance().equilibrium_residual().imaginary())
        .all(|value| value.is_finite())));
    results.reverse();
    let request = StructuralFrequencyResultAssemblyRequest::new(&results).unwrap();
    let assembled = request.assemble(&results).unwrap();
    assert_eq!(
        assembled
            .result
            .samples()
            .iter()
            .map(|sample| sample.entry().frequency_hz())
            .collect::<Vec<_>>(),
        vec![20.0, 40.0]
    );
    assert_eq!(assembled.receipt.samples().len(), 2);
    let topology = StructuralRecoveredFieldTopology::derive(workload.model(), Some(&mesh)).unwrap();
    let mut frequency_catalogs = Vec::new();
    let fields = results
        .iter()
        .map(|entry| {
            let recovery_request = FieldRecoveryRequest::new(
                prepared.step,
                FieldRecoverySource::StructuralFrequencyEntry(entry.identity()),
                crate::execution::FieldTopologyIdentity::Structural {
                    topology: topology.identity,
                },
                crate::execution::FieldRecoveryPlan::resolve_structural(
                    workload.procedure_step().unwrap(),
                    topology,
                    0,
                )
                .unwrap(),
            )
            .unwrap();
            let artifact = recover_structural_frequency_entry_fields(
                &workload,
                &mesh,
                &recovery_request,
                entry,
            )
            .unwrap();
            frequency_catalogs.push(
                crate::results::observation::ObservationFieldCatalog::structural(
                    recovery_request,
                    artifact.manifest().unwrap(),
                )
                .unwrap(),
            );
            artifact
        })
        .collect::<Vec<_>>();
    // Results arrived in reverse order. A complete history must authenticate the
    // original schedule, rather than inheriting the last recovery's authority.
    frequency_catalogs.reverse();
    use crate::results::observation::{
        resolve_structural_observation_plan, ObservationFieldCatalog, ObservationLimits,
    };
    let constraints = crate::constraints::CanonicalStructuralConstraintPlan::compile_for_model(
        workload.model(),
        Some(&mesh),
    )
    .unwrap();
    let definitions = fields[0]
        .fields()
        .iter()
        .map(|field| field.definition().clone())
        .collect::<Vec<_>>();
    let observation_plan = resolve_structural_observation_plan(
        workload.procedure_step().unwrap(),
        workload.model(),
        Some(&mesh),
        &constraints,
        &definitions,
        ObservationLimits::default(),
    )
    .unwrap();
    let complete =
        ObservationFieldCatalog::entry_history(frequency_catalogs.clone()).unwrap();
    complete
        .validate_frequency_history_plan(&observation_plan, &workload, Some(&mesh))
        .unwrap();
    assert_eq!(
        ObservationFieldCatalog::from_canonical_bytes(&complete.canonical_bytes().unwrap())
            .unwrap(),
        complete
    );
    let omitted =
        ObservationFieldCatalog::entry_history(vec![frequency_catalogs[0].clone()])
            .unwrap();
    assert!(omitted
        .validate_frequency_history_plan(&observation_plan, &workload, Some(&mesh))
        .is_err());
    assert!(ObservationFieldCatalog::entry_history(vec![
        frequency_catalogs[0].clone(),
        frequency_catalogs[0].clone()
    ])
    .is_err());
    let mut reordered = frequency_catalogs;
    reordered.reverse();
    assert!(ObservationFieldCatalog::entry_history(reordered).is_err());
    assert!(ObservationFieldCatalog::entry_history(vec![complete]).is_err());
    assert!(fields
        .iter()
        .any(|artifact| artifact.fields().iter().any(|field| {
            field.sample() == RecoveredFieldSample::FrequencyHertz(20.0)
                && field.definition().quantity()
                    == runmat_analysis_core::ObservationQuantity::ReactionForce
        })));
    assert!(fields
        .iter()
        .any(|artifact| artifact.fields().iter().any(|field| {
            field.sample() == RecoveredFieldSample::FrequencyHertz(40.0)
                && field.definition().quantity()
                    == runmat_analysis_core::ObservationQuantity::Stress
        })));
    assert_eq!(topology.entity_counts.connectors, 1);
    let response = results
        .iter()
        .find(|result| result.sample().entry().frequency_hz() == 20.0)
        .unwrap()
        .sample();
    let artifact = fields
        .iter()
        .find(|artifact| {
            artifact
                .fields()
                .iter()
                .any(|field| field.sample() == RecoveredFieldSample::FrequencyHertz(20.0))
        })
        .unwrap();
    let connector_force = artifact
        .fields()
        .iter()
        .find(|field| field.definition().field_id() == crate::FEA_FIELD_STRUCTURAL_CONNECTOR_FORCE)
        .unwrap();
    assert_eq!(connector_force.storage().shape, vec![1, 3, 2]);
    let primary = mesh
        .topology
        .nodes
        .iter()
        .position(|node| node.node_id == 2)
        .unwrap();
    let secondary = mesh
        .topology
        .nodes
        .iter()
        .position(|node| node.node_id == 4)
        .unwrap();
    let delta = std::array::from_fn::<_, 3, _>(|component| {
        mesh.topology.nodes[secondary]
            .coordinates_m
            .canonical_value()[component]
            - mesh.topology.nodes[primary].coordinates_m.canonical_value()[component]
    });
    let length = delta.iter().map(|value| value * value).sum::<f64>().sqrt();
    let axis = delta.map(|value| value / length);
    let relative_real = (0..3)
        .map(|component| {
            axis[component]
                * (response.displacement().real()[secondary * 3 + component]
                    - response.displacement().real()[primary * 3 + component])
        })
        .sum::<f64>();
    let relative_imaginary = (0..3)
        .map(|component| {
            axis[component]
                * (response.displacement().imaginary()[secondary * 3 + component]
                    - response.displacement().imaginary()[primary * 3 + component])
        })
        .sum::<f64>();
    let omega = std::f64::consts::TAU * 20.0;
    let actual = connector_force.storage().as_host_f64().unwrap();
    for component in 0..3 {
        let expected_real = axis[component] * -omega * DASHPOT_COEFFICIENT * relative_imaginary;
        let expected_imaginary = axis[component] * omega * DASHPOT_COEFFICIENT * relative_real;
        let scale = expected_real.abs().max(expected_imaginary.abs()).max(1.0);
        assert!((actual[component * 2] - expected_real).abs() <= 1.0e-10 * scale);
        assert!((actual[component * 2 + 1] - expected_imaginary).abs() <= 1.0e-10 * scale);
    }
}

pub(crate) fn multi_region_tetrahedron_solver_mesh() -> SolverMeshArtifact {
    runmat_meshing_core::fixtures::three_region_tetrahedron_solver_mesh()
}

fn add_face_region(mesh: &mut SolverMeshArtifact, face_index: usize, region_id: &str) {
    let provenance = &mut mesh.topology.boundary_faces[face_index].provenance;
    let entity = match runmat_meshing_core::fixtures::fixture_entity_for_selection(region_id) {
        Some(entity) if entity.kind == PersistentEntityKind::Face => entity,
        Some(_) => return,
        None => {
            runmat_meshing_core::fixtures::fixture_entity(PersistentEntityKind::Face, region_id)
        }
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
fn static_gpu_request_falls_back_to_cpu_reference_with_parity_unless_required() {
    use crate::contracts::accelerator_admission::{
        with_accelerator_admission_policy, AcceleratorAdmissionPolicy,
        ACCELERATOR_ADMISSION_FALLBACK_CODE,
    };
    let model = fixture_model(FixtureId::CantileverLinearStatic);
    let options = |algebra_backend_kind| LinearStaticSolveOptions {
        solver_mesh: Some(single_tetrahedron_solver_mesh_for_model(&model)),
        algebra_backend_kind,
        ..baseline_linear_static_options()
    };
    let cpu = crate::run_linear_static_with_options(
        &model,
        ComputeBackend::Cpu,
        options(crate::solve::backend::kind::LinearAlgebraBackendKind::CpuReference),
    )
    .expect("CPU reference static solve");
    for algebra_backend_kind in [
        crate::solve::backend::kind::LinearAlgebraBackendKind::CpuReference,
        crate::solve::backend::kind::LinearAlgebraBackendKind::Accelerator,
    ] {
        let fallback = crate::run_linear_static_with_options(
            &model,
            ComputeBackend::Gpu,
            options(algebra_backend_kind),
        )
        .expect("GPU placement falls back to the CPU reference static solve");
        assert_eq!(fallback.requested_backend, ComputeBackend::Gpu);
        assert_eq!(fallback.execution_backend, crate::ExecutionBackend::Cpu);
        assert_eq!(
            fallback.solver_backend,
            crate::solve::backend::kind::LinearAlgebraBackendKind::CpuReference
        );
        let warnings = fallback
            .diagnostics
            .iter()
            .filter(|diagnostic| diagnostic.code == ACCELERATOR_ADMISSION_FALLBACK_CODE)
            .collect::<Vec<_>>();
        assert_eq!(warnings.len(), 1, "{:?}", fallback.diagnostics);
        assert!(warnings[0].message.contains("structural static"));
        assert_eq!(
            fallback.fields, cpu.fields,
            "fallback must be the CPU reference solve"
        );

        let error = with_accelerator_admission_policy(AcceleratorAdmissionPolicy::Required, || {
            crate::run_linear_static_with_options(
                &model,
                ComputeBackend::Gpu,
                options(algebra_backend_kind),
            )
        })
        .expect_err("required accelerator placement fails closed");
        assert!(matches!(
            &error,
            crate::contracts::FeaRunError::AcceleratorRequired(required)
                if required.family == "structural static"
        ));
        assert!(error.to_string().starts_with("FEA_ACCELERATOR_REQUIRED"));
    }
}

#[test]
fn fixture_missing_materials_is_rejected() {
    let model = fixture_model(FixtureId::MissingMaterials);
    let err = crate::run_linear_static_with_options(
        &model,
        ComputeBackend::Cpu,
        baseline_linear_static_options(),
    )
    .expect_err("fixture should fail validation");
    assert!(err
        .to_string()
        .contains("ANALYSIS_VALIDATION_MISSING_MATERIALS"));
}

#[test]
fn thermal_pipeline_rejects_moment_loads_before_assembly() {
    let model = fixture_model(FixtureId::StructuralBeamCantileverEndMomentReference);
    let err =
        crate::run_thermal_with_options(&model, ComputeBackend::Cpu, baseline_thermal_options())
            .expect_err("thermal solve should reject structural moment loads");
    let message = err.to_string();

    assert!(message.contains("moment loads are structural loads"));
    assert!(message.contains("thermal"));
    assert!(message.contains("load_id=tip_moment_z"));
}

#[test]
fn thermal_pipeline_requires_a_canonical_solver_mesh() {
    let mut model = fixture_model(FixtureId::CantileverLinearStatic);
    model.thermal = Some(runmat_analysis_core::ThermalDomain {
        reference_temperature: runmat_analysis_core::Temperature::new(293.15),
    });
    model.sequence = runmat_analysis_core::AnalysisSequence::from_steps(
        "thermal_sequence",
        vec![runmat_analysis_core::ProcedureStep::thermal_transient(
            "thermal_step",
        )],
    );

    let error =
        crate::run_thermal_with_options(&model, ComputeBackend::Cpu, baseline_thermal_options())
            .expect_err("thermal execution must not synthesize topology");

    assert!(error
        .to_string()
        .contains("requires a canonical solver mesh"));
}

#[test]
fn modal_solver_emits_modes_for_modal_step_fixture() {
    let mut model = fixture_model(FixtureId::CantileverLinearStatic);
    model.sequence.steps = vec![ProcedureStep::structural_modal("modal_1".to_string())];
    let result =
        crate::run_modal_with_options(&model, ComputeBackend::Cpu, baseline_modal_options())
            .expect("modal solve should succeed");

    assert!(!result.frequencies_hz().is_empty());
    assert_eq!(result.frequencies_hz().len(), result.mode_shapes.len());
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
        result.frequencies_hz().len()
    );
    assert_eq!(
        field(&result.run, FEA_FIELD_MODAL_EIGENVALUE).element_count(),
        result.frequencies_hz().len()
    );
    assert_eq!(
        field(&result.run, FEA_FIELD_MODAL_MODAL_MASS).element_count(),
        result.frequencies_hz().len()
    );
    assert_eq!(
        field(&result.run, FEA_FIELD_MODAL_MODAL_STIFFNESS).element_count(),
        result.frequencies_hz().len()
    );
    assert_eq!(
        field(&result.run, FEA_FIELD_MODAL_PARTICIPATION_FACTOR).element_count(),
        result.frequencies_hz().len() * 3
    );
    for field_id in [
        FEA_FIELD_MODAL_EFFECTIVE_MASS,
        FEA_FIELD_MODAL_CUMULATIVE_EFFECTIVE_MASS,
        FEA_FIELD_MODAL_CUMULATIVE_MASS_RATIO,
    ] {
        assert_eq!(
            field(&result.run, field_id).shape,
            vec![result.frequencies_hz().len(), 3]
        );
    }
    assert_eq!(
        field(&result.run, FEA_FIELD_MODAL_TOTAL_FREE_MASS).shape,
        vec![3]
    );
    assert_eq!(
        field(&result.run, FEA_FIELD_MODAL_RESIDUAL_NORM).element_count(),
        result.frequencies_hz().len()
    );
    assert_eq!(
        field(&result.run, FEA_FIELD_MODAL_RELATIVE_FREQUENCY_SEPARATION).element_count(),
        result.frequencies_hz().len()
    );
    let orthogonality = field(&result.run, FEA_FIELD_MODAL_M_ORTHOGONALITY);
    assert_eq!(
        orthogonality.element_count(),
        result.frequencies_hz().len() * result.frequencies_hz().len()
    );
    assert_eq!(
        orthogonality.shape,
        vec![result.frequencies_hz().len(), result.frequencies_hz().len()]
    );
    let residual_norms = host_field(&result.run, FEA_FIELD_MODAL_RESIDUAL_NORM);
    assert_eq!(residual_norms, result.residual_norms().as_slice());
    assert_eq!(result.recovered_modes.len(), result.basis.modes().len());
    assert!(result
        .recovered_modes
        .iter()
        .all(|mode| mode.fields.len() == 2
            && mode.fields[0].field_id == FEA_FIELD_STRUCTURAL_DISPLACEMENT
            && mode.fields[1].field_id == FEA_FIELD_STRUCTURAL_ROTATION));
    assert!(result.run.field(FEA_FIELD_STRUCTURAL_STRESS).is_none());
    assert!(result.run.field(FEA_FIELD_STRUCTURAL_VON_MISES).is_none());
    let encoded = serde_json::to_value(&result.basis).expect("modal basis should serialize");
    let decoded: crate::ModalBasisArtifact =
        serde_json::from_value(encoded.clone()).expect("canonical modal basis should decode");
    assert_eq!(decoded.identity(), result.basis.identity());
    assert_eq!(
        result.basis.truncation().retained_mode_count(),
        result.basis.modes().len()
    );
    assert!(result.basis.truncation().complete_boundary_eigenspace());
    assert!(!result.basis.eigenspaces().is_empty());
    let mut tampered = encoded;
    tampered["modes"][0]["modal_mass"] = serde_json::json!(17.0);
    assert!(serde_json::from_value::<crate::ModalBasisArtifact>(tampered).is_err());
    assert!(result
        .run
        .diagnostics
        .iter()
        .any(|diag| diag.code == "FEA_MODAL_CONVERGENCE"));
}

#[test]
fn modal_truncation_expands_to_a_complete_governed_eigenspace() {
    let mut model = fixture_model(FixtureId::CantileverLinearStatic);
    model.sequence.steps = vec![ProcedureStep::structural_modal("modal_cluster")];
    let result = crate::run_modal_with_options(
        &model,
        ComputeBackend::Cpu,
        ModalSolveOptions {
            truncation: ModalTruncationPolicy::requested(1)
                .with_eigenspace_relative_tolerance(0.999_999),
            ..baseline_modal_options()
        },
    )
    .expect("modal truncation should retain the complete boundary eigenspace");

    assert!(result.basis.modes().len() > 1);
    assert!(result.basis.truncation().expanded_for_eigenspace());
    assert!(result.basis.truncation().complete_boundary_eigenspace());
    assert_eq!(
        result.basis.truncation().available_mode_count(),
        result.basis.modes().len()
    );
    assert!(result
        .run
        .diagnostics
        .iter()
        .any(|diagnostic| diagnostic.code == "FEA_MODAL_TRUNCATION"));
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
        baseline_transient_options(),
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
    assert!(result.strain_snapshots.is_empty());
    assert!(result.stress_snapshots.is_empty());
    assert!(result.von_mises_snapshots.is_empty());
    assert!(result.strain_energy_density_snapshots.is_empty());
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
        result.time_points_s.len(),
        result.reaction_force_snapshots.len()
    );
    assert_eq!(
        result.time_points_s.len(),
        result.external_work_snapshots.len()
    );
    assert_eq!(
        result.time_points_s.len(),
        result.damping_dissipation_snapshots.len()
    );
    assert_eq!(
        result.time_points_s.len(),
        result.energy_balance_residual_snapshots.len()
    );
    assert_eq!(
        result.time_points_s.len(),
        result.relative_energy_balance_error_snapshots.len()
    );
    assert_eq!(
        result.time_points_s.len(),
        result.dynamic_balance_snapshots.len()
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
        result.reaction_force_snapshots[1].field_id,
        fea_transient_reaction_force_field_id(1)
    );
    assert_eq!(
        result.external_work_snapshots[1].field_id,
        fea_transient_external_work_field_id(1)
    );
    assert_eq!(
        result.damping_dissipation_snapshots[1].field_id,
        fea_transient_damping_dissipation_field_id(1)
    );
    assert_eq!(
        result.energy_balance_residual_snapshots[1].field_id,
        fea_transient_energy_balance_residual_field_id(1)
    );
    assert_eq!(
        result.relative_energy_balance_error_snapshots[1].field_id,
        fea_transient_relative_energy_balance_error_field_id(1)
    );
    assert_eq!(
        result.residual_norm_snapshots[1]
            .as_host_f64()
            .expect("residual field should be host-backed")[0],
        result.residual_norms[1]
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
        .find(|diag| diag.code == "FEA_TRANSIENT_MECHANICAL_ENERGY_HISTORY")
        .expect("transient mechanical-energy history diagnostic should be emitted");
    assert!(energy_diag
        .message
        .contains("maximum_strain_plus_kinetic_energy_joules="));
    assert!(energy_diag
        .message
        .contains("conservation_balance_evaluated=true"));
}

#[test]
fn physical_tet4_transient_converges_to_damped_step_response_and_closes_balance() {
    let model = fixture_model(FixtureId::StructuralTransientTet4Reference);
    let mesh = single_tetrahedron_solver_mesh_for_model(&model);
    let step = &model.sequence.steps[0];
    let excitations = crate::excitation::realize_step_excitations(
        &model,
        step,
        crate::excitation::ExcitationSamplePoint::time(0.0),
    )
    .expect("the physical transient excitation should realize");
    let summary = crate::assembly::assemble_structural_system(
        &model,
        &excitations,
        Some(mesh.clone()),
        crate::assembly::StructuralFormulationRequirement::PhysicalDiscretization,
    )
    .expect("the physical Tet4 reference should assemble");
    let free_dofs = summary
        .operator
        .constrained_dofs()
        .iter()
        .enumerate()
        .filter_map(|(dof, constrained)| (!constrained).then_some(dof))
        .collect::<Vec<_>>();
    assert_eq!(free_dofs.len(), 1);
    let free_dof = free_dofs[0];
    let stiffness = summary
        .operator
        .operators()
        .stiffness()
        .get(free_dof, free_dof);
    let mass = summary.operator.operators().mass().get(free_dof, free_dof);
    let damping = summary
        .operator
        .operators()
        .damping()
        .get(free_dof, free_dof);
    let force = summary.operator.load()[free_dof];
    assert!(stiffness > 0.0 && mass > 0.0 && damping > 0.0 && force > 0.0);

    let natural_frequency = (stiffness / mass).sqrt();
    let decay = damping / (2.0 * mass);
    let damped_frequency = (natural_frequency * natural_frequency - decay * decay).sqrt();
    let static_displacement = force / stiffness;
    let exact_at = |time_s: f64| {
        let exponential = (-decay * time_s).exp();
        let phase = damped_frequency * time_s;
        let displacement = static_displacement
            * (1.0 - exponential * (phase.cos() + decay / damped_frequency * phase.sin()));
        let velocity = static_displacement * exponential * natural_frequency * natural_frequency
            / damped_frequency
            * phase.sin();
        let acceleration = (force - damping * velocity - stiffness * displacement) / mass;
        [displacement, velocity, acceleration]
    };
    let exact = exact_at(1.0);

    let mut finest = None;
    let errors = [40, 80, 160].map(|interval_count| {
        let result = crate::run_transient_with_options(
            &model,
            ComputeBackend::Cpu,
            TransientSolveOptions {
                time_step_policy: StructuralTimeStepPolicy::Fixed { interval_count },
                residual_target: 1.0e-10,
                solver_mesh: Some(mesh.clone()),
                ..baseline_transient_options()
            },
        )
        .expect("the physical Tet4 transient should solve");
        assert_eq!(result.time_points_s.len(), interval_count + 1);
        let last = result.time_points_s.len() - 1;
        let actual = [
            result.displacement_snapshots[last]
                .as_host_f64()
                .expect("displacement should be host-backed")[free_dof],
            result.velocity_snapshots[last]
                .as_host_f64()
                .expect("velocity should be host-backed")[free_dof],
            result.acceleration_snapshots[last]
                .as_host_f64()
                .expect("acceleration should be host-backed")[free_dof],
        ];
        let error = ((actual[0] - exact[0]) / static_displacement)
            .hypot((actual[1] - exact[1]) / (static_displacement * natural_frequency))
            .hypot(
                (actual[2] - exact[2])
                    / (static_displacement * natural_frequency * natural_frequency),
            );
        finest = Some(result);
        error
    });
    assert!((errors[0] / errors[1]).log2() > 1.8, "errors={errors:?}");
    assert!((errors[1] / errors[2]).log2() > 1.8, "errors={errors:?}");

    let result = finest.expect("the finest physical reference run should be retained");
    let snapshot_count = result.time_points_s.len();
    assert_eq!(result.dynamic_balance_snapshots.len(), snapshot_count);
    assert_eq!(result.reaction_force_snapshots.len(), snapshot_count);
    assert_eq!(result.external_work_snapshots.len(), snapshot_count);
    assert_eq!(result.damping_dissipation_snapshots.len(), snapshot_count);
    assert_eq!(
        result.energy_balance_residual_snapshots.len(),
        snapshot_count
    );
    assert_eq!(
        result.relative_energy_balance_error_snapshots.len(),
        snapshot_count
    );
    for balance in &result.dynamic_balance_snapshots {
        balance.validate().expect("dynamic balance should validate");
        for dof in 0..balance.equilibrium_residual.len() {
            assert_eq!(
                balance.equilibrium_residual[dof],
                balance.inertia_force[dof]
                    + balance.damping_force[dof]
                    + balance.elastic_force[dof]
                    - balance.applied_load[dof]
            );
            if dof == free_dof {
                assert!(balance.equilibrium_residual[dof].abs() < 1.0e-9);
            }
        }
    }
    let damping_history = result
        .damping_dissipation_snapshots
        .iter()
        .map(|field| field.as_host_f64().unwrap()[0])
        .collect::<Vec<_>>();
    assert!(damping_history.windows(2).all(|pair| pair[1] >= pair[0]));
    let relative_balance = result
        .relative_energy_balance_error_snapshots
        .last()
        .unwrap()
        .as_host_f64()
        .unwrap()[0];
    assert!(
        relative_balance < 1.0e-9,
        "relative balance={relative_balance}"
    );

    let last = snapshot_count - 1;
    let strain_energy = result.strain_energy_snapshots[last].as_host_f64().unwrap()[0];
    let energy_density = result.strain_energy_density_snapshots[last]
        .as_host_f64()
        .unwrap()[0];
    assert!((strain_energy - energy_density / 6.0).abs() < 1.0e-12);
    let reaction = result.reaction_force_snapshots[last].as_host_f64().unwrap();
    let balance = &result.dynamic_balance_snapshots[last];
    for (dof, actual) in reaction.iter().copied().enumerate() {
        let expected = if summary.operator.constrained_dofs()[dof] {
            balance.equilibrium_residual[dof]
        } else {
            0.0
        };
        assert!((actual - expected).abs() < 1.0e-12);
    }
}

#[test]
fn transient_history_schedule_splits_the_solve_and_publishes_exact_samples() {
    let mut model = fixture_model(FixtureId::CantileverLinearStatic);
    let mut step = ProcedureStep::structural_transient("transient_history");
    let requested_times = vec![0.0, 0.25, 0.7, 1.0];
    let mut history_request = nodal_history_observation(
        "displacement_history",
        runmat_analysis_core::ObservationQuantity::Displacement,
        runmat_analysis_core::SamplingPolicy::TimeSeconds(requested_times.clone()),
    );
    let runmat_analysis_core::ObservationRequest::History(spec) = &mut history_request else {
        unreachable!()
    };
    spec.component = runmat_analysis_core::ObservationComponent::Y;
    spec.reduction = Some(runmat_analysis_core::ReductionPolicy {
        spatial: runmat_analysis_core::SpatialReduction::None,
        series: runmat_analysis_core::SeriesReduction::RootMeanSquare {
            weighting: runmat_analysis_core::SeriesWeighting::TrapezoidalIndependentDomain,
        },
        extrema_ties: runmat_analysis_core::ExtremaTiePolicy::FirstCanonical,
    });
    step.observations = vec![history_request];
    model.sequence.steps = vec![step];
    let mesh = single_tetrahedron_solver_mesh_for_model(&model);

    let result = crate::run_transient_with_options(
        &model,
        ComputeBackend::Cpu,
        TransientSolveOptions {
            time_step_policy: StructuralTimeStepPolicy::Fixed { interval_count: 2 },
            solver_mesh: Some(mesh),
            ..baseline_transient_options()
        },
    )
    .expect("an explicit history schedule should become exact solve boundaries");

    assert!(matches!(
        result.initial_state.basis,
        crate::StructuralInitialStateEvidenceBasis::CanonicalTopology { .. }
    ));
    let encoded_initial_state =
        serde_json::to_value(&result.initial_state).expect("serialize physical initial state");
    let decoded_initial_state: crate::StructuralInitialStateEvidence =
        serde_json::from_value(encoded_initial_state.clone())
            .expect("physical initial-state evidence should round-trip");
    assert_eq!(decoded_initial_state, result.initial_state);
    let mut tampered_initial_state = encoded_initial_state;
    tampered_initial_state["zero_default_displacement_dof_count"] =
        serde_json::json!(result.initial_state.total_dof_count + 1);
    assert!(
        serde_json::from_value::<crate::StructuralInitialStateEvidence>(tampered_initial_state)
            .is_err()
    );

    assert!(requested_times
        .iter()
        .all(|time| result.time_points_s.contains(time)));
    assert_eq!(result.run.observation_histories.len(), 1);
    let history = &result.run.observation_histories[0];
    assert_eq!(result.run.series_reductions.len(), 1);
    let series = &result.run.series_reductions[0];
    assert_eq!(series.source_history, history.manifest().identity);
    assert_eq!(
        series.operation,
        runmat_analysis_core::SeriesReduction::RootMeanSquare {
            weighting: runmat_analysis_core::SeriesWeighting::TrapezoidalIndependentDomain,
        }
    );
    series
        .validate()
        .expect("pipeline series reduction should be self-consistent");
    let crate::results::observation::SeriesReductionPayload::Values {
        field: reduced_field,
        ..
    } = &series.payload
    else {
        panic!("RMS history reduction must produce one field")
    };
    assert_eq!(
        result.run.field(&reduced_field.field_id),
        Some(reduced_field)
    );
    assert!(result.run.field_ids().contains(&reduced_field.field_id));
    assert_eq!(history.manifest().observed_sample_count, 4);
    assert_eq!(
        history
            .samples()
            .iter()
            .map(|sample| sample.coordinate)
            .collect::<Vec<_>>(),
        requested_times
            .iter()
            .copied()
            .map(crate::results::field::RecoveredFieldSample::TimeSeconds)
            .collect::<Vec<_>>()
    );
    for sample in history.samples() {
        let crate::results::field::RecoveredFieldSample::TimeSeconds(time_s) = sample.coordinate
        else {
            panic!("transient history must publish time coordinates")
        };
        let snapshot_index = result
            .time_points_s
            .iter()
            .position(|candidate| *candidate == time_s)
            .expect("history coordinate must identify an accepted snapshot");
        let expected_y = result.displacement_snapshots[snapshot_index]
            .as_host_f64()
            .expect("displacement history source should be host-backed")
            .chunks_exact(3)
            .map(|components| components[1])
            .collect::<Vec<_>>();
        assert_eq!(
            sample.result.field.as_host_f64(),
            Some(expected_y.as_slice())
        );
    }
    let encoded = history
        .canonical_bytes()
        .expect("history should serialize canonically");
    let decoded = crate::results::observation::ObservationHistory::from_canonical_bytes(&encoded)
        .expect("history should deserialize through validation");
    assert_eq!(&decoded, history);
    let json = serde_json::to_vec(history).expect("history should serialize as JSON");
    let decoded_json: crate::results::observation::ObservationHistory =
        serde_json::from_slice(&json).expect("history JSON should deserialize through validation");
    assert_eq!(&decoded_json, history);
}

#[test]
fn transient_every_accepted_increment_history_tracks_the_solver_grid() {
    let mut model = fixture_model(FixtureId::CantileverLinearStatic);
    let mut step = ProcedureStep::structural_transient("accepted_increment_history");
    step.accuracy.resources.maximum_samples = 1_000;
    step.observations = vec![nodal_history_observation(
        "accepted_displacement_history",
        runmat_analysis_core::ObservationQuantity::Displacement,
        runmat_analysis_core::SamplingPolicy::EveryAcceptedIncrement,
    )];
    model.sequence.steps = vec![step];
    let mesh = single_tetrahedron_solver_mesh_for_model(&model);

    let result = crate::run_transient_with_options(
        &model,
        ComputeBackend::Cpu,
        TransientSolveOptions {
            time_step_policy: StructuralTimeStepPolicy::Fixed { interval_count: 2 },
            solver_mesh: Some(mesh),
            ..baseline_transient_options()
        },
    )
    .expect("every-accepted-increment history should follow accepted solve points");

    let history = &result.run.observation_histories[0];
    assert_eq!(
        history
            .samples()
            .iter()
            .map(|sample| sample.coordinate)
            .collect::<Vec<_>>(),
        result
            .time_points_s
            .iter()
            .copied()
            .map(crate::results::field::RecoveredFieldSample::TimeSeconds)
            .collect::<Vec<_>>()
    );
}

#[test]
fn solver_pipeline_without_an_event_detector_rejects_event_histories() {
    let mut model = fixture_model(FixtureId::CantileverLinearStatic);
    let mut step = ProcedureStep::structural_transient("unsupported_transient_event");
    step.observations = vec![nodal_history_observation(
        "transient_first_yield",
        runmat_analysis_core::ObservationQuantity::Displacement,
        runmat_analysis_core::SamplingPolicy::Event(
            runmat_analysis_core::ObservationEvent::Solver(
                runmat_analysis_core::SolverObservationEvent::FirstYield,
            ),
        ),
    )];
    model.sequence.steps = vec![step];
    let mesh = single_tetrahedron_solver_mesh_for_model(&model);

    let error = crate::run_transient_with_options(
        &model,
        ComputeBackend::Cpu,
        TransientSolveOptions {
            time_step_policy: StructuralTimeStepPolicy::Fixed { interval_count: 2 },
            solver_mesh: Some(mesh),
            ..baseline_transient_options()
        },
    )
    .expect_err("a pipeline cannot publish an empty history for a detector it did not run");
    assert!(error.to_string().contains("does not own an event detector"));
}

#[test]
fn nonlinear_history_schedule_splits_load_increments_and_retains_the_initial_state() {
    let mut model = fixture_model(FixtureId::CantileverLinearStatic);
    let mut step = ProcedureStep::structural_nonlinear("nonlinear_history");
    let requested_load_factors = vec![0.0, 0.3, 1.0];
    step.observations = vec![nodal_history_observation(
        "nonlinear_displacement_history",
        runmat_analysis_core::ObservationQuantity::Displacement,
        runmat_analysis_core::SamplingPolicy::LoadFactors(requested_load_factors.clone()),
    )];
    model.sequence.steps = vec![step];
    let mesh = single_tetrahedron_solver_mesh_for_model(&model);

    let result = crate::run_nonlinear_with_options(
        &model,
        ComputeBackend::Cpu,
        NonlinearSolveOptions {
            increment_count: 2,
            solver_mesh: Some(mesh),
            ..baseline_nonlinear_options()
        },
    )
    .expect("an explicit load-factor history should become exact increment boundaries");

    assert_eq!(result.load_factors, vec![0.3, 0.8, 1.0]);
    assert_eq!(result.run.observation_histories.len(), 1);
    let history = &result.run.observation_histories[0];
    assert_eq!(history.manifest().observed_sample_count, 3);
    assert_eq!(
        history
            .samples()
            .iter()
            .map(|sample| sample.coordinate)
            .collect::<Vec<_>>(),
        requested_load_factors
            .iter()
            .copied()
            .map(crate::results::field::RecoveredFieldSample::LoadFactor)
            .collect::<Vec<_>>()
    );
    let initial_values = history.samples()[0]
        .result
        .field
        .as_host_f64()
        .expect("initial displacement history should be host-backed");
    assert!(initial_values.iter().all(|value| *value == 0.0));
    for sample in &history.samples()[1..] {
        let crate::results::field::RecoveredFieldSample::LoadFactor(load_factor) =
            sample.coordinate
        else {
            panic!("nonlinear history must publish load-factor coordinates")
        };
        let snapshot_index = result
            .load_factors
            .iter()
            .position(|candidate| *candidate == load_factor)
            .expect("history coordinate must identify an accepted increment");
        assert_eq!(
            sample.result.field.as_host_f64(),
            result.displacement_snapshots[snapshot_index].as_host_f64()
        );
    }
}

#[test]
fn nonlinear_events_require_a_physical_basis() {
    let mut model = fixture_model(FixtureId::CantileverLinearStatic);
    let mut step = ProcedureStep::structural_nonlinear("unsupported_nonlinear");
    step.observations = vec![nodal_history_observation(
        "first_yield_displacement",
        runmat_analysis_core::ObservationQuantity::Displacement,
        runmat_analysis_core::SamplingPolicy::Event(
            runmat_analysis_core::ObservationEvent::Solver(
                runmat_analysis_core::SolverObservationEvent::FirstYield,
            ),
        ),
    )];
    model.sequence.steps = vec![step];
    let error = crate::run_nonlinear_with_options(
        &model,
        ComputeBackend::Cpu,
        baseline_nonlinear_options(),
    )
    .expect_err("unsupported envelopes must not claim J2 continuum equilibrium");
    assert!(matches!(
        error,
        crate::FeaRunError::InvalidModel(_)
            | crate::FeaRunError::Assembly(_)
            | crate::FeaRunError::FormulationAdmission(_)
    ));
}

#[test]
fn modal_history_publishes_requested_mode_shapes_with_mode_coordinates() {
    let mut model = fixture_model(FixtureId::CantileverLinearStatic);
    let mut step = ProcedureStep::structural_modal("modal_history");
    step.observations = vec![
        nodal_history_observation(
            "modal_displacement_history",
            runmat_analysis_core::ObservationQuantity::Displacement,
            runmat_analysis_core::SamplingPolicy::ModeIndices(vec![1, 2]),
        ),
        history_observation(
            "modal_stress_history",
            runmat_analysis_core::ObservationQuantity::Stress,
            runmat_analysis_core::SamplingPolicy::ModeIndices(vec![1, 2]),
            runmat_analysis_core::FieldLocation::Element,
        ),
    ];
    model.sequence.steps = vec![step];
    let mesh = single_tetrahedron_solver_mesh_for_model(&model);

    let result = crate::run_modal_with_options(
        &model,
        ComputeBackend::Cpu,
        ModalSolveOptions {
            truncation: ModalTruncationPolicy::requested(2),
            solver_mesh: Some(mesh),
            ..baseline_modal_options()
        },
    )
    .expect("requested modal histories should publish solved mode shapes");

    assert_eq!(result.run.observation_histories.len(), 2);
    let history = result
        .run
        .observation_histories
        .iter()
        .find(|history| {
            history.manifest().quantity == runmat_analysis_core::ObservationQuantity::Displacement
        })
        .expect("modal displacement history");
    assert_eq!(history.manifest().observed_sample_count, 2);
    assert_eq!(
        history
            .samples()
            .iter()
            .map(|sample| sample.coordinate)
            .collect::<Vec<_>>(),
        vec![
            crate::results::field::RecoveredFieldSample::ModeIndex(1),
            crate::results::field::RecoveredFieldSample::ModeIndex(2),
        ]
    );
    for (sample, mode_shape) in history.samples().iter().zip(&result.mode_shapes) {
        assert_eq!(sample.result.field.as_host_f64(), mode_shape.as_host_f64());
    }
    let stress = result
        .run
        .observation_histories
        .iter()
        .find(|history| {
            history.manifest().quantity == runmat_analysis_core::ObservationQuantity::Stress
        })
        .expect("modal stress history");
    assert_eq!(stress.samples().len(), 2);
    assert!(stress
        .samples()
        .iter()
        .all(|sample| sample.result.field.shape == vec![1, 6]));
}

#[test]
fn modal_beam_history_separates_rotation_from_translation() {
    let mut model = fixture_model(FixtureId::StructuralBeamCantileverEndMomentReference);
    let mut step = ProcedureStep::structural_modal("beam_modal_history");
    step.observations = vec![
        nodal_history_observation(
            "beam_modal_displacement",
            runmat_analysis_core::ObservationQuantity::Displacement,
            runmat_analysis_core::SamplingPolicy::ModeIndices(vec![1, 2]),
        ),
        nodal_history_observation(
            "beam_modal_rotation",
            runmat_analysis_core::ObservationQuantity::Rotation,
            runmat_analysis_core::SamplingPolicy::ModeIndices(vec![1, 2]),
        ),
    ];
    model.sequence.steps = vec![step];

    let result = crate::run_modal_with_options(
        &model,
        ComputeBackend::Cpu,
        ModalSolveOptions {
            truncation: ModalTruncationPolicy::requested(2),
            ..baseline_modal_options()
        },
    )
    .expect("explicit beam modes should publish typed translation and rotation histories");

    assert_eq!(result.mode_shapes[0].shape, vec![2, 3]);
    assert_eq!(result.run.observation_histories.len(), 2);
    let displacement = result
        .run
        .observation_histories
        .iter()
        .find(|history| {
            history.manifest().quantity == runmat_analysis_core::ObservationQuantity::Displacement
        })
        .expect("modal displacement history should be present");
    let rotation = result
        .run
        .observation_histories
        .iter()
        .find(|history| {
            history.manifest().quantity == runmat_analysis_core::ObservationQuantity::Rotation
        })
        .expect("modal rotation history should be present");
    assert_eq!(
        displacement.samples()[0].result.field.as_host_f64(),
        result.mode_shapes[0].as_host_f64()
    );
    assert!(rotation.samples().iter().any(|sample| {
        sample
            .result
            .field
            .as_host_f64()
            .expect("modal rotation should be host-backed")
            .iter()
            .any(|value| value.abs() > 0.0)
    }));
}

#[test]
fn modal_beam_recovers_the_closed_form_lumped_mass_axial_frequency() {
    let mut model = fixture_model(FixtureId::StructuralBeamCantileverEndMomentReference);
    model.sequence.steps = vec![ProcedureStep::structural_modal("beam_modal_reference")];
    let policy = crate::ModalConvergencePolicy::production();
    let result = crate::run_modal_with_options(
        &model,
        ComputeBackend::Cpu,
        ModalSolveOptions {
            truncation: ModalTruncationPolicy::requested(6),
            convergence: policy,
            ..baseline_modal_options()
        },
    )
    .expect("one-element beam modal reference should solve");

    let expected_hz = (2.0 * 200.0e9_f64 / 7_850.0).sqrt() / (2.0 * std::f64::consts::PI);
    let axial_hz = result
        .frequencies_hz()
        .into_iter()
        .min_by(|left, right| {
            (left - expected_hz)
                .abs()
                .total_cmp(&(right - expected_hz).abs())
        })
        .expect("beam should have modes");
    assert!((axial_hz - expected_hz).abs() / expected_hz <= 1.0e-10);
    assert!(result
        .residual_norms()
        .iter()
        .all(|residual| *residual <= policy.relative_residual_tolerance()));
}

#[test]
fn thermal_history_schedule_adds_exact_solve_points_and_publishes_temperature() {
    let mut model = fixture_model(FixtureId::CantileverLinearStatic);
    model.thermal = Some(runmat_analysis_core::ThermalDomain {
        reference_temperature: runmat_analysis_core::Temperature::new(293.15),
    });
    let mut step = ProcedureStep::thermal_transient("thermal_history");
    let requested_times = vec![0.0, 0.2, 0.65, 1.0];
    step.observations = vec![
        nodal_history_observation(
            "temperature_history",
            runmat_analysis_core::ObservationQuantity::Temperature,
            runmat_analysis_core::SamplingPolicy::TimeSeconds(requested_times.clone()),
        ),
        history_observation(
            "temperature_gradient_history",
            runmat_analysis_core::ObservationQuantity::TemperatureGradient,
            runmat_analysis_core::SamplingPolicy::TimeSeconds(requested_times.clone()),
            runmat_analysis_core::FieldLocation::Element,
        ),
        history_observation(
            "heat_flux_history",
            runmat_analysis_core::ObservationQuantity::HeatFlux,
            runmat_analysis_core::SamplingPolicy::TimeSeconds(requested_times.clone()),
            runmat_analysis_core::FieldLocation::Element,
        ),
        history_observation(
            "storage_rate_history",
            runmat_analysis_core::ObservationQuantity::ThermalEnergyStorageRateDensity,
            runmat_analysis_core::SamplingPolicy::TimeSeconds(requested_times.clone()),
            runmat_analysis_core::FieldLocation::Element,
        ),
    ];
    model.sequence.steps = vec![step];
    let mesh = single_tetrahedron_solver_mesh_for_model(&model);

    let result = crate::run_thermal_with_options(
        &model,
        ComputeBackend::Cpu,
        ThermalSolveOptions {
            procedure: ThermalProcedureControl::BackwardEuler { snapshot_count: 3 },
            solver_mesh: Some(mesh),
            ..baseline_thermal_options()
        },
    )
    .expect("thermal history samples should become exact solve points");

    assert!(requested_times
        .iter()
        .all(|time| result.time_points_s.contains(time)));
    assert_eq!(result.run.observation_histories.len(), 4);
    for (quantity, snapshots) in [
        (
            runmat_analysis_core::ObservationQuantity::Temperature,
            &result.temperature_snapshots,
        ),
        (
            runmat_analysis_core::ObservationQuantity::TemperatureGradient,
            &result.temperature_gradient_snapshots,
        ),
        (
            runmat_analysis_core::ObservationQuantity::HeatFlux,
            &result.heat_flux_snapshots,
        ),
        (
            runmat_analysis_core::ObservationQuantity::ThermalEnergyStorageRateDensity,
            &result.energy_storage_rate_snapshots,
        ),
    ] {
        let history = result
            .run
            .observation_histories
            .iter()
            .find(|history| history.manifest().quantity == quantity)
            .expect("each requested thermal quantity should own one history");
        assert_eq!(history.manifest().observed_sample_count, 4);
        for sample in history.samples() {
            let crate::results::field::RecoveredFieldSample::TimeSeconds(time_s) =
                sample.coordinate
            else {
                panic!("thermal history must publish time coordinates")
            };
            let snapshot_index = result
                .time_points_s
                .iter()
                .position(|candidate| *candidate == time_s)
                .expect("thermal history coordinate must identify a solve point");
            assert_eq!(
                sample.result.field.as_host_f64(),
                snapshots[snapshot_index].as_host_f64()
            );
        }
    }
}

#[test]
fn structural_transient_inserts_active_jump_without_extending_the_horizon() {
    let mut model = fixture_model(FixtureId::CantileverLinearStatic);
    model.sequence.steps = vec![ProcedureStep::structural_transient(
        "transient_jump".to_string(),
    )];
    let runmat_analysis_core::PhysicalSignal::Force(signal) = &mut model.signals[0] else {
        unreachable!()
    };
    let ordinate = |y: f64| {
        runmat_analysis_core::SignalOrdinate::Real(runmat_analysis_core::SignalValue::Vector3([
            runmat_analysis_core::Force::new(0.0),
            runmat_analysis_core::Force::new(y),
            runmat_analysis_core::Force::new(0.0),
        ]))
    };
    signal.representation = runmat_analysis_core::SignalRepresentation::Tabulated(
        runmat_analysis_core::TabulatedSignal {
            domain: runmat_analysis_core::TabulatedSignalDomain::TimeSeconds,
            knots: vec![
                runmat_analysis_core::SignalKnot::continuous(0.0, ordinate(0.0)),
                runmat_analysis_core::SignalKnot::jump(
                    0.5,
                    ordinate(0.0),
                    ordinate(-1000.0),
                    runmat_analysis_core::SignalPointContinuity::Right,
                ),
                runmat_analysis_core::SignalKnot::continuous(1.2, ordinate(-1000.0)),
            ],
            interpolation: runmat_analysis_core::SignalInterpolation::LeftEndpoint,
            extrapolation: runmat_analysis_core::SignalExtrapolation::Reject,
        },
    );
    signal.evidence.sample_count = 3;
    let options = TransientSolveOptions {
        time_step_policy: StructuralTimeStepPolicy::Fixed { interval_count: 2 },
        ..baseline_transient_options()
    };

    let result = crate::run_transient_with_options(&model, ComputeBackend::Cpu, options)
        .expect("jump-aware structural transient should solve");

    assert_eq!(result.time_points_s.last().copied(), Some(1.0));
    assert!(result.time_points_s.contains(&0.5));
    assert!(result
        .time_points_s
        .windows(2)
        .all(|window| !(window[0] < 0.5 && window[1] > 0.5)));
}

#[test]
fn transient_beam_moment_emits_rotational_snapshots() {
    let mut model = fixture_model(FixtureId::StructuralBeamCantileverEndMomentReference);
    let mut step = ProcedureStep::structural_transient("beam_moment_transient".to_string());
    step.accuracy.resources.maximum_samples = 1_000;
    step.observations = vec![nodal_history_observation(
        "beam_rotation_history",
        runmat_analysis_core::ObservationQuantity::Rotation,
        runmat_analysis_core::SamplingPolicy::EveryAcceptedIncrement,
    )];
    model.sequence.steps = vec![step];
    let result = crate::run_transient_with_options(
        &model,
        ComputeBackend::Cpu,
        baseline_transient_options(),
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
    let history = result
        .run
        .observation_histories
        .iter()
        .find(|history| {
            history.manifest().quantity == runmat_analysis_core::ObservationQuantity::Rotation
        })
        .expect("beam transient should publish the requested rotation history");
    assert_eq!(history.samples().len(), result.rotation_snapshots.len());
    for (sample, snapshot) in history.samples().iter().zip(&result.rotation_snapshots) {
        assert_eq!(sample.result.field.as_host_f64(), snapshot.as_host_f64());
    }
}

#[test]
fn thermal_solver_emits_heat_transfer_fields() {
    let mut model = fixture_model(FixtureId::CantileverLinearStatic);
    model.thermal = Some(runmat_analysis_core::ThermalDomain {
        reference_temperature: runmat_analysis_core::Temperature::new(293.15),
    });
    model.sequence = runmat_analysis_core::AnalysisSequence::from_steps(
        "thermal_sequence",
        vec![runmat_analysis_core::ProcedureStep::thermal_transient(
            "thermal_step",
        )],
    );
    let mesh = runmat_meshing_core::fixtures::canonical_tetrahedron_solver_mesh(ElementOrder::Tet4);
    let result = crate::run_thermal_with_options(
        &model,
        ComputeBackend::Cpu,
        ThermalSolveOptions {
            procedure: ThermalProcedureControl::BackwardEuler { snapshot_count: 4 },
            solver_mesh: Some(mesh),
            ..baseline_thermal_options()
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
    assert!(result
        .run
        .diagnostics
        .iter()
        .any(|diag| diag.code == "FEA_THERMAL_FIELD_RECOVERY"
            && diag.message.contains("basis=physical_quadrature")
            && diag.message.contains("node_count=4")
            && diag.message.contains("element_count=1")
            && diag.message.contains("boundary_face_count=4")));
}

#[test]
fn steady_thermal_solver_is_a_distinct_static_procedure() {
    let mut model = fixture_model(FixtureId::CantileverLinearStatic);
    model.thermal = Some(runmat_analysis_core::ThermalDomain {
        reference_temperature: runmat_analysis_core::Temperature::new(293.15),
    });
    model.sequence = runmat_analysis_core::AnalysisSequence::from_steps(
        "steady_thermal_sequence",
        vec![runmat_analysis_core::ProcedureStep::thermal_steady(
            "steady_thermal_step",
        )],
    );
    let mesh = runmat_meshing_core::fixtures::canonical_tetrahedron_solver_mesh(ElementOrder::Tet4);
    let projection = crate::topology::AnalysisProjection::derive(&mesh).unwrap();
    let fixed_selector = projection.boundary_faces[0].provenance[0]
        .analysis_selection_id()
        .unwrap();
    model
        .constraints
        .push(runmat_analysis_core::AnalysisConstraint::Thermal(
            runmat_analysis_core::ThermalConstraint {
                constraint_id: "steady_temperature".into(),
                kind: runmat_analysis_core::ThermalConstraintKind::Prescribed {
                    target: runmat_analysis_core::RegionTarget::boundary(
                        runmat_meshing_core::fixtures::fixture_boundary_selection(
                            &(fixed_selector),
                        ),
                    ),
                    value: runmat_analysis_core::PrescribedValue::Constant(
                        runmat_analysis_core::Temperature::new(315.0),
                    ),
                },
            },
        ));
    let result = crate::run_thermal_with_options(
        &model,
        ComputeBackend::Cpu,
        ThermalSolveOptions {
            procedure: ThermalProcedureControl::Steady,
            solver_mesh: Some(mesh),
            ..baseline_thermal_options()
        },
    )
    .expect("steady thermal solve should succeed");

    assert_eq!(result.run.identity.procedure(), AnalysisProcedure::Static);
    assert_eq!(result.time_points_s, vec![0.0]);
    assert_eq!(result.temperature_snapshots.len(), 1);
    assert_eq!(result.residual_norms.len(), 1);
    assert!(result.temperature_snapshots[0]
        .as_host_f64()
        .unwrap()
        .iter()
        .all(|temperature| (*temperature - 315.0).abs() <= 1.0e-9));
}

#[test]
fn transient_thermal_pipeline_samples_prescribed_temperature_at_each_time_point() {
    let mut model = fixture_model(FixtureId::CantileverLinearStatic);
    model.thermal = Some(runmat_analysis_core::ThermalDomain {
        reference_temperature: runmat_analysis_core::Temperature::new(293.15),
    });
    model.sequence = runmat_analysis_core::AnalysisSequence::from_steps(
        "thermal_signal_sequence",
        vec![runmat_analysis_core::ProcedureStep::thermal_transient(
            "thermal_signal_step",
        )],
    );
    let mesh = runmat_meshing_core::fixtures::canonical_tetrahedron_solver_mesh(ElementOrder::Tet4);
    let projection = crate::topology::AnalysisProjection::derive(&mesh).unwrap();
    let prescribed_face = &projection.boundary_faces[0];
    let prescribed_selector = prescribed_face.provenance[0]
        .analysis_selection_id()
        .unwrap();
    let signal_id = runmat_analysis_core::SignalId("prescribed_temperature_history".into());
    model
        .signals
        .push(runmat_analysis_core::PhysicalSignal::Temperature(
            runmat_analysis_core::Signal {
                signal_id: signal_id.clone(),
                shape: runmat_analysis_core::SignalShape::Scalar,
                frame: None,
                representation: runmat_analysis_core::SignalRepresentation::Tabulated(
                    runmat_analysis_core::TabulatedSignal {
                        domain: runmat_analysis_core::TabulatedSignalDomain::TimeSeconds,
                        knots: vec![
                            runmat_analysis_core::SignalKnot::continuous(
                                0.0,
                                runmat_analysis_core::SignalOrdinate::Real(
                                    runmat_analysis_core::SignalValue::Scalar(
                                        runmat_analysis_core::Temperature::new(300.0),
                                    ),
                                ),
                            ),
                            runmat_analysis_core::SignalKnot::jump(
                                0.5,
                                runmat_analysis_core::SignalOrdinate::Real(
                                    runmat_analysis_core::SignalValue::Scalar(
                                        runmat_analysis_core::Temperature::new(300.0),
                                    ),
                                ),
                                runmat_analysis_core::SignalOrdinate::Real(
                                    runmat_analysis_core::SignalValue::Scalar(
                                        runmat_analysis_core::Temperature::new(320.0),
                                    ),
                                ),
                                runmat_analysis_core::SignalPointContinuity::Right,
                            ),
                            runmat_analysis_core::SignalKnot::continuous(
                                1.0,
                                runmat_analysis_core::SignalOrdinate::Real(
                                    runmat_analysis_core::SignalValue::Scalar(
                                        runmat_analysis_core::Temperature::new(320.0),
                                    ),
                                ),
                            ),
                        ],
                        interpolation: runmat_analysis_core::SignalInterpolation::LeftEndpoint,
                        extrapolation: runmat_analysis_core::SignalExtrapolation::Reject,
                    },
                ),
                precision: runmat_analysis_core::SignalPrecision::Real64,
                evidence: runmat_analysis_core::SignalApproximationEvidence {
                    max_absolute_error: Some(0.0),
                    max_relative_error: Some(0.0),
                    derivative_max_absolute_error: None,
                    sample_count: 3,
                },
                provenance: runmat_analysis_core::SignalProvenance::Authored,
            },
        ));
    model
        .constraints
        .push(runmat_analysis_core::AnalysisConstraint::Thermal(
            runmat_analysis_core::ThermalConstraint {
                constraint_id: "time_temperature".into(),
                kind: runmat_analysis_core::ThermalConstraintKind::Prescribed {
                    target: runmat_analysis_core::RegionTarget::boundary(
                        runmat_meshing_core::fixtures::fixture_boundary_selection(
                            &(prescribed_selector.clone()),
                        ),
                    ),
                    value: runmat_analysis_core::PrescribedValue::Signal(signal_id),
                },
            },
        ));
    let mut inactive_signal = model.signals.last().unwrap().clone();
    let runmat_analysis_core::PhysicalSignal::Temperature(inactive) = &mut inactive_signal else {
        unreachable!()
    };
    inactive.signal_id = runmat_analysis_core::SignalId("inactive_temperature_history".into());
    let runmat_analysis_core::SignalRepresentation::Tabulated(inactive_table) =
        &mut inactive.representation
    else {
        unreachable!()
    };
    inactive_table.knots[1].abscissa = 0.25;
    let inactive_id = inactive.signal_id.clone();
    model.signals.push(inactive_signal);
    model
        .constraints
        .push(runmat_analysis_core::AnalysisConstraint::Thermal(
            runmat_analysis_core::ThermalConstraint {
                constraint_id: "inactive_time_temperature".into(),
                kind: runmat_analysis_core::ThermalConstraintKind::Prescribed {
                    target: runmat_analysis_core::RegionTarget::boundary(
                        runmat_meshing_core::fixtures::fixture_boundary_selection(
                            &(prescribed_selector),
                        ),
                    ),
                    value: runmat_analysis_core::PrescribedValue::Signal(inactive_id),
                },
            },
        ));
    model.sequence.steps[0].activations = vec![runmat_analysis_core::StepObjectActivation {
        object: runmat_analysis_core::StepObjectRef::Constraint("time_temperature".into()),
        lifecycle: runmat_analysis_core::ActivationLifecycle::Activate,
    }];

    let result = crate::run_thermal_with_options(
        &model,
        ComputeBackend::Cpu,
        ThermalSolveOptions {
            procedure: ThermalProcedureControl::BackwardEuler { snapshot_count: 2 },
            solver_mesh: Some(mesh),
            ..baseline_thermal_options()
        },
    )
    .expect("time-dependent prescribed temperature should solve");

    assert_eq!(result.time_points_s, vec![0.0, 0.5, 1.0]);
    for (snapshot, expected_temperature_k) in result
        .temperature_snapshots
        .iter()
        .zip([300.0, 300.0, 320.0])
    {
        let values = snapshot.as_host_f64().unwrap();
        for &node in &prescribed_face.node_indices {
            assert!((values[node] - expected_temperature_k).abs() <= 1.0e-12);
        }
    }
}

#[test]
fn canonical_mesh_thermal_pipeline_enforces_exact_boundaries_and_reports_energy_closure() {
    let mut model = fixture_model(FixtureId::CantileverLinearStatic);
    model.thermal = Some(runmat_analysis_core::ThermalDomain {
        reference_temperature: runmat_analysis_core::Temperature::new(293.15),
    });
    model.sequence = runmat_analysis_core::AnalysisSequence::from_steps(
        "thermal_sequence",
        vec![runmat_analysis_core::ProcedureStep::thermal_transient(
            "thermal_step",
        )],
    );
    let mesh = runmat_meshing_core::fixtures::canonical_tetrahedron_solver_mesh(ElementOrder::Tet4);
    let projection = crate::topology::AnalysisProjection::derive(&mesh).unwrap();
    let fixed_face = &projection.boundary_faces[0];
    let heated_face = &projection.boundary_faces[1];
    let fixed_selector = fixed_face.provenance[0].analysis_selection_id().unwrap();
    let heated_selector = heated_face.provenance[0].analysis_selection_id().unwrap();
    model
        .constraints
        .push(runmat_analysis_core::AnalysisConstraint::Thermal(
            runmat_analysis_core::ThermalConstraint {
                constraint_id: "fixed_temperature".into(),
                kind: runmat_analysis_core::ThermalConstraintKind::Prescribed {
                    target: runmat_analysis_core::RegionTarget::boundary(
                        runmat_meshing_core::fixtures::fixture_boundary_selection(
                            &(fixed_selector),
                        ),
                    ),
                    value: runmat_analysis_core::PrescribedValue::Constant(
                        runmat_analysis_core::Temperature::new(293.15),
                    ),
                },
            },
        ));
    model
        .boundaries
        .push(runmat_analysis_core::AnalysisBoundary {
            boundary_id: "applied_heat_flux".into(),
            target: runmat_analysis_core::RegionTarget::boundary(
                runmat_meshing_core::fixtures::fixture_boundary_selection(&(heated_selector)),
            ),
            operator: runmat_analysis_core::BoundaryOperator::ThermalHeatFlux {
                value: runmat_analysis_core::PrescribedValue::Constant(
                    runmat_analysis_core::HeatFlux::new(600.0),
                ),
            },
        });

    let result = crate::run_thermal_with_options(
        &model,
        ComputeBackend::Cpu,
        ThermalSolveOptions {
            procedure: ThermalProcedureControl::BackwardEuler { snapshot_count: 3 },
            solver_mesh: Some(mesh),
            ..baseline_thermal_options()
        },
    )
    .unwrap();

    let final_temperature = result
        .temperature_snapshots
        .last()
        .and_then(runmat_analysis_core::AnalysisField::as_host_f64)
        .unwrap();
    for &node in &fixed_face.node_indices {
        assert!((final_temperature[node] - 293.15).abs() <= 1.0e-12);
    }
    assert!(final_temperature
        .iter()
        .any(|temperature| *temperature > 293.15));
    assert_eq!(result.energy_balances.len(), 2);
    for balance in &result.energy_balances {
        balance.validate().unwrap();
        assert!(balance.applied_power_w.canonical_value() > 0.0);
        assert!(balance.prescribed_boundary_power_w.canonical_value().abs() > 0.0);
        assert!(balance.relative_residual <= result.residual_target);
    }
    assert!(result
        .run
        .identity
        .realization()
        .components
        .iter()
        .any(|component| {
            component.spatial_operator
                == runmat_analysis_core::SpatialOperatorImplementation::ThermalGalerkinConduction
        }));
    let initial_state = &result.initial_state;
    assert_eq!(initial_state.procedure_step_id, "thermal_step");
    assert_eq!(
        initial_state.prescribed_node_count,
        fixed_face.node_indices.len()
    );
    assert_eq!(
        initial_state.reference_node_count,
        final_temperature.len() - fixed_face.node_indices.len()
    );
    assert_eq!(
        result.run.solver_method,
        crate::contracts::SolverMethod::SparseDirectCholesky
    );
    assert!(!result
        .run
        .diagnostics
        .iter()
        .any(|diagnostic| diagnostic.code == "FEA_THERMAL_OUTCOME"));
}

#[test]
fn canonical_mesh_thermal_pipeline_starts_from_the_authored_temperature_state() {
    let mut model = fixture_model(FixtureId::CantileverLinearStatic);
    model.thermal = Some(runmat_analysis_core::ThermalDomain {
        reference_temperature: runmat_analysis_core::Temperature::new(293.15),
    });
    model.sequence = runmat_analysis_core::AnalysisSequence::from_steps(
        "thermal_sequence",
        vec![runmat_analysis_core::ProcedureStep::thermal_transient(
            "thermal_step",
        )],
    );
    model.initial_conditions = vec![runmat_analysis_core::InitialCondition::Temperature(
        runmat_analysis_core::InitialConditionValue {
            initial_condition_id: "uniform_start".into(),
            target: runmat_analysis_core::InitialConditionTarget::WholeModel,
            frame: runmat_analysis_core::CoordinateFrame::Global,
            value: runmat_analysis_core::InitialValue::Uniform(
                runmat_analysis_core::Temperature::new(317.0),
            ),
            applies_before_step_id: "thermal_step".into(),
        },
    )];
    let mesh = runmat_meshing_core::fixtures::canonical_tetrahedron_solver_mesh(ElementOrder::Tet4);

    let result = crate::run_thermal_with_options(
        &model,
        ComputeBackend::Cpu,
        ThermalSolveOptions {
            procedure: ThermalProcedureControl::BackwardEuler { snapshot_count: 3 },
            solver_mesh: Some(mesh),
            ..baseline_thermal_options()
        },
    )
    .unwrap();

    for snapshot in &result.temperature_snapshots {
        assert!(snapshot
            .as_host_f64()
            .unwrap()
            .iter()
            .all(|temperature| (*temperature - 317.0).abs() <= 1.0e-9));
    }
    let evidence = result.initial_state;
    assert_eq!(evidence.applied_initial_condition_ids, ["uniform_start"]);
    assert_eq!(evidence.authored_node_count, 4);
    assert_eq!(evidence.reference_node_count, 0);
}

#[test]
fn modal_large_fixture_emits_orthogonality_and_separation_diagnostics() {
    let model = fixture_model(FixtureId::ModalLarge);
    let result = crate::run_modal_with_options(
        &model,
        ComputeBackend::Cpu,
        ModalSolveOptions {
            truncation: ModalTruncationPolicy::requested(8),
            convergence: crate::ModalConvergencePolicy::default(),
            residual_vector_directions: Vec::new(),
            solver_mesh: None,
        },
    )
    .expect("modal large fixture should solve");

    assert!(!result.frequencies_hz().is_empty());
    assert_eq!(result.frequencies_hz().len(), result.mode_shapes.len());
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
            time_step_policy: StructuralTimeStepPolicy::Fixed { interval_count: 24 },
            ..baseline_transient_options()
        },
    )
    .expect("transient long fixture should solve");

    assert!(result.time_points_s.len() > 8);
    assert!(result.run.diagnostics.iter().any(|diag| {
        diag.code == "FEA_TRANSIENT_METHOD"
            && diag
                .message
                .contains("solver=newmark_average_acceleration_pcg")
            && diag.message.contains("beta=0.25")
            && diag.message.contains("gamma=0.5")
    }));
    assert!(result.run.diagnostics.iter().any(|diag| {
        diag.code == "FEA_TRANSIENT_STABILITY"
            && diag.message.contains("time_step_policy=fixed")
            && diag.message.contains("max_projected_dynamic_residual=")
    }));
    assert!(result
        .run
        .diagnostics
        .iter()
        .any(|diag| diag.code == "FEA_TRANSIENT_ENERGY"));
}

#[test]
fn transient_shock_fixture_emits_method_and_stability_diagnostics() {
    let model = fixture_model(FixtureId::TransientShock);
    let result = crate::run_transient_with_options(
        &model,
        ComputeBackend::Cpu,
        TransientSolveOptions {
            time_step_policy: StructuralTimeStepPolicy::Fixed { interval_count: 48 },
            ..baseline_transient_options()
        },
    )
    .expect("transient shock fixture should solve");

    assert!(result.time_points_s.len() > 24);
    assert!(result
        .run
        .diagnostics
        .iter()
        .any(|diag| diag.code == "FEA_TRANSIENT_METHOD"
            && diag
                .message
                .contains("solver=newmark_average_acceleration_pcg")
            && diag.message.contains("beta=0.25")
            && diag.message.contains("gamma=0.5")));
    assert!(result.run.diagnostics.iter().any(|diag| {
        diag.code == "FEA_TRANSIENT_STABILITY"
            && diag.message.contains("time_step_policy=fixed")
            && diag.message.contains("max_projected_dynamic_residual=")
    }));
}

#[test]
fn thermo_mechanical_transient_fails_closed_without_a_coupled_thermal_state() {
    let mut model = fixture_model(FixtureId::ThermoMechanicalKickoff);
    configure_thermo_mechanical_excitation(&mut model, 65.0, 1.2e-5);
    model.sequence.steps = vec![ProcedureStep::coupled(
        "thermomechanical_transient",
        runmat_analysis_core::CoupledPhysicsFamily::Thermomechanical,
        runmat_analysis_core::AnalysisProcedure::TransientResponse,
    )];
    let error = crate::run_transient_with_options(
        &model,
        ComputeBackend::Cpu,
        TransientSolveOptions {
            time_step_policy: StructuralTimeStepPolicy::Fixed { interval_count: 24 },
            ..baseline_transient_options()
        },
    )
    .expect_err("thermo-mechanical transient requires a solver-owned coupled thermal state");

    assert!(matches!(
        error,
        crate::FeaRunError::FormulationAvailability(rejection)
            if rejection.reason
                == runmat_analysis_core::FormulationAdmissionRejectionReason::UnsupportedFamilyProcedure
    ));
}

#[test]
fn structural_transient_rejects_unimplemented_electrothermal_formulation() {
    let mut model = fixture_model(FixtureId::ElectroThermalJouleBenign);
    configure_electro_thermal_excitation(&mut model);
    let error = crate::run_transient_with_options(
        &model,
        ComputeBackend::Cpu,
        TransientSolveOptions {
            time_step_policy: StructuralTimeStepPolicy::Fixed { interval_count: 12 },
            ..baseline_transient_options()
        },
    )
    .expect_err("structural transient must not relabel its profile math as electrothermal");
    assert!(error
        .to_string()
        .contains("do not implement an electrothermal formulation"));
}

#[test]
fn nonlinear_synthetic_fixture_requires_a_physical_canonical_basis() {
    let mut model = fixture_model(FixtureId::TransientShock);
    model.sequence.steps = vec![ProcedureStep::structural_nonlinear("unsupported_nonlinear")];
    let error = crate::run_nonlinear_with_options(
        &model,
        ComputeBackend::Cpu,
        baseline_nonlinear_options(),
    )
    .expect_err("unsupported envelopes must not claim J2 continuum equilibrium");
    assert!(matches!(
        error,
        crate::FeaRunError::InvalidModel(_)
            | crate::FeaRunError::Assembly(_)
            | crate::FeaRunError::FormulationAdmission(_)
    ));
}

#[test]
fn nonlinear_beam_moment_is_outside_the_canonical_solid_envelope() {
    let mut model = fixture_model(FixtureId::StructuralBeamCantileverEndMomentReference);
    model.sequence.steps = vec![ProcedureStep::structural_nonlinear("unsupported_nonlinear")];
    let error = crate::run_nonlinear_with_options(
        &model,
        ComputeBackend::Cpu,
        baseline_nonlinear_options(),
    )
    .expect_err("unsupported envelopes must not claim J2 continuum equilibrium");
    assert!(matches!(
        error,
        crate::FeaRunError::InvalidModel(_)
            | crate::FeaRunError::Assembly(_)
            | crate::FeaRunError::FormulationAdmission(_)
    ));
}

#[test]
fn nonlinear_contact_reference_is_explicitly_unsupported() {
    let mut model = fixture_model(FixtureId::NonlinearContactFrictionlessReference);
    model.sequence.steps = vec![ProcedureStep::structural_nonlinear("unsupported_nonlinear")];
    let error = crate::run_nonlinear_with_options(
        &model,
        ComputeBackend::Cpu,
        NonlinearSolveOptions {
            contact_enforcement: Some(FeaContactEnforcement {
                penalty_stiffness_scale: 2.0,
                max_penetration_ratio: 0.01,
            }),
            ..baseline_nonlinear_options()
        },
    )
    .expect_err("unsupported envelopes must not claim J2 continuum equilibrium");
    assert!(matches!(
        error,
        crate::FeaRunError::InvalidModel(_)
            | crate::FeaRunError::Assembly(_)
            | crate::FeaRunError::FormulationAdmission(_)
    ));
}

#[test]
fn thermo_mechanical_nonlinear_cannot_borrow_the_uncoupled_j2_gate() {
    let mut model = fixture_model(FixtureId::NonlinearLoadPathMix);
    configure_thermo_mechanical_excitation(&mut model, 90.0, 1.4e-5);
    model.sequence.steps = vec![ProcedureStep::coupled(
        "thermomechanical_nonlinear",
        runmat_analysis_core::CoupledPhysicsFamily::Thermomechanical,
        runmat_analysis_core::AnalysisProcedure::Nonlinear,
    )];
    let error = crate::run_nonlinear_with_options(
        &model,
        ComputeBackend::Cpu,
        baseline_nonlinear_options(),
    )
    .expect_err("coupled nonlinear equilibrium is unimplemented");
    assert!(matches!(
        error,
        crate::FeaRunError::FormulationAvailability(_)
    ));
}

#[test]
fn nonlinear_proxy_benchmarks_do_not_claim_physical_qualification() {
    for fixture in [
        FixtureId::NonlinearSofteningBenchmark,
        FixtureId::NonlinearLoadPathMix,
    ] {
        let mut model = fixture_model(fixture);
        model.sequence.steps = vec![ProcedureStep::structural_nonlinear("unsupported_nonlinear")];
        let error = crate::run_nonlinear_with_options(
            &model,
            ComputeBackend::Cpu,
            baseline_nonlinear_options(),
        )
        .expect_err("unsupported envelopes must not claim J2 continuum equilibrium");
        assert!(matches!(
            error,
            crate::FeaRunError::InvalidModel(_)
                | crate::FeaRunError::Assembly(_)
                | crate::FeaRunError::FormulationAdmission(_)
        ));
    }
}

#[test]
fn load_sweep_preserves_solver_mesh_dof_count() {
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
            target.region_id = crate::fixtures::fixture_target_selection("tip");
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
    assert!(
        result
            .diagnostics
            .iter()
            .any(|diag| diag.code == "FEA_SOLVER_METHOD"),
        "diagnostics={:#?}",
        result.diagnostics
    );
    assert_eq!(
        field(&result, FEA_FIELD_STRUCTURAL_DISPLACEMENT).element_count(),
        12,
        "four Tet4 nodes must own exactly three displacement components each",
    );

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

// witness: FEA-STRUCTURAL-SOLID-001/load_excitation_diagnostics @
//   linear_structural/beam/numerical_kernel; coverage: partial;
//   boundary: A large load sweep does not fabricate degrees of freedom from the excitation count;
//   diagnostic completeness is not asserted.
#[test]
fn large_load_sweep_does_not_fabricate_dofs_from_excitation_count() {
    let mut model = fixture_model(FixtureId::CantileverLargeLoadSweep);
    for action in &mut model.excitations {
        if let runmat_analysis_core::ExcitationTarget::Region(target) =
            &mut action.excitation_mut().target
        {
            target.region_id = crate::fixtures::fixture_target_selection("tip");
        }
    }
    let result = run_linear_static_with_single_tetrahedron_mesh(&model, ComputeBackend::Cpu)
        .expect("solve should succeed");

    assert_eq!(
        field(&result, FEA_FIELD_STRUCTURAL_DISPLACEMENT).element_count(),
        12,
        "excitation count must not alter canonical mesh topology",
    );
    assert!(
        result
            .diagnostics
            .iter()
            .any(|diag| diag.code == "FEA_SOLVER_METHOD"),
        "diagnostics={:#?}",
        result.diagnostics
    );
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

    assert!(
        axial_x > axial_y,
        "axial reference should be x dominated: axial_x={axial_x} axial_y={axial_y} diagnostics={:#?}",
        axial.diagnostics,
    );
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
            baseline_linear_static_options(),
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
        baseline_linear_static_options(),
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
        baseline_linear_static_options(),
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
    let multi_model = fixture_model(FixtureId::MultiMaterialAssembly);
    let mesh = multi_region_tetrahedron_solver_mesh();
    let mut all_steel_model = multi_model.clone();
    for assignment in &mut all_steel_model.material_assignments {
        assignment.expected_material_id = "mat_steel".to_string();
        assignment.assigned_material_id = "mat_steel".to_string();
        assignment.confidence = runmat_analysis_core::EvidenceConfidence::Verified;
    }
    let baseline = crate::run_linear_static_with_options(
        &all_steel_model,
        ComputeBackend::Cpu,
        LinearStaticSolveOptions {
            solver_mesh: Some(mesh.clone()),
            solver_mesh_artifact_path: Some("all-steel-solver-mesh.cbor".to_string()),
            ..baseline_linear_static_options()
        },
    )
    .expect("all-steel reference solve should succeed");
    let multi_material = crate::run_linear_static_with_options(
        &multi_model,
        ComputeBackend::Cpu,
        LinearStaticSolveOptions {
            solver_mesh: Some(mesh),
            solver_mesh_artifact_path: Some("multi-region-solver-mesh.cbor".to_string()),
            ..baseline_linear_static_options()
        },
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

// witness: FEA-STRUCTURAL-EXECUTION-001/bounded_parallel_accelerator_remote @
//   linear_structural/canonical_mixed_topology/parallel_cpu; coverage: partial;
//   boundary: Admitted parallel solid recovery matches the serial static fields exactly; assembly
//   and solve parallelism and accelerator or remote placement are not covered.
#[test]
fn admitted_parallel_solid_recovery_matches_serial_static_fields_exactly() {
    let model = fixture_model(FixtureId::MultiMaterialAssembly);
    let mesh = multi_region_tetrahedron_solver_mesh();
    let options = LinearStaticSolveOptions {
        solver_mesh: Some(mesh),
        solver_mesh_artifact_path: Some("parallel-recovery-solver-mesh.cbor".to_string()),
        ..baseline_linear_static_options()
    };
    let serial =
        crate::run_linear_static_with_options(&model, ComputeBackend::Cpu, options.clone())
            .expect("serial multi-element static solve");
    let parallel_options = LinearStaticSolveOptions {
        local_execution: crate::planning::resource::LocalExecutionPlan::fixed(
            crate::planning::resource::LocalThreadCount::new(3).unwrap(),
        )
        .unwrap(),
        ..options
    };
    for _ in 0..3 {
        let parallel = crate::run_linear_static_with_options(
            &model,
            ComputeBackend::Cpu,
            parallel_options.clone(),
        )
        .expect("parallel multi-element static recovery");
        assert_eq!(parallel.fields, serial.fields);
        assert_eq!(parallel.observations, serial.observations);
    }
}

#[test]
fn admitted_affine_and_multiplier_static_solves_match_closed_form() {
    use crate::algebra::{
        AssembledRealSystem, CanonicalRealOperators, MatrixProperties, OperatorRole,
        RealSparseMatrix, ScopedSpmvError,
    };
    use crate::constraints::ResolvedConstraintMatrix;
    use crate::planning::resource::{
        execute_scoped_chunks, ByteCount, Count, LocalExecutionPlan, LocalThreadCount,
        ScopedLocalExecutionError, SolverAdmissionResourcePolicy,
    };
    use crate::solve::{
        backend::{cpu_reference::CpuReferenceBackend, kind::LinearAlgebraBackendKind},
        linear::{
            solve_linear_system, solve_linear_system_with_execution,
            solve_linear_system_with_execution_admitted,
        },
        preconditioner::SpdPreconditionerKind,
    };

    let model = fixture_model(FixtureId::StructuralBeamCantileverEndMomentReference);
    let mut summary = crate::assembly::assemble_structural_system(
        &model,
        &static_excitations(&model),
        None,
        crate::assembly::StructuralFormulationRequirement::PhysicalDiscretization,
    )
    .expect("fixture should provide a complete structural assembly summary");
    let dofs = summary.dof_count;
    let stiffness = RealSparseMatrix::diagonal(
        vec![4.0; dofs],
        OperatorRole::Stiffness,
        MatrixProperties::SYMMETRIC_POSITIVE_DEFINITE,
    )
    .unwrap();
    let mass = RealSparseMatrix::diagonal(
        vec![1.0; dofs],
        OperatorRole::Mass,
        MatrixProperties::SYMMETRIC_POSITIVE_DEFINITE,
    )
    .unwrap();
    let damping = RealSparseMatrix::diagonal(
        vec![0.0; dofs],
        OperatorRole::Damping,
        MatrixProperties::SYMMETRIC_POSITIVE_SEMIDEFINITE,
    )
    .unwrap();
    let mut load = vec![0.0; dofs];
    load[0] = 1.0;
    load[1] = 2.0;
    summary.operator = AssembledRealSystem::try_new(
        CanonicalRealOperators::try_new(stiffness, mass, damping).unwrap(),
        vec![false; dofs],
        load,
    )
    .unwrap();
    let matrix = ResolvedConstraintMatrix {
        full_dof_count: dofs,
        row_offsets: vec![0, 2],
        column_indices: vec![0, 1],
        coefficients: vec![1.0, -1.0],
        right_hand_side: vec![0.0],
        row_identities: vec![runmat_execution::Digest::sha256([1u8])],
        constraint_ids: vec!["tie_0_1".into()],
    };
    summary.affine_constraint_reduction = Some(matrix.affine_reduction().unwrap());
    summary.constraint_matrix = Some(matrix);
    let serial = solve_linear_system(
        &summary,
        SpdPreconditionerKind::Jacobi,
        LinearAlgebraBackendKind::CpuReference,
        &CpuReferenceBackend,
        None,
    )
    .unwrap();
    let plan = LocalExecutionPlan::fixed(LocalThreadCount::new(3).unwrap()).unwrap();
    let parallel = solve_linear_system_with_execution(
        &summary,
        SpdPreconditionerKind::Jacobi,
        LinearAlgebraBackendKind::CpuReference,
        &CpuReferenceBackend,
        None,
        plan,
    )
    .unwrap();
    assert!(parallel.converged, "diagnostics={:?}", parallel.diagnostics);
    assert_eq!(parallel, serial);
    assert!((parallel.solution[0] - 3.0 / 8.0).abs() < 1.0e-12);
    assert!((parallel.solution[1] - 3.0 / 8.0).abs() < 1.0e-12);
    for threads in [2, 7, 8] {
        let plan = LocalExecutionPlan::fixed(LocalThreadCount::new(threads).unwrap()).unwrap();
        assert_eq!(
            solve_linear_system_with_execution(
                &summary,
                SpdPreconditionerKind::Jacobi,
                LinearAlgebraBackendKind::CpuReference,
                &CpuReferenceBackend,
                None,
                plan,
            )
            .unwrap(),
            serial,
        );
    }
    assert_eq!(
        solve_linear_system_with_execution(
            &summary,
            SpdPreconditionerKind::Jacobi,
            LinearAlgebraBackendKind::CpuReference,
            &CpuReferenceBackend,
            None,
            plan,
        )
        .unwrap(),
        parallel,
    );
    assert!(parallel
        .diagnostics
        .iter()
        .any(|diagnostic| { diagnostic.code == "FEA_AFFINE_CONSTRAINT_REDUCTION" }));

    let nested = execute_scoped_chunks(LocalExecutionPlan::serial(), 1, |_| {
        Ok::<_, ()>(solve_linear_system_with_execution(
            &summary,
            SpdPreconditionerKind::Jacobi,
            LinearAlgebraBackendKind::CpuReference,
            &CpuReferenceBackend,
            None,
            plan,
        ))
    })
    .unwrap();
    assert!(matches!(
        &nested[0],
        Err(ScopedSpmvError::Execution(
            ScopedLocalExecutionError::NestedExecution
        ))
    ));

    summary.affine_constraint_reduction = None;
    let multiplier_serial = solve_linear_system(
        &summary,
        SpdPreconditionerKind::Jacobi,
        LinearAlgebraBackendKind::CpuReference,
        &CpuReferenceBackend,
        None,
    )
    .unwrap();
    let multiplier_parallel = solve_linear_system_with_execution(
        &summary,
        SpdPreconditionerKind::Jacobi,
        LinearAlgebraBackendKind::CpuReference,
        &CpuReferenceBackend,
        None,
        plan,
    )
    .unwrap();
    assert!(
        multiplier_parallel.converged,
        "diagnostics={:?}",
        multiplier_parallel.diagnostics
    );
    assert_eq!(multiplier_parallel, multiplier_serial);
    let reduction = summary
        .constraint_matrix
        .as_ref()
        .unwrap()
        .affine_reduction()
        .unwrap();
    let (reduced_operator, reduced_rhs) = reduction
        .reduce_sparse_system(
            summary.operator.operators().stiffness(),
            summary.operator.load(),
        )
        .unwrap();
    let prepared = crate::algebra::PreparedRealLinearSystem::try_new(
        reduced_operator,
        reduced_rhs,
        vec![false; reduction.reduced_dof_count],
    )
    .unwrap();
    let (dimension, factor_nonzeros, updates, resident_bytes) =
        prepared.admissibility().unwrap().coercivity().work_counts();
    let admission = SolverAdmissionResourcePolicy {
        dimension: Count(dimension),
        factor_nonzero_upper_bound: Count(factor_nonzeros),
        update_count_upper_bound: Count(updates),
        resident_bytes_upper_bound: ByteCount(resident_bytes),
    };
    let admitted = solve_linear_system_with_execution_admitted(
        &summary,
        SpdPreconditionerKind::Jacobi,
        LinearAlgebraBackendKind::CpuReference,
        &CpuReferenceBackend,
        None,
        plan,
        Some(&admission),
        None,
        None,
    )
    .unwrap();
    assert_eq!(admitted, multiplier_serial);
    let rejected = solve_linear_system_with_execution_admitted(
        &summary,
        SpdPreconditionerKind::Jacobi,
        LinearAlgebraBackendKind::CpuReference,
        &CpuReferenceBackend,
        None,
        plan,
        Some(&SolverAdmissionResourcePolicy {
            factor_nonzero_upper_bound: Count(factor_nonzeros - 1),
            ..admission
        }),
        None,
        None,
    )
    .unwrap();
    assert!(!rejected.converged);
    assert_eq!(rejected.iterations, 0);
    assert_eq!(
        rejected.diagnostics[0].code,
        "FEA_INVALID_AFFINE_SOLVER_ADMISSION"
    );
    assert!((multiplier_parallel.solution[0] - 3.0 / 8.0).abs() < 1.0e-12);
    assert!((multiplier_parallel.solution[1] - 3.0 / 8.0).abs() < 1.0e-12);
    let equilibrium = crate::algebra::spmv(
        summary.operator.operators().stiffness(),
        &multiplier_serial.solution,
    )
    .unwrap()
    .into_iter()
    .zip(summary.physical_load.iter())
    .map(|(internal, applied)| internal - applied)
    .collect::<Vec<_>>();
    assert!(summary.constraint_matrix.is_some());
    let serial_reaction =
        crate::solve::recover_structural_reaction(&summary, &equilibrium).unwrap();
    for threads in [2, 3, 7] {
        let plan = LocalExecutionPlan::fixed(LocalThreadCount::new(threads).unwrap()).unwrap();
        assert_eq!(
            crate::solve::recover_structural_reaction_scoped(&summary, &equilibrium, plan).unwrap(),
            serial_reaction,
        );
        for _ in 0..2 {
            assert_eq!(
                solve_linear_system_with_execution(
                    &summary,
                    SpdPreconditionerKind::Jacobi,
                    LinearAlgebraBackendKind::CpuReference,
                    &CpuReferenceBackend,
                    None,
                    plan,
                )
                .unwrap(),
                multiplier_serial,
            );
        }
    }
    let nested = execute_scoped_chunks(LocalExecutionPlan::serial(), 1, |_| {
        Ok::<_, ()>(solve_linear_system_with_execution(
            &summary,
            SpdPreconditionerKind::Jacobi,
            LinearAlgebraBackendKind::CpuReference,
            &CpuReferenceBackend,
            None,
            plan,
        ))
    })
    .unwrap();
    assert!(matches!(
        &nested[0],
        Err(ScopedSpmvError::Execution(
            ScopedLocalExecutionError::NestedExecution
        ))
    ));
}

fn component_peak(values: &[f64], component: usize) -> f64 {
    values
        .chunks_exact(3)
        .map(|components| components.get(component).copied().unwrap_or(0.0).abs())
        .fold(0.0_f64, f64::max)
}

// witness: FEA-STRUCTURAL-CONSTRAINT-001/distributing_ties_equations @
//   linear_structural/tet10/numerical_kernel; coverage: partial;
//   boundary: A homogeneous two-term multipoint equation reduces the modal operator of one Tet10
//   solid: every extracted mode satisfies the equation and no frequency drops. Distributing
//   couplings, ties and nonhomogeneous equations in modal extraction are not exercised.
#[test]
fn modal_extraction_reduces_a_homogeneous_equation_constraint() {
    use runmat_analysis_core::{
        AnalysisConstraint, CoordinateFrame, Displacement, RegionTarget, StructuralConstraint,
        StructuralConstraintEquation, StructuralConstraintEquationTerm, StructuralConstraintKind,
        StructuralTranslationDof,
    };
    let mut model = fixture_model(FixtureId::CantileverLinearStatic);
    model.sequence.steps = vec![ProcedureStep::structural_modal("modal")];
    let mesh = single_tetrahedron_solver_mesh_with_order_for_model(&model, ElementOrder::Tet10);
    let options = || ModalSolveOptions {
        truncation: ModalTruncationPolicy::requested(3),
        solver_mesh: Some(mesh.clone()),
        ..baseline_modal_options()
    };
    let free = crate::run_modal_with_options(&model, ComputeBackend::Cpu, options())
        .expect("modal solve without the equation");
    // Mode shapes follow canonical mesh node order; tie the axial motion of the two free nodes
    // that move most differently in the first mode.
    let first = free.mode_shapes[0].as_host_f64().unwrap();
    let nodes = free.mode_shapes[0].shape[0];
    assert_eq!(nodes, mesh.topology.nodes.len());
    let ux = |node: usize| first[node * 3];
    let left = (0..nodes).max_by(|a, b| ux(*a).total_cmp(&ux(*b))).unwrap();
    let right = (0..nodes).min_by(|a, b| ux(*a).total_cmp(&ux(*b))).unwrap();
    assert!(ux(left) - ux(right) > 0.0);
    let term = |node: usize, coefficient: f64| StructuralConstraintEquationTerm {
        target: RegionTarget::point(format!("node:{}", mesh.topology.nodes[node].node_id)),
        dof: StructuralTranslationDof::Ux,
        coefficient,
    };
    model
        .constraints
        .push(AnalysisConstraint::Structural(StructuralConstraint {
            constraint_id: "axial_tie".into(),
            frame: CoordinateFrame::Global,
            kind: StructuralConstraintKind::Equation(StructuralConstraintEquation {
                terms: vec![term(left, 1.0), term(right, -1.0)],
                right_hand_side: Displacement::new(0.0),
            }),
        }));
    let tied = crate::run_modal_with_options(&model, ComputeBackend::Cpu, options())
        .expect("modal extraction admits a homogeneous equation constraint");
    // Every mode satisfies the constraint, and a constraint can only raise each frequency.
    for mode in &tied.mode_shapes {
        let shape = mode.as_host_f64().unwrap();
        let scale = shape.iter().fold(0.0_f64, |m, v| m.max(v.abs()));
        assert!((shape[left * 3] - shape[right * 3]).abs() <= 1.0e-9 * scale);
    }
    for (constrained, unconstrained) in tied.frequencies_hz().iter().zip(free.frequencies_hz()) {
        assert!(*constrained >= unconstrained * (1.0 - 1.0e-9));
    }
    assert!(tied.frequencies_hz()[0] > free.frequencies_hz()[0] * (1.0 + 1.0e-6));
}

fn assert_one_fallback_warning(run: &crate::contracts::FeaRunResult, family: &str) {
    let warnings = run
        .diagnostics
        .iter()
        .filter(|diagnostic| {
            diagnostic.code
                == crate::contracts::accelerator_admission::ACCELERATOR_ADMISSION_FALLBACK_CODE
        })
        .collect::<Vec<_>>();
    assert_eq!(warnings.len(), 1, "{:?}", run.diagnostics);
    assert!(
        warnings[0].message.contains(family),
        "{}",
        warnings[0].message
    );
    assert_eq!(run.requested_backend, ComputeBackend::Gpu);
    assert_eq!(run.execution_backend, crate::ExecutionBackend::Cpu);
}

fn required<R>(body: impl FnOnce() -> R) -> R {
    crate::contracts::accelerator_admission::with_accelerator_admission_policy(
        crate::contracts::accelerator_admission::AcceleratorAdmissionPolicy::Required,
        body,
    )
}

#[test]
fn modal_gpu_request_falls_back_to_cpu_reference_with_parity_unless_required() {
    let mut model = fixture_model(FixtureId::CantileverLinearStatic);
    model.sequence.steps = vec![ProcedureStep::structural_modal("modal_1".to_string())];
    let cpu = crate::run_modal_with_options(&model, ComputeBackend::Cpu, baseline_modal_options())
        .unwrap();
    let gpu = crate::run_modal_with_options(&model, ComputeBackend::Gpu, baseline_modal_options())
        .unwrap();
    assert_one_fallback_warning(&gpu.run, "structural modal");
    assert_eq!(gpu.frequencies_hz(), cpu.frequencies_hz());
    assert_eq!(gpu.mode_shapes, cpu.mode_shapes);
    assert!(matches!(
        required(|| crate::run_modal_with_options(
            &model,
            ComputeBackend::Gpu,
            baseline_modal_options()
        )),
        Err(crate::contracts::FeaRunError::AcceleratorRequired(_))
    ));
}

#[test]
fn transient_gpu_request_without_a_device_falls_back_with_parity_unless_required() {
    if crate::contracts::accelerator_admission::device_unavailable_reason().is_none() {
        return;
    }
    let mut model = fixture_model(FixtureId::CantileverLinearStatic);
    model.sequence.steps = vec![ProcedureStep::structural_transient(
        "transient_1".to_string(),
    )];
    let cpu = crate::run_transient_with_options(
        &model,
        ComputeBackend::Cpu,
        baseline_transient_options(),
    )
    .unwrap();
    let gpu = crate::run_transient_with_options(
        &model,
        ComputeBackend::Gpu,
        baseline_transient_options(),
    )
    .unwrap();
    assert_one_fallback_warning(&gpu.run, "structural transient");
    assert_eq!(gpu.time_points_s, cpu.time_points_s);
    assert_eq!(gpu.displacement_snapshots, cpu.displacement_snapshots);
    assert!(matches!(
        required(|| crate::run_transient_with_options(
            &model,
            ComputeBackend::Gpu,
            baseline_transient_options()
        )),
        Err(crate::contracts::FeaRunError::AcceleratorRequired(_))
    ));
}
