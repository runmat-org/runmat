use runmat_types::physical::Quantity;
use std::sync::atomic::{AtomicBool, AtomicU64, Ordering};
use std::sync::{Arc, Mutex};
use std::{
    fs,
    path::{Path, PathBuf},
};

use chrono::Utc;
use runmat_accelerate_api::{AccelDownloadFuture, AccelProvider, GpuTensorHandle, HostTensorView};
use runmat_analysis_core::{
    AnalysisExcitation, AnalysisField, AnalysisFieldValues, AnalysisModel, AnalysisModelId,
    AnalysisProcedure, AnalysisSequence, BeamElementModel, BeamSectionModel, CoordinateFrame,
    CoupledPhysicsFamily, Direction, DynamicViscosity, ElectricPotential, ElectromagneticDomain,
    EvidenceConfidence, ExcitationAction, ExcitationCase, ExcitationCaseActivation,
    ExcitationCaseId, ExcitationDistribution, ExcitationId, ExcitationKinematics, ExcitationTarget,
    Frequency, MassDensity, MaterialAssignment, MaterialElasticityModel, MaterialElectricalModel,
    MaterialMechanicalModel, MaterialModel, MaterialThermalModel, PhysicalSignal, PhysicsFamily,
    ProcedureStep, ProjectionRequirement, RegionTarget, Signal, SignalApproximationEvidence,
    SignalExtrapolation, SignalId, SignalInterpolation, SignalKnot, SignalOrdinate,
    SignalPrecision, SignalProvenance, SignalRepresentation, SignalShape, SignalValue,
    StructuralElement, StructuralElementKind, StructuralModel, StructuralNode, TabulatedSignal,
    TabulatedSignalDomain, Temperature, TurbulenceIntensity, Velocity,
};
use runmat_analysis_fea::solve::backend::kind::LinearAlgebraBackendKind;
use runmat_analysis_fea::{
    fea_acoustic_frequency_response_field_id, fea_cht_interface_heat_flux_field_id,
    fea_fsi_interface_traction_field_id, fea_modal_mode_shape_field_id,
    fea_nonlinear_contact_pressure_field_id, fea_thermal_energy_storage_rate_field_id,
    fea_thermal_heat_flux_field_id, fea_thermal_temperature_gradient_field_id,
    fea_thermo_mechanical_coupling_residual_field_id, fea_thermo_mechanical_displacement_field_id,
    fea_thermo_mechanical_temperature_field_id, fea_thermo_mechanical_thermal_strain_field_id,
    fea_thermo_mechanical_thermal_stress_field_id, fea_thermo_mechanical_von_mises_field_id,
    fea_transient_acceleration_field_id, fea_transient_angular_acceleration_field_id,
    fea_transient_angular_velocity_field_id, fea_transient_damping_dissipation_field_id,
    fea_transient_energy_balance_residual_field_id, fea_transient_external_work_field_id,
    fea_transient_kinetic_energy_field_id, fea_transient_reaction_force_field_id,
    fea_transient_relative_energy_balance_error_field_id, fea_transient_residual_norm_field_id,
    fea_transient_rotation_field_id, fea_transient_strain_energy_density_field_id,
    fea_transient_strain_energy_field_id, fea_transient_strain_field_id,
    fea_transient_stress_field_id, fea_transient_support_work_field_id,
    fea_transient_velocity_field_id, ComputeBackend, FeaProgressPhase, FeaProgressStatus,
    SolverMethod, SolverPreconditioner, FEA_FIELD_ACOUSTIC_PARTICLE_VELOCITY,
    FEA_FIELD_ACOUSTIC_PRESSURE_IMAG, FEA_FIELD_ACOUSTIC_PRESSURE_MAGNITUDE,
    FEA_FIELD_ACOUSTIC_PRESSURE_REAL, FEA_FIELD_CFD_WALL_SHEAR_STRESS,
    FEA_FIELD_ELECTRO_THERMAL_JOULE_HEAT, FEA_FIELD_MODAL_EIGENVALUE, FEA_FIELD_MODAL_FREQUENCY_HZ,
    FEA_FIELD_MODAL_MODAL_MASS, FEA_FIELD_MODAL_MODAL_STIFFNESS, FEA_FIELD_MODAL_M_ORTHOGONALITY,
    FEA_FIELD_MODAL_PARTICIPATION_FACTOR, FEA_FIELD_MODAL_RELATIVE_FREQUENCY_SEPARATION,
    FEA_FIELD_MODAL_RESIDUAL_NORM, FEA_FIELD_STRUCTURAL_BEAM_AXIAL_FORCE,
    FEA_FIELD_STRUCTURAL_BEAM_BENDING_MOMENT, FEA_FIELD_STRUCTURAL_BEAM_BENDING_STRESS,
    FEA_FIELD_STRUCTURAL_BEAM_SHEAR_FORCE, FEA_FIELD_STRUCTURAL_BEAM_TORSION_MOMENT,
    FEA_FIELD_STRUCTURAL_BEAM_TORSION_STRESS, FEA_FIELD_STRUCTURAL_DISPLACEMENT,
    FEA_FIELD_STRUCTURAL_EQUATION_SCALE, FEA_FIELD_STRUCTURAL_NODAL_VON_MISES,
    FEA_FIELD_STRUCTURAL_REACTION_FORCE, FEA_FIELD_STRUCTURAL_REACTION_MOMENT,
    FEA_FIELD_STRUCTURAL_RESIDUAL_NORM, FEA_FIELD_STRUCTURAL_ROTATION,
    FEA_FIELD_STRUCTURAL_SHELL_BENDING_MOMENT, FEA_FIELD_STRUCTURAL_SHELL_MEMBRANE_FORCE,
    FEA_FIELD_STRUCTURAL_SHELL_TRANSVERSE_SHEAR, FEA_FIELD_STRUCTURAL_SHELL_VON_MISES,
    FEA_FIELD_STRUCTURAL_STRAIN, FEA_FIELD_STRUCTURAL_STRAIN_ENERGY_DENSITY,
    FEA_FIELD_STRUCTURAL_STRESS, FEA_FIELD_STRUCTURAL_TOTAL_STRAIN_ENERGY,
    FEA_FIELD_STRUCTURAL_VON_MISES,
};
use runmat_geometry_core::{
    EntityIdRange, EntityKind, GeometryAsset, GeometrySource, MeshDescriptor, MeshKind, Region,
    RegionEntityMapping, SourceGeometry, SourceGeometryKind, SurfaceMesh, TessellationProfile,
};
use runmat_meshing_core::{
    CanonicalMeshingContract, ElementOrder, PersistentEntityId, PersistentEntityKind, StableDigest,
};

use super::*;

mod admission_roots;
mod admission_validation;
mod nonlinear_physical;
mod persisted_record_shape;
mod static_admission;
mod structural_study_intent;
mod structural_study_storage;
mod topology_identity;
mod transient_assessments;

fn test_linear_workload_identity(
    workload: &runmat_analysis_fea::execution::StructuralLinearWorkload,
    mesh: &runmat_meshing_core::SolverMeshArtifact,
) -> runmat_analysis_fea::execution::AnalysisWorkloadIdentity {
    use runmat_analysis_fea::execution::{
        AnalysisWorkKind, AnalysisWorkloadIdentity, ConstraintPlanIdentity, DiscretizationIdentity,
        ExecutionPlanIdentity, NumericalPlanIdentity, OperatorIdentity, ProblemIdentity,
        SequenceIdentity, StepIdentity, StudyIdentity,
    };
    let digest = runmat_execution::Digest::sha256(b"runtime-linear-completion-test");
    let study = StudyIdentity::from_authored_digest(digest, None).unwrap();
    let sequence = SequenceIdentity::from_solver_contract(&workload.model.sequence).unwrap();
    let step =
        StepIdentity::from_solver_contract(sequence, workload.procedure_step().unwrap()).unwrap();
    let constraints = ConstraintPlanIdentity::new(step, digest, digest).unwrap();
    let problem = ProblemIdentity::new(step, digest, None, constraints, Vec::new()).unwrap();
    let mesh_digest = runmat_execution::Digest::from_bytes(*mesh.canonical_digest.bytes());
    let discretization = DiscretizationIdentity::new(problem, mesh_digest, digest).unwrap();
    let operator = OperatorIdentity::new(discretization, digest).unwrap();
    let numerical = NumericalPlanIdentity::new(operator, digest).unwrap();
    let execution = ExecutionPlanIdentity::new(numerical, digest).unwrap();
    AnalysisWorkloadIdentity::new(
        study,
        step,
        execution,
        AnalysisWorkKind::Solve,
        digest,
        vec![
            runmat_execution::Digest::sha256(workload.canonical_bytes().unwrap()),
            mesh_digest,
        ],
    )
    .unwrap()
}

fn test_structural_sequence_result(
    spec: &AnalysisStudySpec,
    workload: &runmat_analysis_fea::execution::StructuralLinearWorkload,
    mesh: &runmat_meshing_core::SolverMeshArtifact,
    result: runmat_analysis_fea::execution::StructuralLinearWorkloadResult,
) -> StructuralStudyNumericalResult {
    let _study = runmat_analysis_fea::execution::StudyIdentity::from_authored_digest(
        analysis_study_content_digest(spec),
        None,
    )
    .unwrap();
    let sequence = runmat_analysis_fea::execution::SequenceIdentity::from_solver_contract(
        &workload.model.sequence,
    )
    .unwrap();
    let step = workload.procedure_step().unwrap();
    let producer =
        runmat_analysis_fea::execution::StepIdentity::from_solver_contract(sequence, step).unwrap();
    let source_workload = test_linear_workload_identity(workload, mesh);
    let numerical_receipt =
        runmat_analysis_fea::execution::StructuralLinearNumericalReceiptArtifact::new(
            source_workload,
            &result,
        )
        .unwrap();
    let solver_result_identity = numerical_receipt.source_result();
    let fields = result
        .run
        .fields
        .iter()
        .filter(|field| !field.is_empty() && field.as_host_f64().is_some())
        .cloned()
        .collect::<Vec<_>>();
    let states = step
        .state_outputs
        .iter()
        .map(|declaration| {
            CanonicalSemanticState::new(CanonicalSemanticStateInput {
                producer,
                declaration: declaration.clone(),
                topology_digest: runmat_execution::Digest::from_bytes(
                    *mesh.canonical_digest.bytes(),
                ),
                source_result: solver_result_identity,
                payload: runmat_analysis_fea::execution::CanonicalSemanticStatePayload::Fields {
                    fields: fields.clone(),
                },
            })
            .unwrap()
        })
        .collect();
    StructuralStudyNumericalResult::LinearSequence(StructuralStudySequenceNumericalResult {
        frozen_project_digest: None,
        effective_memory_ceiling_bytes: None,
        steps: vec![StructuralStudyStepNumericalResult {
            step_id: workload.step_id.clone(),
            solver_result_identity,
            result,
            numerical_receipt,
            static_solve_path: None,
        }],
        states,
    })
}

fn fixed_constraint(
    id: impl Into<String>,
    region: impl Into<String>,
) -> runmat_analysis_core::AnalysisConstraint {
    runmat_analysis_core::AnalysisConstraint::Structural(
        runmat_analysis_core::StructuralConstraint {
            constraint_id: id.into(),
            frame: CoordinateFrame::Global,
            kind: runmat_analysis_core::StructuralConstraintKind::Fixed {
                target: runmat_analysis_core::RegionTarget::authored(
                    runmat_meshing_core::fixtures::fixture_boundary_selection(&region.into()),
                ),
            },
        },
    )
}

fn sample_cfd_constraints(
    inlet_velocity_m_per_s: [f64; 3],
) -> Vec<runmat_analysis_core::AnalysisConstraint> {
    let fluid = |id: &str, kind| {
        runmat_analysis_core::AnalysisConstraint::Fluid(runmat_analysis_core::FluidConstraint {
            constraint_id: id.to_string(),
            kind,
        })
    };
    vec![
        fluid(
            "bc_cfd_inlet",
            runmat_analysis_core::FluidConstraintKind::PrescribedVelocity {
                target: runmat_analysis_core::RegionTarget::boundary(
                    runmat_meshing_core::fixtures::fixture_face("fluid_inlet"),
                ),
                value: runmat_analysis_core::PrescribedValue::Constant(
                    inlet_velocity_m_per_s.map(runmat_analysis_core::Velocity::new),
                ),
            },
        ),
        fluid(
            "bc_cfd_outlet",
            runmat_analysis_core::FluidConstraintKind::PressureReference {
                target: runmat_analysis_core::RegionTarget::boundary(
                    runmat_meshing_core::fixtures::fixture_face("fluid_outlet"),
                ),
                value: runmat_analysis_core::PrescribedValue::Constant(
                    runmat_analysis_core::Pressure::new(0.0),
                ),
            },
        ),
        fluid(
            "bc_cfd_wall_upper",
            runmat_analysis_core::FluidConstraintKind::NoSlipWall {
                target: runmat_analysis_core::RegionTarget::boundary(
                    runmat_meshing_core::fixtures::fixture_face("fluid_wall_upper"),
                ),
            },
        ),
        fluid(
            "bc_cfd_wall_lower",
            runmat_analysis_core::FluidConstraintKind::NoSlipWall {
                target: runmat_analysis_core::RegionTarget::boundary(
                    runmat_meshing_core::fixtures::fixture_face("fluid_wall_lower"),
                ),
            },
        ),
    ]
}

/// Explicit test host: numerical tests opt into admission while holding the same
/// lock used for their process-wide artifact store and runtime configuration.
pub(super) struct AnalysisTestHost {
    // Restore admission before releasing the shared host-state lock (field drop order).
    _admission: fea_run_admission::FeaRunAdmissionGuard,
    _storage: storage::test_support::IsolatedArtifactStore,
}

pub(super) fn analysis_test_guard() -> AnalysisTestHost {
    let storage = storage::test_support::isolated_artifact_store();
    let admission = fea_run_admission::replace_fea_run_admission(Some(
        fea_run_admission::entrypoints::test_admission(),
    ))
    .expect("install explicit numerical-test FEA host");
    AnalysisTestHost {
        _admission: admission,
        _storage: storage,
    }
}

fn test_electrical_material() -> MaterialElectricalModel {
    MaterialElectricalModel {
        reference_temperature_k: runmat_types::physical::Temperature::new(293.15),
        conductivity_s_per_m: runmat_types::physical::ElectricalConductivity::new(1.0),
        resistive_heating_coefficient: runmat_types::physical::InverseTemperature::new(0.0),
        relative_permittivity: runmat_types::physical::RelativePermittivity::new(1.0),
        relative_permeability: runmat_types::physical::RelativePermeability::new(1.0),
        conductivity_frequency_response: Vec::new(),
    }
}

pub(super) fn sample_model() -> AnalysisModel {
    let mut model = AnalysisModel {
        model_id: AnalysisModelId("beam_model".to_string()),
        geometry_id: "geo:beam".to_string(),
        geometry_revision: 1,
        units: runmat_types::physical::CoordinateUnits::canonical(),
        frame: CoordinateFrame::Global,
        materials: vec![MaterialModel {
            fluid: None,
            material_id: "mat_steel".to_string(),
            name: "Steel".to_string(),
            provenance: runmat_analysis_core::MaterialProvenance::Authored,
            mass_density_kg_per_m3: Some(Quantity::new(7850.0)),
            elasticity: Some(MaterialElasticityModel::IsotropicLinear(
                MaterialMechanicalModel {
                    youngs_modulus_pa: Quantity::new(200e9),
                    poisson_ratio: Quantity::new(0.3),
                },
            )),
            damping: None,
            thermal: Some(MaterialThermalModel {
                reference_temperature_k: Quantity::new(293.15),
                modulus_temp_coeff_per_k: Quantity::new(-2.5e-4),
                conductivity_w_per_mk: Quantity::new(45.0),
                specific_heat_j_per_kgk: Quantity::new(500.0),
                expansion_coefficient_per_k: Quantity::new(1.2e-5),
            }),
            acoustic: None,
            electrical: None,
            plastic: None,
        }],
        material_assignments: vec![MaterialAssignment {
            target: runmat_analysis_core::PropertyRegionTarget::beam(
                runmat_meshing_core::fixtures::fixture_edge("beam_span"),
            ),
            expected_material_id: "mat_steel".to_string(),
            assigned_material_id: "mat_steel".to_string(),
            confidence: EvidenceConfidence::Verified,
        }],
        structural: Some(StructuralModel {
            nodes: vec![
                StructuralNode {
                    node_id: 1,
                    coordinates_m: Quantity::new([0.0, 0.0, 0.0]),
                    solid_attachment: None,
                },
                StructuralNode {
                    node_id: 2,
                    coordinates_m: Quantity::new([1.0, 0.0, 0.0]),
                    solid_attachment: None,
                },
            ],
            elements: vec![StructuralElement {
                element_id: "beam_1".to_string(),
                region_id: runmat_meshing_core::fixtures::fixture_edge("beam_span"),
                kind: StructuralElementKind::Beam(BeamElementModel {
                    node_ids: [1, 2],
                    section_id: "rect".to_string(),
                    reference_axis: [0.0, 1.0, 0.0],
                }),
            }],
            beam_sections: vec![BeamSectionModel {
                section_id: "rect".to_string(),
                area_m2: Quantity::new(2.0e-4),
                iy_m4: Quantity::new(1.6e-9),
                iz_m4: Quantity::new(6.4e-9),
                torsion_j_m4: Quantity::new(2.4e-9),
                outer_fiber_y_m: Quantity::new(0.01),
                outer_fiber_z_m: Quantity::new(0.005),
                torsion_outer_radius_m: Quantity::new(0.011_180_339_887_498_949),
            }],
            shell_sections: Vec::new(),
        }),
        thermal: None,
        thermo_mechanical: None,
        electro_thermal: None,
        electromagnetic: None,
        cfd: None,
        interfaces: Vec::new(),
        constraints: vec![fixed_constraint("bc_root", "node:1")],
        boundaries: Vec::new(),
        connectors: Vec::new(),
        signals: Vec::new(),
        excitations: Vec::new(),
        excitation_cases: Vec::new(),
        initial_conditions: Vec::new(),
        response_spectra: Vec::new(),
        random_excitations: Vec::new(),
        sequence: AnalysisSequence::from_steps(
            "sample_sequence",
            vec![ProcedureStep::structural_static("step_static")],
        ),
    };
    append_constant_excitation(
        &mut model,
        "load_tip",
        "node:2",
        ConstantExcitation::Force([0.0, -1000.0, 0.0]),
    );
    let runmat_analysis_core::ExcitationTarget::Region(target) =
        &mut model.excitations[0].excitation_mut().target
    else {
        panic!("expected region excitation target");
    };
    *target = runmat_analysis_core::RegionTarget::point("node:2");
    model
}

pub(super) fn sample_solid_model() -> AnalysisModel {
    let mut model = sample_model();
    model.model_id = AnalysisModelId("solid_model".to_string());
    model.geometry_id = "geo:closed_cube".to_string();
    model.geometry_revision = 1;
    model.structural = None;
    model.material_assignments[0].target = runmat_analysis_core::PropertyRegionTarget::solid(
        runmat_meshing_core::fixtures::fixture_region("region"),
    );
    model.constraints = vec![fixed_constraint("bc_root", "root")];
    model.signals.clear();
    model.excitations.clear();
    model.excitation_cases.clear();
    append_constant_excitation(
        &mut model,
        "load_tip",
        runmat_meshing_core::fixtures::fixture_face("tip"),
        ConstantExcitation::Force([0.0, -1000.0, 0.0]),
    );
    model
}

fn sample_model_with_material_assignment_mismatch() -> AnalysisModel {
    let mut model = sample_model();
    model.materials.push(MaterialModel {
        fluid: None,
        material_id: "mat_polymer".to_string(),
        name: "Polymer".to_string(),
        provenance: runmat_analysis_core::MaterialProvenance::Authored,
        mass_density_kg_per_m3: Some(runmat_types::physical::MassDensity::new(1150.0)),
        elasticity: Some(MaterialElasticityModel::IsotropicLinear(
            MaterialMechanicalModel {
                youngs_modulus_pa: runmat_types::physical::YoungsModulus::new(3.2e9),
                poisson_ratio: runmat_types::physical::PoissonRatio::new(0.37),
            },
        )),
        damping: None,
        thermal: Some(MaterialThermalModel {
            reference_temperature_k: runmat_types::physical::Temperature::new(293.15),
            modulus_temp_coeff_per_k: runmat_types::physical::InverseTemperature::new(-7.0e-4),
            conductivity_w_per_mk: runmat_types::physical::ThermalConductivity::new(45.0),
            specific_heat_j_per_kgk: runmat_types::physical::SpecificHeat::new(500.0),
            expansion_coefficient_per_k: runmat_types::physical::ThermalExpansion::new(1.2e-5),
        }),
        acoustic: None,
        electrical: None,
        plastic: None,
    });
    model.material_assignments = vec![MaterialAssignment {
        target: runmat_analysis_core::PropertyRegionTarget::beam(
            runmat_meshing_core::fixtures::fixture_edge("beam_span"),
        ),
        expected_material_id: "mat_steel".to_string(),
        assigned_material_id: "mat_polymer".to_string(),
        confidence: EvidenceConfidence::Verified,
    }];
    model
}

fn sample_cfd_domain() -> runmat_analysis_core::CfdDomain {
    runmat_analysis_core::CfdDomain {
        density: MassDensity::new(1.225),
        dynamic_viscosity: DynamicViscosity::new(1.81e-5),
        turbulence_intensity: TurbulenceIntensity::new(0.06),
        rheology: None,
        porous_zones: Vec::new(),
    }
}

fn append_scalar_history_excitation<Q>(
    model: &mut AnalysisModel,
    id: &str,
    axis: Vec<f64>,
    values: Vec<Q>,
    target: ExcitationTarget,
    direction: Option<Direction>,
    signal: impl FnOnce(Signal<Q>) -> PhysicalSignal,
    action: impl FnOnce(AnalysisExcitation) -> ExcitationAction,
) {
    let signal_id = SignalId(format!("{id}.signal"));
    let case_id = ExcitationCaseId(format!("{id}.case"));
    let sample_count = values.len();
    model.signals.push(signal(Signal {
        signal_id: signal_id.clone(),
        shape: SignalShape::Scalar,
        frame: None,
        representation: SignalRepresentation::Tabulated(TabulatedSignal {
            domain: TabulatedSignalDomain::TimeSeconds,
            knots: axis
                .into_iter()
                .zip(values)
                .map(|(abscissa, value)| {
                    SignalKnot::continuous(
                        abscissa,
                        SignalOrdinate::Real(SignalValue::Scalar(value)),
                    )
                })
                .collect(),
            interpolation: SignalInterpolation::Linear,
            extrapolation: SignalExtrapolation::HoldEndpoint,
        }),
        precision: SignalPrecision::Real64,
        evidence: SignalApproximationEvidence {
            max_absolute_error: None,
            max_relative_error: None,
            derivative_max_absolute_error: None,
            sample_count,
        },
        provenance: SignalProvenance::Authored,
    }));
    let excitation_id = ExcitationId(id.to_string());
    model.excitations.push(action(AnalysisExcitation {
        excitation_id: excitation_id.clone(),
        target,
        frame: CoordinateFrame::Global,
        direction,
        distribution: ExcitationDistribution::TopologyField,
        kinematics: ExcitationKinematics::FixedFrame,
        signal_id,
        active_step_ids: Vec::new(),
        combination_group: None,
        projection: ProjectionRequirement {
            preserve_resultant: false,
            preserve_moment: false,
            relative_tolerance: 1.0e-10,
        },
    }));
    model.excitation_cases.push(ExcitationCase {
        case_id,
        action_ids: vec![excitation_id],
        activation: ExcitationCaseActivation::Always,
    });
}

fn append_scalar_constant_excitation<Q>(
    model: &mut AnalysisModel,
    id: &str,
    value: Q,
    target: ExcitationTarget,
    direction: Option<Direction>,
    signal: impl FnOnce(Signal<Q>) -> PhysicalSignal,
    action: impl FnOnce(AnalysisExcitation) -> ExcitationAction,
) {
    let signal_id = SignalId(format!("{id}.signal"));
    let case_id = ExcitationCaseId(format!("{id}.case"));
    model.signals.push(signal(Signal {
        signal_id: signal_id.clone(),
        shape: SignalShape::Scalar,
        frame: None,
        representation: SignalRepresentation::Constant(SignalValue::Scalar(value)),
        precision: SignalPrecision::Real64,
        evidence: SignalApproximationEvidence {
            max_absolute_error: None,
            max_relative_error: None,
            derivative_max_absolute_error: None,
            sample_count: 1,
        },
        provenance: SignalProvenance::Authored,
    }));
    let excitation_id = ExcitationId(id.to_string());
    model.excitations.push(action(AnalysisExcitation {
        excitation_id: excitation_id.clone(),
        target,
        frame: CoordinateFrame::Global,
        direction,
        distribution: ExcitationDistribution::TopologyField,
        kinematics: ExcitationKinematics::FixedFrame,
        signal_id,
        active_step_ids: Vec::new(),
        combination_group: None,
        projection: ProjectionRequirement {
            preserve_resultant: false,
            preserve_moment: false,
            relative_tolerance: 1.0e-10,
        },
    }));
    model.excitation_cases.push(ExcitationCase {
        case_id,
        action_ids: vec![excitation_id],
        activation: ExcitationCaseActivation::Always,
    });
}

fn replace_with_moment_excitation(model: &mut AnalysisModel, excitation_id: &str) {
    use runmat_analysis_core::{
        AnalysisExcitation, ExcitationCase, ExcitationCaseActivation, ExcitationCaseId,
        ExcitationDistribution, ExcitationId, ExcitationKinematics, ExcitationTarget, Moment,
        ProjectionRequirement, RegionTarget, Signal, SignalApproximationEvidence, SignalId,
        SignalPrecision, SignalProvenance, SignalShape,
    };

    let signal_id = SignalId(format!("{excitation_id}.signal"));
    let case_id = ExcitationCaseId(format!("{excitation_id}.case"));
    model.signals = vec![PhysicalSignal::Moment(Signal {
        signal_id: signal_id.clone(),
        shape: SignalShape::Vector3,
        frame: Some(CoordinateFrame::Global),
        representation: SignalRepresentation::Constant(SignalValue::Vector3([
            Moment::new(1.0),
            Moment::new(2.0),
            Moment::new(3.0),
        ])),
        precision: SignalPrecision::Real64,
        evidence: SignalApproximationEvidence {
            max_absolute_error: None,
            max_relative_error: None,
            derivative_max_absolute_error: None,
            sample_count: 1,
        },
        provenance: SignalProvenance::Authored,
    })];
    model.excitations = vec![ExcitationAction::Moment(AnalysisExcitation {
        excitation_id: ExcitationId(excitation_id.to_string()),
        target: ExcitationTarget::Region(RegionTarget::boundary(
            runmat_meshing_core::fixtures::fixture_face("tip"),
        )),
        frame: CoordinateFrame::Global,
        direction: None,
        distribution: ExcitationDistribution::PreserveResultant,
        kinematics: ExcitationKinematics::FixedFrame,
        signal_id,
        active_step_ids: Vec::new(),
        combination_group: None,
        projection: ProjectionRequirement {
            preserve_resultant: true,
            preserve_moment: true,
            relative_tolerance: 1.0e-10,
        },
    })];
    model.excitation_cases = vec![ExcitationCase {
        case_id,
        action_ids: vec![ExcitationId(excitation_id.to_string())],
        activation: ExcitationCaseActivation::Always,
    }];
}

fn sample_cht_model() -> AnalysisModel {
    let mut model = sample_solid_model();
    model.thermal = Some(runmat_analysis_core::ThermalDomain {
        reference_temperature: Temperature::new(293.15),
    });
    model.sequence = AnalysisSequence::from_steps(
        "cht_sequence",
        vec![ProcedureStep::coupled(
            "cht",
            CoupledPhysicsFamily::ConjugateHeatTransfer,
            AnalysisProcedure::TransientResponse,
        )],
    );
    model.cfd = Some(sample_cfd_domain());
    set_model_thermo_coupling(
        &mut model,
        TestThermoMechanicalCoupling {
            reference_temperature_k: runmat_types::physical::Temperature::new(293.15),
            applied_temperature_delta_k: runmat_types::physical::TemperatureDifference::new(60.0),
            thermal_expansion_coefficient: runmat_types::physical::ThermalExpansion::new(1.2e-5),
            region_temperature_deltas: Vec::new(),
        },
    );
    model
}

#[derive(Clone)]
struct TestThermoRegionTemperatureDelta {
    region_id: String,
    temperature_delta_k: f64,
}

#[derive(Clone)]
struct TestThermoMechanicalCoupling {
    reference_temperature_k: runmat_types::physical::Temperature,
    applied_temperature_delta_k: runmat_types::physical::TemperatureDifference,
    thermal_expansion_coefficient: runmat_types::physical::ThermalExpansion,
    region_temperature_deltas: Vec<TestThermoRegionTemperatureDelta>,
}

#[derive(Clone, Copy)]
struct TestElectroThermalCoupling {
    reference_temperature_k: runmat_types::physical::Temperature,
    applied_voltage_v: f64,
    base_electrical_conductivity_s_per_m: f64,
    resistive_heating_coefficient: runmat_types::physical::InverseTemperature,
}

fn sample_fsi_model() -> AnalysisModel {
    let mut model = sample_model();
    model.sequence = AnalysisSequence::from_steps(
        "fsi_sequence",
        vec![ProcedureStep::coupled(
            "fsi",
            CoupledPhysicsFamily::FluidStructureInteraction,
            AnalysisProcedure::TransientResponse,
        )],
    );
    append_scalar_history_excitation(
        &mut model,
        "fsi_inlet_velocity",
        vec![0.0, 1.0],
        vec![Velocity::new(2.4), Velocity::new(4.0)],
        ExcitationTarget::Region(RegionTarget::boundary(
            runmat_meshing_core::fixtures::fixture_face("inlet"),
        )),
        Some(Direction {
            vector: [1.0, 0.0, 0.0],
        }),
        PhysicalSignal::Velocity,
        ExcitationAction::InletVelocity,
    );
    model.cfd = Some(sample_cfd_domain());
    model
}

#[derive(Clone, Copy)]
enum TestModelKind {
    LinearStaticStructural,
    ModalStructural,
    AcousticHarmonic,
    ElectromagneticHarmonic,
}

struct TestModelSpec {
    model_id: String,
    kind: TestModelKind,
}

fn test_model_for_geometry(
    geometry: &GeometryAsset,
    spec: TestModelSpec,
    context: OperationContext,
) -> Result<OperationEnvelope<AnalysisModel>, OperationErrorEnvelope> {
    let mut model = sample_solid_model();
    model.model_id = AnalysisModelId(spec.model_id);
    model.geometry_id = geometry.geometry_id.clone();
    model.geometry_revision = geometry.revision;
    model.units = runmat_types::physical::CoordinateUnits::canonical();
    model.material_assignments = vec![MaterialAssignment {
        target: match spec.kind {
            TestModelKind::AcousticHarmonic | TestModelKind::ElectromagneticHarmonic => {
                runmat_analysis_core::PropertyRegionTarget::all_cells()
            }
            _ => runmat_analysis_core::PropertyRegionTarget::all_solids(),
        },
        expected_material_id: "mat_steel".to_string(),
        assigned_material_id: "mat_steel".to_string(),
        confidence: EvidenceConfidence::Verified,
    }];
    match spec.kind {
        TestModelKind::LinearStaticStructural => {
            model.sequence = AnalysisSequence::from_steps(
                "test_linear_static",
                vec![ProcedureStep::structural_static("step_default_static")],
            );
        }
        TestModelKind::ModalStructural => {
            model.sequence = AnalysisSequence::from_steps(
                "test_modal",
                vec![ProcedureStep::structural_modal("step_default_modal")],
            );
            model.signals.clear();
            model.excitations.clear();
            model.excitation_cases.clear();
        }
        TestModelKind::AcousticHarmonic => {
            for material in &mut model.materials {
                material.acoustic = Some(runmat_analysis_core::MaterialAcousticModel {
                    speed_of_sound_m_per_s: runmat_types::physical::SpeedOfSound::new(343.0),
                    damping_ratio: runmat_types::physical::DampingRatio::new(0.02),
                });
            }
            model.sequence = AnalysisSequence::from_steps(
                "test_acoustic_harmonic",
                vec![ProcedureStep::acoustic_harmonic(
                    "step_default_acoustic_harmonic",
                )],
            );
            model.constraints.clear();
            model.boundaries = vec![runmat_analysis_core::AnalysisBoundary {
                boundary_id: "bc_default_acoustic_rigid_wall".to_string(),
                target: RegionTarget::boundary(runmat_meshing_core::fixtures::fixture_face("root")),
                operator: runmat_analysis_core::BoundaryOperator::AcousticRigidWall,
            }];
            model.signals.clear();
            model.excitations.clear();
            model.excitation_cases.clear();
            append_constant_excitation(
                &mut model,
                "excitation_default_acoustic_pressure",
                runmat_meshing_core::fixtures::fixture_face("tip"),
                ConstantExcitation::AcousticPressure(1.0),
            );
        }
        TestModelKind::ElectromagneticHarmonic => {
            for material in &mut model.materials {
                material.electrical = Some(test_electrical_material());
            }
            model.electromagnetic = Some(ElectromagneticDomain::Harmonic {
                reference_frequency: Frequency::new(60.0),
            });
            model.sequence = AnalysisSequence::from_steps(
                "test_electromagnetic_harmonic",
                vec![ProcedureStep::electromagnetic_harmonic(
                    "step_default_electromagnetic",
                )],
            );
            model.sequence.steps[0].independent_domain =
                runmat_analysis_core::IndependentDomain::FrequencyHertz(vec![60.0]);
            model.constraints = vec![runmat_analysis_core::AnalysisConstraint::Electrical(
                runmat_analysis_core::ElectricalConstraint {
                    constraint_id: "bc_default_em_pec".to_string(),
                    kind:
                        runmat_analysis_core::ElectricalConstraintKind::ElectricPerfectConductor {
                            target: RegionTarget::boundary(
                                runmat_meshing_core::fixtures::fixture_face("root"),
                            ),
                        },
                },
            )];
            model.boundaries.clear();
            model.signals.clear();
            model.excitations.clear();
            model.excitation_cases.clear();
            append_constant_excitation(
                &mut model,
                "excitation_default_em_current",
                "tip",
                ConstantExcitation::ElectricCurrent {
                    real_amperes: 100.0,
                    imaginary_amperes: 0.0,
                    frequency_hz: None,
                },
            );
        }
    }
    Ok(OperationEnvelope::new(
        "test.explicit_model",
        "test.explicit_model/v1",
        &context,
        model,
    ))
}

fn set_model_thermo_coupling(model: &mut AnalysisModel, coupling: TestThermoMechanicalCoupling) {
    for material in &mut model.materials {
        let thermal = material.thermal.as_mut().expect("thermal fixture material");
        thermal.reference_temperature_k = runmat_types::physical::Temperature::new(
            coupling.reference_temperature_k.canonical_value(),
        );
        thermal.expansion_coefficient_per_k = runmat_types::physical::ThermalExpansion::new(
            coupling.thermal_expansion_coefficient.canonical_value(),
        );
    }
    append_scalar_constant_excitation(
        model,
        "thermo_temperature",
        Temperature::new(
            coupling.reference_temperature_k.canonical_value()
                + coupling.applied_temperature_delta_k.canonical_value(),
        ),
        ExcitationTarget::WholeModel,
        None,
        PhysicalSignal::Temperature,
        ExcitationAction::TemperatureField,
    );
    for (index, delta) in coupling.region_temperature_deltas.iter().enumerate() {
        append_scalar_constant_excitation(
            model,
            &format!("thermo_region_temperature_{index}"),
            Temperature::new(
                coupling.reference_temperature_k.canonical_value() + delta.temperature_delta_k,
            ),
            ExcitationTarget::Region(RegionTarget::volume(&delta.region_id)),
            None,
            PhysicalSignal::Temperature,
            ExcitationAction::TemperatureField,
        );
    }
    model.thermo_mechanical = Some(runmat_analysis_core::ThermoMechanicalDomain {
        reference_temperature: Temperature::new(coupling.reference_temperature_k.canonical_value()),
    });
}

fn set_model_electro_coupling(model: &mut AnalysisModel, coupling: TestElectroThermalCoupling) {
    for material in &mut model.materials {
        material.electrical = Some(runmat_analysis_core::MaterialElectricalModel {
            reference_temperature_k: runmat_types::physical::Temperature::new(
                coupling.reference_temperature_k.canonical_value(),
            ),
            conductivity_s_per_m: runmat_types::physical::ElectricalConductivity::new(
                coupling.base_electrical_conductivity_s_per_m,
            ),
            resistive_heating_coefficient: runmat_types::physical::InverseTemperature::new(
                coupling.resistive_heating_coefficient.canonical_value(),
            ),
            relative_permittivity: runmat_types::physical::RelativePermittivity::new(1.0),
            relative_permeability: runmat_types::physical::RelativePermeability::new(1.0),
            conductivity_frequency_response: Vec::new(),
        });
    }
    append_scalar_constant_excitation(
        model,
        "electro_thermal_potential",
        ElectricPotential::new(coupling.applied_voltage_v),
        ExcitationTarget::Region(RegionTarget::boundary(
            runmat_meshing_core::fixtures::fixture_face("tip"),
        )),
        None,
        PhysicalSignal::ElectricPotential,
        ExcitationAction::TerminalVoltage,
    );
    model.electro_thermal = Some(runmat_analysis_core::ElectroThermalDomain {
        reference_temperature: Temperature::new(coupling.reference_temperature_k.canonical_value()),
    });
}

fn set_model_plasticity(
    model: &mut AnalysisModel,
    plasticity: runmat_analysis_core::MaterialPlasticModel,
) {
    for material in &mut model.materials {
        material.plastic = Some(plasticity.clone());
    }
}

fn set_model_contact(model: &mut AnalysisModel, friction_coefficient: f64) {
    model.interfaces = vec![runmat_analysis_core::AnalysisInterface {
        interface_id: "contact_1".to_string(),
        primary: runmat_analysis_core::RegionTarget::boundary(
            runmat_meshing_core::fixtures::fixture_face("root"),
        ),
        secondary: runmat_analysis_core::RegionTarget::boundary(
            runmat_meshing_core::fixtures::fixture_face("tip"),
        ),
        kind: runmat_analysis_core::AnalysisInterfaceKind::Contact(
            runmat_analysis_core::ContactInterfaceModel {
                friction: if friction_coefficient == 0.0 {
                    runmat_analysis_core::ContactFriction::Frictionless
                } else {
                    runmat_analysis_core::ContactFriction::Coulomb {
                        coefficient: runmat_analysis_core::FrictionCoefficient::new(
                            friction_coefficient,
                        ),
                    }
                },
            },
        ),
    }];
}

fn sample_geometry_asset() -> GeometryAsset {
    GeometryAsset {
        geometry_id: "geo:beam".to_string(),
        source: GeometrySource {
            unit_evidence: Default::default(),
            path: "/fixtures/beam.stl".to_string(),
            sha256: "hash-beam".to_string(),
            importer_version: "stl/v1".to_string(),
        },
        source_geometry: SourceGeometry {
            kind: SourceGeometryKind::Mesh,
            assembly: None,
            material_evidence: Vec::new(),
            exchange: None,
            cad_evaluators: Vec::new(),
        },
        tessellation_profile: TessellationProfile::default(),
        coordinate_units: None,
        revision: 2,
        meshes: vec![MeshDescriptor {
            mesh_id: "mesh_1".to_string(),
            kind: MeshKind::Surface,
            vertex_count: 3,
            element_count: 1,
        }],
        surface_meshes: vec![SurfaceMesh::new(
            "mesh_1",
            vec![[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0]],
            vec![[0, 1, 2]],
        )],
        regions: vec![Region {
            region_id: "region_default".to_string(),
            name: "Default Region".into(),
            tag: Some("mesh_default".to_string()),
            cad_ownership: None,
        }],
        region_entity_mappings: vec![RegionEntityMapping::all_faces(
            "region_default",
            "mesh_1",
            1,
        )],
        diagnostics: Vec::new(),
    }
}

fn closed_cube_geometry_asset() -> GeometryAsset {
    GeometryAsset {
        geometry_id: "geo:closed_cube".to_string(),
        source: GeometrySource {
            unit_evidence: runmat_geometry_core::physical::LengthUnitEvidence::explicit_fallback(
                runmat_geometry_core::LengthUnit::Meter,
                "canonical solver fixture",
            ),
            path: "/fixtures/closed_cube.step".to_string(),
            sha256: "hash-closed-cube".to_string(),
            importer_version: "test".to_string(),
        },
        source_geometry: SourceGeometry {
            kind: SourceGeometryKind::Cad,
            assembly: None,
            material_evidence: Vec::new(),
            exchange: None,
            cad_evaluators: Vec::new(),
        },
        tessellation_profile: TessellationProfile::default(),
        coordinate_units: None,
        revision: 1,
        meshes: vec![MeshDescriptor {
            mesh_id: "cube_surface".to_string(),
            kind: MeshKind::Surface,
            vertex_count: 8,
            element_count: 12,
        }],
        surface_meshes: vec![SurfaceMesh::new(
            "cube_surface",
            vec![
                [0.0, 0.0, 0.0],
                [1.0, 0.0, 0.0],
                [1.0, 1.0, 0.0],
                [0.0, 1.0, 0.0],
                [0.0, 0.0, 1.0],
                [1.0, 0.0, 1.0],
                [1.0, 1.0, 1.0],
                [0.0, 1.0, 1.0],
            ],
            vec![
                [0, 2, 1],
                [0, 3, 2],
                [4, 5, 6],
                [4, 6, 7],
                [0, 1, 5],
                [0, 5, 4],
                [1, 2, 6],
                [1, 6, 5],
                [2, 3, 7],
                [2, 7, 6],
                [3, 0, 4],
                [3, 4, 7],
            ],
        )],
        regions: vec![
            Region {
                region_id: "root".to_string(),
                name: "root".into(),
                tag: Some("fixed".to_string()),
                cad_ownership: None,
            },
            Region {
                region_id: "tip".to_string(),
                name: "tip".into(),
                tag: Some("load".to_string()),
                cad_ownership: None,
            },
        ],
        region_entity_mappings: vec![
            RegionEntityMapping::new(
                "root",
                "cube_surface",
                EntityKind::Face,
                vec![EntityIdRange::new(0, 2)],
            ),
            RegionEntityMapping::new(
                "tip",
                "cube_surface",
                EntityKind::Face,
                vec![EntityIdRange::new(2, 2)],
            ),
        ],
        diagnostics: Vec::new(),
    }
}

fn sample_linear_static_study_spec() -> AnalysisStudySpec {
    linear_static_study_spec_for_geometry(closed_cube_geometry_asset())
}

fn linear_static_study_spec_for_geometry(geometry: GeometryAsset) -> AnalysisStudySpec {
    let procedure_step = ProcedureStep::structural_static("step_default_static");
    let model = test_model_for_geometry(
        &geometry,
        TestModelSpec {
            model_id: "study_model_linear_static_001".to_string(),
            kind: TestModelKind::LinearStaticStructural,
        },
        OperationContext::new(None, None),
    )
    .expect("explicit structural test model")
    .data;
    AnalysisStudySpec {
        study_id: "study_linear_static_001".to_string(),
        geometry,
        model_source_evidence: None,
        model,
        execution: AnalysisExecutionPlan {
            steps: vec![AnalysisExecutionStep {
                step_id: procedure_step.step_id.clone(),
                backend: ComputeBackend::Cpu,
                accelerator_admission: Default::default(),
                solver_options: AnalysisSolverOptions::defaults_for(&procedure_step).unwrap(),
            }],
        },
        meshing_settings: Some(runmat_meshing_core::MeshingRequestSettings::default()),
        immutable_values: Vec::new(),
        solver_mesh_artifact_path: None,
        meshing_evidence_artifact_path: None,
    }
}

fn sample_electromagnetic_study_spec() -> AnalysisStudySpec {
    let procedure_step = ProcedureStep::electromagnetic_harmonic("step_default_electromagnetic");
    let geometry = sample_geometry_asset();
    let model = test_model_for_geometry(
        &geometry,
        TestModelSpec {
            model_id: "study_model_electromagnetic_001".to_string(),
            kind: TestModelKind::ElectromagneticHarmonic,
        },
        OperationContext::new(None, None),
    )
    .expect("explicit electromagnetic test model")
    .data;
    AnalysisStudySpec {
        study_id: "study_electromagnetic_001".to_string(),
        geometry,
        model_source_evidence: None,
        model,
        execution: AnalysisExecutionPlan {
            steps: vec![AnalysisExecutionStep {
                step_id: procedure_step.step_id.clone(),
                backend: ComputeBackend::Cpu,
                accelerator_admission: Default::default(),
                solver_options: AnalysisSolverOptions::defaults_for(&procedure_step).unwrap(),
            }],
        },
        meshing_settings: None,
        immutable_values: Vec::new(),
        solver_mesh_artifact_path: None,
        meshing_evidence_artifact_path: None,
    }
}

#[test]
fn study_content_digest_excludes_artifact_destinations() {
    let mut first = sample_linear_static_study_spec();
    first.solver_mesh_artifact_path = Some("first/solver-mesh.rmmesh".to_string());
    first.meshing_evidence_artifact_path = Some("first/meshing-evidence.json".to_string());
    let mut second = first.clone();
    second.solver_mesh_artifact_path = Some("second/solver-mesh.rmmesh".to_string());
    second.meshing_evidence_artifact_path = Some("second/meshing-evidence.json".to_string());

    assert_eq!(
        analysis_study_content_digest(&first),
        analysis_study_content_digest(&second)
    );
    second.execution.steps[0].backend = ComputeBackend::Gpu;
    assert_ne!(
        analysis_study_content_digest(&first),
        analysis_study_content_digest(&second)
    );
}

#[test]
fn runtime_contracts_reject_retired_mesh_artifact_field_names() {
    let mut study = serde_json::to_value(sample_linear_static_study_spec())
        .expect("study contract should serialize");
    study
        .as_object_mut()
        .expect("study should serialize as an object")
        .insert(
            "analysis_mesh_artifact_path".to_owned(),
            serde_json::Value::String("retired.json".to_owned()),
        );
    assert!(serde_json::from_value::<AnalysisStudySpec>(study).is_err());

    let mut options =
        serde_json::to_value(AnalysisRunOptions::default()).expect("run options should serialize");
    options
        .as_object_mut()
        .expect("run options should serialize as an object")
        .insert(
            "analysis_mesh_artifact_path".to_owned(),
            serde_json::Value::String("retired.json".to_owned()),
        );
    assert!(serde_json::from_value::<AnalysisRunOptions>(options).is_err());
}

#[test]
fn transient_run_option_presets_are_ordered_for_cost_vs_accuracy() {
    let coarse = AnalysisTransientRunOptions::coarse();
    let balanced = AnalysisTransientRunOptions::balanced();
    let solid = AnalysisTransientRunOptions::solid_recommended();
    let high_accuracy = AnalysisTransientRunOptions::high_accuracy();

    assert!(coarse.time_step_policy.interval_count() < balanced.time_step_policy.interval_count());
    assert!(
        balanced.time_step_policy.interval_count()
            < high_accuracy.time_step_policy.interval_count()
    );
    assert!(coarse.residual_target > balanced.residual_target);
    assert!(balanced.residual_target > high_accuracy.residual_target);
    let serialized = serde_json::to_value(&balanced).unwrap();
    assert!(serialized.get("time_step_policy").is_some());
    assert!(serialized.get("adaptive_time_step").is_none());
    assert!(serialized.get("precision_mode").is_none());
    let mut with_ignored_precision = serialized;
    with_ignored_precision
        .as_object_mut()
        .expect("transient options should serialize as an object")
        .insert("precision_mode".to_string(), serde_json::json!("fp32"));
    assert!(serde_json::from_value::<AnalysisTransientRunOptions>(with_ignored_precision).is_err());

    assert_eq!(solid.quality_policy, QualityPolicy::Balanced);
    assert!(solid.deterministic_mode);
    assert_eq!(
        solid.time_step_policy,
        AnalysisStructuralTimeStepPolicy::Fixed { interval_count: 10 }
    );
}

#[test]
fn modal_run_option_presets_are_ordered_for_cost_vs_accuracy() {
    let coarse = AnalysisModalRunOptions::coarse();
    let balanced = AnalysisModalRunOptions::balanced();
    let high_accuracy = AnalysisModalRunOptions::high_accuracy();

    assert!(coarse.truncation.requested_mode_count() < balanced.truncation.requested_mode_count());
    assert!(
        balanced.truncation.requested_mode_count()
            < high_accuracy.truncation.requested_mode_count()
    );
    assert!(
        coarse.convergence.relative_residual_tolerance()
            > balanced.convergence.relative_residual_tolerance()
    );
    assert!(
        balanced.convergence.relative_residual_tolerance()
            > high_accuracy.convergence.relative_residual_tolerance()
    );
    assert!(
        coarse.convergence.maximum_subspace_iterations()
            < balanced.convergence.maximum_subspace_iterations()
    );
    assert!(
        balanced.convergence.maximum_subspace_iterations()
            < high_accuracy.convergence.maximum_subspace_iterations()
    );
}

#[test]
fn nonlinear_run_option_presets_are_ordered_for_cost_vs_accuracy() {
    let coarse = AnalysisNonlinearRunOptions::coarse();
    let balanced = AnalysisNonlinearRunOptions::balanced();
    let solid = AnalysisNonlinearRunOptions::solid_recommended();
    let high_accuracy = AnalysisNonlinearRunOptions::high_accuracy();

    assert!(coarse.increment_count < balanced.increment_count);
    assert!(balanced.increment_count <= solid.increment_count);
    assert!(solid.increment_count <= high_accuracy.increment_count);

    assert!(coarse.max_newton_iters < balanced.max_newton_iters);
    assert!(balanced.max_newton_iters <= solid.max_newton_iters);
    assert!(solid.max_newton_iters <= high_accuracy.max_newton_iters);

    assert!(coarse.tolerance > balanced.tolerance);
    assert!(balanced.tolerance >= solid.tolerance);
    assert!(solid.tolerance >= high_accuracy.tolerance);

    assert_eq!(solid.quality_policy, QualityPolicy::Balanced);
    assert!(solid.deterministic_mode);
    assert_eq!(solid.precision_mode, PrecisionMode::Fp64);
    assert!(solid.line_search);
    assert!(solid.max_line_search_backtracks >= balanced.max_line_search_backtracks);
}

#[test]
fn analysis_validate_study_reports_invalid_study_id() {
    let _guard = analysis_test_guard();
    let root = temp_artifact_root("validate-study-evidence");
    let _ = fs::remove_dir_all(&root);
    let env_guard = EnvVarRestoreGuard {
        key: "RUNMAT_FEA_STUDY_ARTIFACT_ROOT",
        previous: std::env::var("RUNMAT_FEA_STUDY_ARTIFACT_ROOT").ok(),
    };
    std::env::set_var("RUNMAT_FEA_STUDY_ARTIFACT_ROOT", root.display().to_string());
    let mut spec = sample_linear_static_study_spec();
    spec.study_id = "   ".to_string();

    let envelope = analysis_validate_study_op(&spec, OperationContext::new(None, None))
        .expect("study validation should return typed output");

    assert_eq!(envelope.operation, "fea.validate_study");
    assert_eq!(envelope.op_version, "fea.validate_study/v1");
    assert!(!envelope.data.valid);
    assert!(envelope
        .data
        .issue_codes
        .iter()
        .any(|code| code == "RM.FEA.STUDY.ID_EMPTY"));
    assert!(envelope.data.issues.iter().any(|issue| {
        issue.code == "RM.FEA.STUDY.ID_EMPTY"
            && issue.message.contains("study_id must be non-empty")
    }));
    assert!(envelope
        .data
        .evidence_artifact_path
        .contains("validate_sha256_"));
    drop(env_guard);
    let _ = fs::remove_dir_all(&root);
}

#[test]
fn analysis_validate_study_preserves_typed_model_failure_detail() {
    let _guard = analysis_test_guard();
    let mut spec = sample_linear_static_study_spec();
    let mut model = sample_model();
    model.materials.clear();
    spec.model = model;

    let envelope = analysis_validate_study_op(&spec, OperationContext::new(None, None))
        .expect("study validation should return typed output");

    let issue = envelope
        .data
        .issues
        .iter()
        .find(|issue| issue.code == "RM.FEA.STUDY.MODEL_INVALID")
        .expect("model validation issue");
    assert!(issue
        .message
        .contains("ANALYSIS_VALIDATION_MISSING_MATERIALS"));
}

#[test]
fn analysis_validate_study_rejects_solver_options_for_a_different_procedure() {
    let _guard = analysis_test_guard();
    let mut spec = sample_linear_static_study_spec();
    spec.execution.steps[0].solver_options =
        AnalysisSolverOptions::MaxwellHarmonic(AnalysisMaxwellRunOptions::default());

    let envelope = analysis_validate_study_op(&spec, OperationContext::new(None, None))
        .expect("study validation should return typed output");

    assert!(!envelope.data.valid);
    assert!(envelope
        .data
        .issue_codes
        .iter()
        .any(|code| code == "RM.FEA.STUDY.EXECUTION_PLAN_INVALID"));
    assert!(envelope.data.issues.iter().any(|issue| {
        issue.code == "RM.FEA.STUDY.EXECUTION_PLAN_INVALID"
            && issue.message.contains("canonical model procedure step")
    }));
}

#[test]
fn analysis_plan_study_returns_canonical_linear_static_sequence() {
    let _guard = analysis_test_guard();
    let root = temp_artifact_root("plan-study-evidence");
    let _ = fs::remove_dir_all(&root);
    let env_guard = EnvVarRestoreGuard {
        key: "RUNMAT_FEA_STUDY_ARTIFACT_ROOT",
        previous: std::env::var("RUNMAT_FEA_STUDY_ARTIFACT_ROOT").ok(),
    };
    std::env::set_var("RUNMAT_FEA_STUDY_ARTIFACT_ROOT", root.display().to_string());
    let spec = sample_linear_static_study_spec();

    let envelope = analysis_plan_study_op(&spec, OperationContext::new(None, None))
        .expect("study plan should succeed");

    assert_eq!(envelope.operation, "fea.plan_study");
    assert_eq!(envelope.op_version, "fea.plan_study/v1");
    assert_eq!(envelope.data.study_id, spec.study_id);
    assert_eq!(envelope.data.model_id, spec.model.model_id.0);
    assert_eq!(envelope.data.steps.len(), 1);
    assert_eq!(
        envelope.data.steps[0].run_operation,
        "fea.run_linear_static"
    );
    assert_eq!(
        envelope.data.steps[0].run_op_version,
        "fea.run_linear_static/v1"
    );
    assert!(matches!(
        envelope.data.steps[0].execution.solver_options,
        AnalysisSolverOptions::StructuralLinear(_)
    ));
    assert_eq!(
        envelope.data.operation_sequence,
        vec![
            "fea.validate/v1".to_string(),
            "fea.run_linear_static/v1".to_string(),
        ]
    );
    assert!(envelope.data.study_fingerprint.starts_with("sha256:"));
    assert!(envelope
        .data
        .evidence_artifact_path
        .contains("plan_sha256_"));
    drop(env_guard);
    let _ = fs::remove_dir_all(&root);
}

#[test]
fn analysis_plan_study_sweep_returns_typed_plan_entries() {
    let _guard = analysis_test_guard();
    let root = temp_artifact_root("plan-study-sweep-evidence");
    let _ = fs::remove_dir_all(&root);
    let env_guard = EnvVarRestoreGuard {
        key: "RUNMAT_FEA_STUDY_ARTIFACT_ROOT",
        previous: std::env::var("RUNMAT_FEA_STUDY_ARTIFACT_ROOT").ok(),
    };
    std::env::set_var("RUNMAT_FEA_STUDY_ARTIFACT_ROOT", root.display().to_string());
    let sweep_spec = AnalysisStudySweepSpec {
        sweep_id: "study_sweep_plan_001".to_string(),
        studies: vec![
            sample_linear_static_study_spec(),
            sample_electromagnetic_study_spec(),
        ],
        fail_fast: true,
    };

    let envelope = analysis_plan_study_sweep_op(&sweep_spec, OperationContext::new(None, None))
        .expect("study sweep plan should succeed");

    assert_eq!(envelope.operation, "fea.plan_study_sweep");
    assert_eq!(envelope.op_version, "fea.plan_study_sweep/v1");
    assert_eq!(envelope.data.sweep_id, "study_sweep_plan_001");
    assert_eq!(envelope.data.study_count, 2);
    assert_eq!(envelope.data.planned_count, 2);
    assert_eq!(envelope.data.failed_count, 0);
    assert!(envelope.data.failure_entries.is_empty());
    assert_eq!(envelope.data.plan_entries.len(), 2);
    assert!(envelope
        .data
        .plan_entries
        .iter()
        .all(|entry| entry.study_fingerprint.starts_with("sha256:")));
    assert!(envelope
        .data
        .plan_entries
        .iter()
        .any(
            |entry| entry.steps[0].procedure_step.family == PhysicsFamily::Structural
                && entry.steps[0].procedure_step.procedure == AnalysisProcedure::Static
        ));
    assert!(envelope
        .data
        .plan_entries
        .iter()
        .any(
            |entry| entry.steps[0].procedure_step.family == PhysicsFamily::Electromagnetic
                && entry.steps[0].procedure_step.procedure == AnalysisProcedure::HarmonicResponse
        ));
    assert!(envelope.data.evidence_artifact_path.ends_with("plan.json"));
    assert!(PathBuf::from(&envelope.data.evidence_artifact_path).exists());
    drop(env_guard);
    let _ = fs::remove_dir_all(&root);
}

#[test]
fn analysis_plan_study_sweep_rejects_empty_study_set() {
    let _guard = analysis_test_guard();
    let spec = AnalysisStudySweepSpec {
        sweep_id: "study_sweep_plan_empty".to_string(),
        studies: Vec::new(),
        fail_fast: true,
    };

    let err = analysis_plan_study_sweep_op(&spec, OperationContext::new(None, None))
        .expect_err("empty sweep plan should be rejected");
    assert_eq!(err.operation, "fea.plan_study_sweep");
    assert_eq!(err.op_version, "fea.plan_study_sweep/v1");
    assert_eq!(err.error_code, "RM.FEA.PLAN_STUDY_SWEEP.INVALID_SPEC");
}

#[test]
fn analysis_plan_study_sweep_can_continue_on_study_failure() {
    let _guard = analysis_test_guard();
    let root = temp_artifact_root("plan-study-sweep-continue-evidence");
    let _ = fs::remove_dir_all(&root);
    let _runtime_guard = scoped_study_artifact_root(&root);
    let mut invalid = sample_linear_static_study_spec();
    invalid.study_id = "   ".to_string();
    let spec = AnalysisStudySweepSpec {
        sweep_id: "study_sweep_plan_continue".to_string(),
        studies: vec![sample_linear_static_study_spec(), invalid],
        fail_fast: false,
    };

    let envelope = analysis_plan_study_sweep_op(&spec, OperationContext::new(None, None))
        .expect("continue-on-failure sweep planning should succeed");

    assert_eq!(envelope.data.study_count, 2);
    assert_eq!(envelope.data.planned_count, 1);
    assert_eq!(envelope.data.failed_count, 1);
    assert_eq!(envelope.data.plan_entries.len(), 1);
    assert_eq!(envelope.data.failure_entries.len(), 1);
    assert_eq!(envelope.data.failure_entries[0].study_index, 1);
    assert_eq!(
        envelope.data.failure_entries[0].error_code,
        "RM.FEA.PLAN_STUDY.INVALID_SPEC"
    );
    let _ = fs::remove_dir_all(&root);
}

#[test]
fn analysis_run_study_executes_linear_static_path() {
    let _guard = analysis_test_guard();
    storage::reset_artifact_store_for_tests();
    let root = temp_artifact_root("run-study-evidence");
    let _ = fs::remove_dir_all(&root);
    let env_guard = EnvVarRestoreGuard {
        key: "RUNMAT_FEA_STUDY_ARTIFACT_ROOT",
        previous: std::env::var("RUNMAT_FEA_STUDY_ARTIFACT_ROOT").ok(),
    };
    std::env::set_var("RUNMAT_FEA_STUDY_ARTIFACT_ROOT", root.display().to_string());
    let mut spec = sample_linear_static_study_spec();
    let (solver_mesh_root, solver_mesh_path) =
        write_ready_minimal_analysis_mesh_artifact("run-study-solver-mesh");
    spec.solver_mesh_artifact_path = Some(solver_mesh_path.display().to_string());

    let envelope = analysis_run_study_op(&spec, OperationContext::new(None, None))
        .expect("study run should succeed");

    assert_eq!(envelope.operation, "fea.run_study");
    assert_eq!(envelope.op_version, "fea.run_study/v1");
    assert_eq!(envelope.data.study_id, spec.study_id);
    assert_eq!(envelope.data.model_id, spec.model.model_id.0);
    assert_eq!(
        envelope.data.steps[0].procedure_step.family,
        PhysicsFamily::Structural
    );
    assert_eq!(
        envelope.data.steps[0].procedure_step.procedure,
        AnalysisProcedure::Static
    );
    assert_eq!(
        envelope.data.steps[0].execution.backend,
        ComputeBackend::Cpu
    );
    assert!(matches!(
        envelope.data.steps[0].execution.solver_options,
        AnalysisSolverOptions::StructuralLinear(_)
    ));
    assert_eq!(
        envelope.data.steps[0].run_operation,
        "fea.run_linear_static"
    );
    assert_eq!(
        envelope.data.steps[0].run_op_version,
        "fea.run_linear_static/v1"
    );
    assert_eq!(
        envelope.data.operation_sequence,
        vec![
            "fea.validate/v1".to_string(),
            "fea.run_linear_static/v1".to_string(),
        ]
    );
    assert!(envelope.data.study_fingerprint.starts_with("sha256:"));
    assert!(envelope.data.steps[0].run_id.starts_with("run_"));
    let evidence_path = Path::new(&envelope.data.evidence_artifact_path);
    assert_eq!(
        evidence_path
            .parent()
            .and_then(Path::file_name)
            .and_then(|name| name.to_str()),
        Some(
            format!(
                "sha256_{}",
                envelope
                    .data
                    .study_fingerprint
                    .trim_start_matches("sha256:")
            )
            .as_str()
        )
    );
    let evidence_file_name = evidence_path
        .file_name()
        .and_then(|name| name.to_str())
        .expect("evidence file name");
    let evidence_digest = evidence_file_name
        .strip_prefix("sequence-run_sha256_")
        .and_then(|suffix| suffix.strip_suffix(".json"))
        .expect("sequence evidence is content addressed");
    assert_eq!(evidence_digest.len(), 64);
    assert!(evidence_digest
        .bytes()
        .all(|byte| byte.is_ascii_hexdigit() && !byte.is_ascii_uppercase()));

    let persisted = storage::load_run_result(&envelope.data.steps[0].run_id)
        .expect("run load should succeed")
        .expect("run should be persisted");
    assert_eq!(persisted.run_id, envelope.data.steps[0].run_id);
    assert_eq!(persisted.run_status, envelope.data.run_status);
    assert_eq!(
        persisted.numerically_acceptable,
        envelope.data.numerically_acceptable
    );
    assert_eq!(
        persisted.solver_convergence,
        envelope.data.solver_convergence
    );
    assert_eq!(persisted.result_quality, envelope.data.result_quality);
    assert_eq!(
        persisted.quality_reasons,
        envelope.data.steps[0].quality_reasons
    );
    assert_eq!(persisted.provenance, envelope.data.steps[0].provenance);
    let solver_mesh_artifact_path = envelope
        .data
        .solver_mesh_artifact_path
        .as_ref()
        .expect("study run should preserve the canonical solver mesh artifact");
    let solver_mesh = runmat_meshing_core::SolverMeshArtifact::canonical_decode(
        &fs::read(solver_mesh_artifact_path).expect("read canonical solver mesh artifact"),
    )
    .expect("decode canonical solver mesh artifact");
    let solver_node_count = solver_mesh.topology.nodes.len();
    let solver_boundary_face_count = solver_mesh.topology.boundary_faces.len();
    let solver_volume_element_count = solver_mesh.topology.volume_elements.len();
    let render_topology = persisted
        .render_topology
        .as_ref()
        .expect("study run should persist solver render topology");
    assert_eq!(
        render_topology.schema_version,
        "analysis_render_topology/v1"
    );
    assert_eq!(render_topology.meshes.len(), 1);
    assert_eq!(render_topology.meshes[0].vertices.len(), solver_node_count);
    assert_eq!(
        render_topology.meshes[0].triangles.len(),
        solver_boundary_face_count
    );
    let von_mises = persisted
        .run
        .field(FEA_FIELD_STRUCTURAL_VON_MISES)
        .expect("study run should persist von Mises field");
    assert_eq!(von_mises.shape, vec![solver_volume_element_count]);
    let run_evidence_payload: serde_json::Value = serde_json::from_slice(
        &fs::read(&envelope.data.evidence_artifact_path).expect("read run evidence artifact"),
    )
    .expect("parse run evidence artifact");
    assert_eq!(run_evidence_payload["$evidence"]["stage"], "sequence-run");
    assert_eq!(
        run_evidence_payload["$evidence"]["study_fingerprint"],
        envelope.data.study_fingerprint
    );
    assert_eq!(
        run_evidence_payload["$evidence"]["content_digest"],
        format!("sha256:{evidence_digest}")
    );
    assert!(run_evidence_payload.get("model_template").is_none());
    drop(env_guard);
    let _ = fs::remove_dir_all(&root);
    let _ = fs::remove_dir_all(&solver_mesh_root);
}

#[test]
fn fea_step_result_selects_one_procedure_step_by_authored_id() {
    let _guard = analysis_test_guard();
    storage::reset_artifact_store_for_tests();
    let root = temp_artifact_root("step-result-selection");
    let _ = fs::remove_dir_all(&root);
    let env_guard = EnvVarRestoreGuard {
        key: "RUNMAT_FEA_STUDY_ARTIFACT_ROOT",
        previous: std::env::var("RUNMAT_FEA_STUDY_ARTIFACT_ROOT").ok(),
    };
    std::env::set_var("RUNMAT_FEA_STUDY_ARTIFACT_ROOT", root.display().to_string());
    let mut spec = sample_linear_static_study_spec();
    let (solver_mesh_root, solver_mesh_path) =
        write_ready_minimal_analysis_mesh_artifact("step-result-solver-mesh");
    spec.solver_mesh_artifact_path = Some(solver_mesh_path.display().to_string());
    let envelope = analysis_run_study_op(&spec, OperationContext::new(None, None))
        .expect("study run should succeed");
    let step = &envelope.data.steps[0];
    let run = crate::builtins::fea::run_results::run_result_object_for_tests(&envelope.data);

    let selected = futures::executor::block_on(crate::builtins::fea::run_results::step_result(
        run.clone(),
        step.procedure_step.step_id.0.clone(),
    ))
    .expect("the authored step id should select its result");
    let runmat_value::Value::Object(object) = &selected else {
        panic!("expected a fea.StepRunResult object");
    };
    assert_eq!(object.class_name().display_name(), "fea.StepRunResult");
    assert_eq!(
        object.properties().get("run_id"),
        Some(&runmat_value::Value::String(step.run_id.clone().into()))
    );

    let missing = futures::executor::block_on(crate::builtins::fea::run_results::step_result(
        run,
        "not_a_step".to_string(),
    ))
    .expect_err("an unknown step id must be rejected");
    assert_eq!(missing.identifier(), Some("RunMat:fea:InvalidInput"));
    drop(env_guard);
    let _ = fs::remove_dir_all(&root);
    let _ = fs::remove_dir_all(&solver_mesh_root);
}

struct CanonicalStructuralExecutor {
    calls: AtomicU64,
}

impl StructuralLinearExecutor for CanonicalStructuralExecutor {
    fn execute(
        &self,
        workload: &runmat_analysis_fea::execution::StructuralLinearWorkload,
        mesh: &runmat_meshing_core::SolverMeshArtifact,
    ) -> Result<runmat_analysis_fea::execution::StructuralLinearWorkloadResult, String> {
        use runmat_meshing_core::CanonicalMeshingContract;

        self.calls.fetch_add(1, Ordering::Relaxed);
        let workload =
            runmat_analysis_fea::execution::StructuralLinearWorkload::from_canonical_bytes(
                &workload
                    .canonical_bytes()
                    .map_err(|error| error.to_string())?,
            )
            .map_err(|error| error.to_string())?;
        let mesh = runmat_meshing_core::SolverMeshArtifact::canonical_decode(
            &mesh.canonical_encode().map_err(|error| error.to_string())?,
        )
        .map_err(|error| error.to_string())?;
        let run = runmat_analysis_fea::run_linear_static_with_options(
            &workload.model,
            workload.backend,
            runmat_analysis_fea::LinearStaticSolveOptions {
                preconditioner_kind: workload.preconditioner,
                algebra_backend_kind: workload.algebra_backend,
                solver_mesh_artifact_path: None,
                solver_mesh: Some(mesh.clone()),
                require_solver_mesh_for_solid: workload.require_solver_mesh_for_solid,
                initial_displacement: None,
                local_execution:
                    runmat_analysis_fea::planning::resource::LocalExecutionPlan::serial(),
            },
        )
        .map_err(|error| error.to_string())?;
        let constraint_plan =
            runmat_analysis_fea::constraints::CanonicalStructuralConstraintPlan::compile(
                &workload.model,
                &mesh,
            )
            .map_err(|error| error.to_string())?;
        let constraint_result =
            runmat_analysis_fea::constraints::CanonicalStructuralConstraintResult::recover(
                &constraint_plan,
                &mesh,
                &run.fields,
            )
            .map_err(|error| error.to_string())?;
        runmat_analysis_fea::execution::StructuralLinearWorkloadResult::from_canonical_bytes(
            &runmat_analysis_fea::execution::StructuralLinearWorkloadResult::new(
                run,
                constraint_result,
            )
            .map_err(|error| error.to_string())?
            .canonical_bytes()
            .map_err(|error| error.to_string())?,
        )
        .map_err(|error| error.to_string())
    }
}

struct CanonicalStudyPlacement {
    reverse_results: bool,
}

impl StructuralStudyPlacementExecutor for CanonicalStudyPlacement {
    fn allocated_cpu_millicores(&self, _study_count: usize) -> u32 {
        1000
    }
    fn execute<'a>(
        &'a self,
        entries: &'a [StructuralStudyPlacementEntry<'a>],
    ) -> StructuralStudyPlacementFuture<'a> {
        Box::pin(async move {
            let solver = CanonicalStructuralExecutor {
                calls: AtomicU64::new(0),
            };
            let mut results = Vec::with_capacity(entries.len());
            for entry in entries {
                let runmat_analysis_fea::execution::StructuralStudyWorkloads::LinearSequence {
                    workloads,
                } = entry.prepared.workloads()
                else {
                    return Err("prepared placement study is not a linear sequence".to_string());
                };
                let workload = workloads
                    .into_values()
                    .next()
                    .ok_or_else(|| "prepared placement study has no workload".to_string())?;
                let run = solver.execute(&workload, entry.prepared.solver_mesh())?;
                results.push(StructuralStudyPlacementResult {
                    study_index: entry.study_index,
                    outcome: Ok(test_structural_sequence_result(
                        entry.study,
                        &workload,
                        entry.prepared.solver_mesh(),
                        run,
                    )),
                });
            }
            if self.reverse_results {
                results.reverse();
            }
            Ok(results)
        })
    }
}

struct CancelledStudyPlacement;

impl StructuralStudyPlacementExecutor for CancelledStudyPlacement {
    fn allocated_cpu_millicores(&self, _study_count: usize) -> u32 {
        1000
    }
    fn execute<'a>(
        &'a self,
        entries: &'a [StructuralStudyPlacementEntry<'a>],
    ) -> StructuralStudyPlacementFuture<'a> {
        Box::pin(async move {
            Ok(entries
                .iter()
                .map(|entry| StructuralStudyPlacementResult {
                    study_index: entry.study_index,
                    outcome: Err(StructuralStudyExecutionFailure::Cancelled),
                })
                .collect())
        })
    }
}

struct FailingStudyPlacement {
    calls: AtomicU64,
}

impl StructuralStudyPlacementExecutor for FailingStudyPlacement {
    fn allocated_cpu_millicores(&self, _study_count: usize) -> u32 {
        1000
    }
    fn execute<'a>(
        &'a self,
        entries: &'a [StructuralStudyPlacementEntry<'a>],
    ) -> StructuralStudyPlacementFuture<'a> {
        self.calls.fetch_add(1, Ordering::Relaxed);
        Box::pin(async move {
            Ok(entries
                .iter()
                .map(|entry| StructuralStudyPlacementResult {
                    study_index: entry.study_index,
                    outcome: Err(StructuralStudyExecutionFailure::Failed(
                        "injected placement failure".to_string(),
                    )),
                })
                .collect())
        })
    }
}

#[test]
fn placement_lifecycle_restores_authored_sweep_order_before_publication() {
    let _guard = analysis_test_guard();
    storage::reset_artifact_store_for_tests();
    let root = temp_artifact_root("placement-lifecycle-sweep");
    let _ = fs::remove_dir_all(&root);
    let _runtime_guard = scoped_study_artifact_root(&root);
    let (solver_mesh_root, solver_mesh_path) =
        write_ready_minimal_analysis_mesh_artifact("placement-lifecycle-solver-mesh");
    let mut first = sample_linear_static_study_spec();
    first.study_id = "placement_first".to_string();
    first.model.model_id = AnalysisModelId("placement_first_model".to_string());
    first.solver_mesh_artifact_path = Some(solver_mesh_path.display().to_string());
    let mut second = first.clone();
    second.study_id = "placement_second".to_string();
    second.model.model_id = AnalysisModelId("placement_second_model".to_string());
    let sweep = AnalysisStudySweepSpec {
        sweep_id: "placement_order_sweep".to_string(),
        studies: vec![first, second],
        fail_fast: false,
    };

    let envelope = pollster::block_on(analysis_execute_structural_sweep_with_placement(
        &sweep,
        OperationContext::new(None, None),
        &CanonicalStudyPlacement {
            reverse_results: true,
        },
    ))
    .expect("placement lifecycle should publish the complete sweep");

    assert_eq!(envelope.data.run_entries.len(), 2);
    assert_eq!(envelope.data.run_entries[0].study_id, "placement_first");
    assert_eq!(envelope.data.run_entries[1].study_id, "placement_second");
    let _ = fs::remove_dir_all(&root);
    let _ = fs::remove_dir_all(&solver_mesh_root);
}

#[test]
fn placement_lifecycle_preserves_cancelled_terminal() {
    let _guard = analysis_test_guard();
    storage::reset_artifact_store_for_tests();
    let root = temp_artifact_root("placement-lifecycle-cancelled");
    let _ = fs::remove_dir_all(&root);
    let _runtime_guard = scoped_study_artifact_root(&root);
    let (solver_mesh_root, solver_mesh_path) =
        write_ready_minimal_analysis_mesh_artifact("placement-cancelled-solver-mesh");
    let mut spec = sample_linear_static_study_spec();
    spec.solver_mesh_artifact_path = Some(solver_mesh_path.display().to_string());

    let error = pollster::block_on(analysis_execute_structural_study_with_placement(
        &spec,
        OperationContext::new(None, None),
        &CancelledStudyPlacement,
    ))
    .expect_err("cancelled placement must remain cancelled");

    assert_eq!(error.error_type, OperationErrorType::Cancelled);
    assert_eq!(
        error.error_code,
        "RM.FEA.RUN_STUDY.PORTABLE_EXECUTION_CANCELLED"
    );
    let _ = fs::remove_dir_all(&root);
    let _ = fs::remove_dir_all(&solver_mesh_root);
}

#[test]
fn placement_lifecycle_owns_fail_fast_admission() {
    let _guard = analysis_test_guard();
    storage::reset_artifact_store_for_tests();
    let root = temp_artifact_root("placement-lifecycle-fail-fast");
    let _ = fs::remove_dir_all(&root);
    let _runtime_guard = scoped_study_artifact_root(&root);
    let (solver_mesh_root, solver_mesh_path) =
        write_ready_minimal_analysis_mesh_artifact("placement-fail-fast-solver-mesh");
    let mut first = sample_linear_static_study_spec();
    first.study_id = "placement_fail_fast_first".to_string();
    first.model.model_id = AnalysisModelId("placement_fail_fast_first_model".to_string());
    first.solver_mesh_artifact_path = Some(solver_mesh_path.display().to_string());
    let mut second = first.clone();
    second.study_id = "placement_fail_fast_second".to_string();
    second.model.model_id = AnalysisModelId("placement_fail_fast_second_model".to_string());
    let sweep = AnalysisStudySweepSpec {
        sweep_id: "placement_fail_fast_sweep".to_string(),
        studies: vec![first, second],
        fail_fast: true,
    };
    let placement = FailingStudyPlacement {
        calls: AtomicU64::new(0),
    };

    let error = pollster::block_on(analysis_execute_structural_sweep_with_placement(
        &sweep,
        OperationContext::new(None, None),
        &placement,
    ))
    .expect_err("fail-fast sweep must stop at its first failed study");

    assert_eq!(placement.calls.load(Ordering::Relaxed), 1);
    assert_eq!(error.error_code, "RM.FEA.RUN_STUDY_SWEEP.STUDY_FAILED");
    let _ = fs::remove_dir_all(&root);
    let _ = fs::remove_dir_all(&solver_mesh_root);
}

#[test]
fn portable_structural_study_reenters_shared_runtime_persistence() {
    let _guard = analysis_test_guard();
    storage::reset_artifact_store_for_tests();
    let root = temp_artifact_root("portable-run-study-evidence");
    let _ = fs::remove_dir_all(&root);
    let _runtime_guard = scoped_study_artifact_root(&root);
    let mut spec = sample_linear_static_study_spec();
    let (solver_mesh_root, solver_mesh_path) =
        write_ready_minimal_analysis_mesh_artifact("portable-run-study-solver-mesh");
    spec.solver_mesh_artifact_path = Some(solver_mesh_path.display().to_string());
    let executor = CanonicalStructuralExecutor {
        calls: AtomicU64::new(0),
    };

    let envelope = analysis_run_study_with_structural_executor_op(
        &spec,
        OperationContext::new(None, None),
        &executor,
    )
    .expect("portable structural study should succeed");

    assert_eq!(executor.calls.load(Ordering::Relaxed), 1);
    assert_eq!(envelope.operation, "fea.run_study");
    let persisted = storage::load_run_result(&envelope.data.steps[0].run_id)
        .expect("portable run load should succeed")
        .expect("portable run should be persisted");
    assert_eq!(persisted.run.identity, envelope.data.steps[0].identity);
    assert_eq!(persisted.run_status, envelope.data.run_status);
    assert!(Path::new(&envelope.data.evidence_artifact_path).is_file());
    let _ = fs::remove_dir_all(&root);
    let _ = fs::remove_dir_all(&solver_mesh_root);
}

#[test]
fn portable_structural_study_executes_every_independent_step_in_authored_order() {
    let _guard = analysis_test_guard();
    storage::reset_artifact_store_for_tests();
    let root = temp_artifact_root("portable-multi-step-study-evidence");
    let _ = fs::remove_dir_all(&root);
    let _runtime_guard = scoped_study_artifact_root(&root);
    let mut spec = sample_linear_static_study_spec();
    let (solver_mesh_root, solver_mesh_path) =
        write_ready_minimal_analysis_mesh_artifact("portable-multi-step-solver-mesh");
    spec.solver_mesh_artifact_path = Some(solver_mesh_path.display().to_string());
    let first = ProcedureStep::structural_static("step_static_first");
    let second = ProcedureStep::structural_static("step_static_second");
    let mut model = spec.model.clone();
    model.sequence = AnalysisSequence::from_steps(
        "sequence_two_independent_static_steps",
        vec![first.clone(), second.clone()],
    );
    spec.model = model;
    spec.execution = AnalysisExecutionPlan::for_sequence(
        &AnalysisSequence::from_steps(
            "sequence_two_independent_static_steps",
            vec![first.clone(), second.clone()],
        ),
        ComputeBackend::Cpu,
    )
    .expect("two-step execution plan should be valid");
    let executor = CanonicalStructuralExecutor {
        calls: AtomicU64::new(0),
    };

    let envelope = analysis_run_study_with_structural_executor_op(
        &spec,
        OperationContext::new(None, None),
        &executor,
    )
    .expect("independent structural sequence should succeed");

    assert_eq!(executor.calls.load(Ordering::Relaxed), 2);
    assert_eq!(envelope.data.steps.len(), 2);
    assert_eq!(envelope.data.steps[0].procedure_step.step_id, first.step_id);
    assert_eq!(
        envelope.data.steps[1].procedure_step.step_id,
        second.step_id
    );
    assert_ne!(envelope.data.steps[0].run_id, envelope.data.steps[1].run_id);
    for step in &envelope.data.steps {
        assert!(storage::load_run_result(&step.run_id)
            .expect("step run load should succeed")
            .is_some());
    }
    let _ = fs::remove_dir_all(&root);
    let _ = fs::remove_dir_all(&solver_mesh_root);
}

#[test]
fn prepared_structural_study_completes_without_reinterpreting_the_study() {
    let _guard = analysis_test_guard();
    storage::reset_artifact_store_for_tests();
    let root = temp_artifact_root("prepared-run-study-evidence");
    let _ = fs::remove_dir_all(&root);
    let _runtime_guard = scoped_study_artifact_root(&root);
    let mut spec = sample_linear_static_study_spec();
    let (solver_mesh_root, solver_mesh_path) =
        write_ready_minimal_analysis_mesh_artifact("prepared-run-study-solver-mesh");
    spec.solver_mesh_artifact_path = Some(solver_mesh_path.display().to_string());
    let prepared = analysis_prepare_structural_study_op(
        &spec,
        OperationContext::new(
            Some("trace-prepared".into()),
            Some("request-prepared".into()),
        ),
    )
    .expect("structural study preparation should succeed");
    let executor = CanonicalStructuralExecutor {
        calls: AtomicU64::new(0),
    };

    let runmat_analysis_fea::execution::StructuralStudyWorkloads::LinearSequence { workloads } =
        prepared.workloads()
    else {
        panic!("prepared static study is not a linear sequence");
    };
    let workload = workloads.into_values().next().unwrap();
    let numerical_result = executor
        .execute(&workload, prepared.solver_mesh())
        .expect("prepared numerical solve should succeed");
    let numerical_result =
        test_structural_sequence_result(&spec, &workload, prepared.solver_mesh(), numerical_result);
    let envelope = analysis_complete_structural_study_op(prepared, Ok(numerical_result))
        .expect("prepared structural study completion should succeed");

    assert_eq!(executor.calls.load(Ordering::Relaxed), 1);
    assert_eq!(envelope.trace_id.as_deref(), Some("trace-prepared"));
    assert_eq!(envelope.request_id.as_deref(), Some("request-prepared"));
    assert_eq!(envelope.data.study_id, spec.study_id);
    assert_eq!(
        envelope.data.steps[0].run_operation,
        "fea.run_linear_static"
    );
    assert!(storage::load_run_result(&envelope.data.steps[0].run_id)
        .expect("prepared run lookup should succeed")
        .is_some());
    assert!(Path::new(&envelope.data.evidence_artifact_path).is_file());
    let _ = fs::remove_dir_all(&root);
    let _ = fs::remove_dir_all(&solver_mesh_root);
}

#[test]
fn analysis_run_study_preserves_requested_canonical_solver_mesh_artifact() {
    let _guard = analysis_test_guard();
    storage::reset_artifact_store_for_tests();
    let root = temp_artifact_root("run-study-analysis-mesh");
    let _ = fs::remove_dir_all(&root);
    let _runtime_guard = scoped_study_artifact_root(&root);
    let mut spec = sample_linear_static_study_spec();
    spec.geometry = closed_cube_geometry_asset();
    let mut model = sample_solid_model();
    model.geometry_id = spec.geometry.geometry_id.clone();
    model.geometry_revision = spec.geometry.revision;
    model.units = runmat_types::physical::CoordinateUnits::canonical();
    model.material_assignments = vec![MaterialAssignment {
        target: runmat_analysis_core::PropertyRegionTarget::solid(
            runmat_meshing_core::fixtures::fixture_region("region"),
        ),
        expected_material_id: "mat_steel".to_string(),
        assigned_material_id: "mat_steel".to_string(),
        confidence: EvidenceConfidence::Verified,
    }];
    spec.execution.steps[0].step_id = model.sequence.steps[0].step_id.clone();
    spec.model = model;
    let (solver_mesh_root, solver_mesh_path) =
        write_ready_minimal_analysis_mesh_artifact("requested-canonical-solver-mesh");
    spec.solver_mesh_artifact_path = Some(solver_mesh_path.display().to_string());

    let envelope = analysis_run_study_op(&spec, OperationContext::new(None, None))
        .expect("study run should succeed");

    let artifact_path = envelope
        .data
        .solver_mesh_artifact_path
        .as_ref()
        .expect("study run should preserve solver mesh artifact path");
    assert_eq!(artifact_path, &solver_mesh_path.display().to_string());
    let decoded = runmat_meshing_core::SolverMeshArtifact::canonical_decode(
        &fs::read(artifact_path).expect("read canonical solver mesh artifact"),
    )
    .expect("decode canonical solver mesh artifact");
    assert_eq!(decoded.topology.volume_elements.len(), 1);
    let _ = fs::remove_dir_all(&solver_mesh_root);
}

#[test]
fn analysis_run_study_sweep_executes_multiple_studies() {
    let _guard = analysis_test_guard();
    storage::reset_artifact_store_for_tests();
    let root = temp_artifact_root("run-study-sweep-evidence");
    let _ = fs::remove_dir_all(&root);
    let env_guard = EnvVarRestoreGuard {
        key: "RUNMAT_FEA_STUDY_ARTIFACT_ROOT",
        previous: std::env::var("RUNMAT_FEA_STUDY_ARTIFACT_ROOT").ok(),
    };
    std::env::set_var("RUNMAT_FEA_STUDY_ARTIFACT_ROOT", root.display().to_string());
    let mut linear = sample_linear_static_study_spec();
    let (solver_mesh_root, solver_mesh_path) =
        write_ready_minimal_analysis_mesh_artifact("run-study-sweep-solver-mesh");
    linear.solver_mesh_artifact_path = Some(solver_mesh_path.display().to_string());
    let mut secondary = linear.clone();
    secondary.study_id = "study_secondary_structural".into();
    let sweep_spec = AnalysisStudySweepSpec {
        sweep_id: "study_sweep_001".to_string(),
        studies: vec![linear, secondary],
        fail_fast: true,
    };

    let envelope = analysis_run_study_sweep_op(&sweep_spec, OperationContext::new(None, None))
        .expect("study sweep should succeed");

    assert_eq!(envelope.operation, "fea.run_study_sweep");
    assert_eq!(envelope.op_version, "fea.run_study_sweep/v1");
    assert_eq!(envelope.data.sweep_id, "study_sweep_001");
    assert_eq!(envelope.data.study_count, 2);
    assert_eq!(envelope.data.success_count, 2);
    assert_eq!(envelope.data.failed_count, 0);
    assert!(envelope.data.failure_entries.is_empty());
    assert_eq!(envelope.data.run_entries.len(), 2);
    assert!(envelope
        .data
        .run_entries
        .iter()
        .any(
            |entry| entry.result.steps[0].procedure_step.family == PhysicsFamily::Structural
                && entry.result.steps[0].procedure_step.procedure == AnalysisProcedure::Static
        ));
    assert!(envelope
        .data
        .run_entries
        .iter()
        .any(
            |entry| entry.result.steps[0].procedure_step.family == PhysicsFamily::Structural
                && entry.result.steps[0].procedure_step.procedure == AnalysisProcedure::Static
        ));
    assert!(envelope
        .data
        .run_entries
        .iter()
        .all(|entry| entry.result.steps[0].run_id.starts_with("run_")));
    assert!(envelope.data.evidence_artifact_path.ends_with("run.json"));
    drop(env_guard);
    let _ = fs::remove_dir_all(&root);
    let _ = fs::remove_dir_all(&solver_mesh_root);
}

#[test]
fn analysis_run_study_sweep_rejects_empty_study_set() {
    let _guard = analysis_test_guard();
    let spec = AnalysisStudySweepSpec {
        sweep_id: "study_sweep_empty".to_string(),
        studies: Vec::new(),
        fail_fast: true,
    };

    let err = analysis_run_study_sweep_op(&spec, OperationContext::new(None, None))
        .expect_err("empty sweep should be rejected");
    assert_eq!(err.operation, "fea.run_study_sweep");
    assert_eq!(err.op_version, "fea.run_study_sweep/v1");
    assert_eq!(err.error_code, "RM.FEA.RUN_STUDY_SWEEP.INVALID_SPEC");
}

#[test]
fn analysis_complete_study_sweep_rejects_missing_and_substituted_outcomes() {
    let _guard = analysis_test_guard();
    storage::reset_artifact_store_for_tests();
    let context = OperationContext::new(None, None);
    let mut actual = sample_linear_static_study_spec();
    let (mesh_root, mesh_path) =
        write_ready_minimal_analysis_mesh_artifact("sweep-substitution-mesh");
    actual.solver_mesh_artifact_path = Some(mesh_path.display().to_string());
    let run = analysis_run_study_op(&actual, context.clone())
        .expect("structural study should produce a composable result");
    let mut expected = actual;
    expected.study_id = "different_authored_study".into();
    let spec = AnalysisStudySweepSpec {
        sweep_id: "substituted_sweep".into(),
        studies: vec![expected],
        fail_fast: false,
    };

    let missing = analysis_complete_study_sweep_op(&spec, context.clone(), Vec::new())
        .expect_err("missing sweep outcomes must fail closed");
    assert_eq!(
        missing.error_code,
        "RM.FEA.RUN_STUDY_SWEEP.OUTCOME_CARDINALITY"
    );

    let substituted = analysis_complete_study_sweep_op(
        &spec,
        context,
        vec![AnalysisStudySweepRunOutcome::Succeeded(Box::new(run))],
    )
    .expect_err("a result from a different authored study must fail closed");
    assert_eq!(
        substituted.error_code,
        "RM.FEA.RUN_STUDY_SWEEP.RESULT_SUBSTITUTED"
    );
    let _ = fs::remove_dir_all(mesh_root);
}

#[test]
fn analysis_complete_study_sweep_rejects_duplicate_logical_studies() {
    let _guard = analysis_test_guard();
    let study = sample_electromagnetic_study_spec();
    let spec = AnalysisStudySweepSpec {
        sweep_id: "duplicate_study_sweep".into(),
        studies: vec![study.clone(), study],
        fail_fast: false,
    };

    let error = analysis_run_study_sweep_op(&spec, OperationContext::new(None, None))
        .expect_err("duplicate logical studies must not receive a sweep identity");

    assert_eq!(error.error_code, "RM.FEA.RUN_STUDY_SWEEP.INVALID_SPEC");
    assert_eq!(
        error.context.get("issue_codes").map(String::as_str),
        Some("RM.FEA.STUDY_SWEEP.IDENTITY_INVALID")
    );
}

#[test]
fn analysis_study_sweep_identity_binds_authored_order_and_composition_policy() {
    let context = OperationContext::new(None, None);
    let first = sample_linear_static_study_spec();
    let second = sample_electromagnetic_study_spec();
    let collect_all = AnalysisStudySweepSpec {
        sweep_id: "identity_sweep".into(),
        studies: vec![first.clone(), second.clone()],
        fail_fast: false,
    };
    let reversed = AnalysisStudySweepSpec {
        studies: vec![second, first],
        ..collect_all.clone()
    };
    let fail_fast = AnalysisStudySweepSpec {
        fail_fast: true,
        ..collect_all.clone()
    };

    let collect_all_identity = analysis_study_sweep_identity(&collect_all, &context).unwrap();
    assert_ne!(
        collect_all_identity,
        analysis_study_sweep_identity(&reversed, &context).unwrap()
    );
    assert_ne!(
        collect_all_identity,
        analysis_study_sweep_identity(&fail_fast, &context).unwrap()
    );
}

#[test]
fn analysis_complete_study_sweep_finds_the_authored_fail_fast_error_before_not_run_entries() {
    let _guard = analysis_test_guard();
    let context = OperationContext::new(None, None);
    let first = sample_linear_static_study_spec();
    let mut invalid = sample_linear_static_study_spec();
    invalid.study_id = "   ".into();
    let error = analysis_run_study_op(&invalid, context.clone())
        .expect_err("invalid authored study must fail preparation");
    let spec = AnalysisStudySweepSpec {
        sweep_id: "prepared_fail_fast_sweep".into(),
        studies: vec![first, invalid],
        fail_fast: true,
    };

    let composed = analysis_complete_study_sweep_op(
        &spec,
        context,
        vec![
            AnalysisStudySweepRunOutcome::NotRun,
            AnalysisStudySweepRunOutcome::Failed(error),
        ],
    )
    .expect_err("the authored preparation failure must determine fail-fast completion");
    assert_eq!(composed.error_code, "RM.FEA.RUN_STUDY_SWEEP.STUDY_FAILED");
    assert_eq!(
        composed.context.get("study_index").map(String::as_str),
        Some("1")
    );
}

#[test]
fn analysis_complete_study_sweep_preserves_indeterminate_terminal_kind() {
    let _guard = analysis_test_guard();
    let context = OperationContext::new(None, None);
    let study = sample_linear_static_study_spec();
    let mut invalid = study.clone();
    invalid.study_id = "   ".into();
    let error = analysis_run_study_op(&invalid, context.clone())
        .expect_err("invalid authored study must provide a representative operation error");
    let spec = AnalysisStudySweepSpec {
        sweep_id: "indeterminate_sweep".into(),
        studies: vec![study],
        fail_fast: false,
    };

    let composed = analysis_complete_study_sweep_op(
        &spec,
        context.clone(),
        vec![AnalysisStudySweepRunOutcome::Indeterminate(error.clone())],
    )
    .expect("non-fail-fast sweep must publish indeterminate evidence");

    assert_eq!(composed.data.failure_entries.len(), 1);
    assert_eq!(
        composed.data.failure_entries[0].kind,
        AnalysisStudySweepFailureKind::Indeterminate
    );

    let fail_fast = AnalysisStudySweepSpec {
        sweep_id: "indeterminate_fail_fast_sweep".into(),
        studies: spec.studies,
        fail_fast: true,
    };
    let terminal = analysis_complete_study_sweep_op(
        &fail_fast,
        context,
        vec![AnalysisStudySweepRunOutcome::Indeterminate(error)],
    )
    .expect_err("fail-fast sweep must preserve an indeterminate terminal");
    assert_eq!(
        terminal.error_code,
        "RM.FEA.RUN_STUDY_SWEEP.STUDY_INDETERMINATE"
    );
    assert_eq!(terminal.error_type, OperationErrorType::Validation);
    assert_eq!(
        terminal.context.get("failure_kind").map(String::as_str),
        Some("indeterminate")
    );
}

#[test]
fn analysis_run_study_sweep_fail_fast_returns_error_on_invalid_study() {
    let _guard = analysis_test_guard();
    let mut invalid = sample_linear_static_study_spec();
    invalid.study_id = "   ".to_string();
    let spec = AnalysisStudySweepSpec {
        sweep_id: "study_sweep_fail_fast".to_string(),
        studies: vec![sample_linear_static_study_spec(), invalid],
        fail_fast: true,
    };

    let err = analysis_run_study_sweep_op(&spec, OperationContext::new(None, None))
        .expect_err("fail-fast sweep should return error");
    assert_eq!(err.operation, "fea.run_study_sweep");
    assert_eq!(err.op_version, "fea.run_study_sweep/v1");
    assert_eq!(err.error_code, "RM.FEA.RUN_STUDY_SWEEP.STUDY_FAILED");
}

#[test]
fn analysis_run_study_sweep_can_continue_on_study_failure() {
    let _guard = analysis_test_guard();
    storage::reset_artifact_store_for_tests();
    let mut invalid = sample_linear_static_study_spec();
    invalid.study_id = "   ".to_string();
    let mut valid = sample_linear_static_study_spec();
    let (solver_mesh_root, solver_mesh_path) =
        write_ready_minimal_analysis_mesh_artifact("run-study-sweep-continue-solver-mesh");
    valid.solver_mesh_artifact_path = Some(solver_mesh_path.display().to_string());
    let spec = AnalysisStudySweepSpec {
        sweep_id: "study_sweep_continue".to_string(),
        studies: vec![valid, invalid],
        fail_fast: false,
    };

    let envelope = analysis_run_study_sweep_op(&spec, OperationContext::new(None, None))
        .expect("continue-on-failure sweep should succeed");

    assert_eq!(envelope.data.study_count, 2);
    assert_eq!(envelope.data.success_count, 1);
    assert_eq!(envelope.data.failed_count, 1);
    assert_eq!(envelope.data.run_entries.len(), 1);
    assert_eq!(envelope.data.failure_entries.len(), 1);
    assert_eq!(envelope.data.failure_entries[0].study_index, 1);
    assert_eq!(
        envelope.data.failure_entries[0].error_code,
        "RM.FEA.RUN_STUDY.INVALID_SPEC"
    );
    assert!(envelope.data.failure_entries[0].study_id.trim().is_empty());
    let _ = fs::remove_dir_all(&solver_mesh_root);
}

#[test]
fn analysis_validate_study_sweep_reports_valid_entries_and_persists_artifact() {
    let _guard = analysis_test_guard();
    storage::reset_artifact_store_for_tests();
    let root = temp_artifact_root("validate-study-sweep-evidence");
    let _ = fs::remove_dir_all(&root);
    let env_guard = EnvVarRestoreGuard {
        key: "RUNMAT_FEA_STUDY_ARTIFACT_ROOT",
        previous: std::env::var("RUNMAT_FEA_STUDY_ARTIFACT_ROOT").ok(),
    };
    std::env::set_var("RUNMAT_FEA_STUDY_ARTIFACT_ROOT", root.display().to_string());
    let spec = AnalysisStudySweepSpec {
        sweep_id: "study_sweep_validate_001".to_string(),
        studies: vec![
            sample_linear_static_study_spec(),
            sample_electromagnetic_study_spec(),
        ],
        fail_fast: true,
    };

    let envelope = analysis_validate_study_sweep_op(&spec, OperationContext::new(None, None))
        .expect("study sweep validation should succeed");

    assert_eq!(envelope.operation, "fea.validate_study_sweep");
    assert_eq!(envelope.op_version, "fea.validate_study_sweep/v1");
    assert_eq!(envelope.data.sweep_id, "study_sweep_validate_001");
    assert!(envelope.data.valid);
    assert!(envelope.data.issue_codes.is_empty());
    assert_eq!(envelope.data.study_entries.len(), 2);
    assert!(envelope.data.study_entries.iter().all(|entry| entry.valid));
    assert!(envelope
        .data
        .study_entries
        .iter()
        .all(|entry| entry.issue_codes.is_empty() && entry.issues.is_empty()));
    assert!(envelope
        .data
        .evidence_artifact_path
        .ends_with("validate.json"));
    assert!(PathBuf::from(&envelope.data.evidence_artifact_path).exists());

    drop(env_guard);
    let _ = fs::remove_dir_all(&root);
}

#[test]
fn analysis_validate_study_sweep_reports_sweep_and_study_issue_details() {
    let _guard = analysis_test_guard();
    let mut invalid = sample_linear_static_study_spec();
    invalid.study_id = "   ".to_string();
    let spec = AnalysisStudySweepSpec {
        sweep_id: "   ".to_string(),
        studies: vec![invalid],
        fail_fast: true,
    };

    let envelope = analysis_validate_study_sweep_op(&spec, OperationContext::new(None, None))
        .expect("study sweep validation should return a typed payload");

    assert_eq!(envelope.operation, "fea.validate_study_sweep");
    assert_eq!(envelope.op_version, "fea.validate_study_sweep/v1");
    assert!(!envelope.data.valid);
    assert_eq!(
        envelope.data.issue_codes,
        vec!["RM.FEA.STUDY_SWEEP.ID_EMPTY".to_string()]
    );
    assert_eq!(envelope.data.study_entries.len(), 1);
    assert!(!envelope.data.study_entries[0].valid);
    assert!(envelope.data.study_entries[0]
        .issue_codes
        .iter()
        .any(|code| code == "RM.FEA.STUDY.ID_EMPTY"));
    assert!(envelope.data.study_entries[0]
        .issues
        .iter()
        .any(|issue| issue.code == "RM.FEA.STUDY.ID_EMPTY" && !issue.message.is_empty()));
}

#[test]
fn analysis_validate_returns_typed_envelope() {
    let _guard = analysis_test_guard();
    let model = sample_model();
    let context =
        OperationContext::new(Some("trace-a1".to_string()), Some("request-a1".to_string()));
    let envelope = analysis_validate(&model, &CoordinateFrame::Global, context)
        .expect("validation should pass");

    assert_eq!(envelope.operation, "fea.validate");
    assert_eq!(envelope.op_version, "fea.validate/v1");
    assert!(envelope.data.valid);
    assert_eq!(envelope.trace_id.as_deref(), Some("trace-a1"));
}

#[test]
fn analysis_validate_maps_typed_error_code() {
    let _guard = analysis_test_guard();
    let mut model = sample_model();
    model.materials.clear();
    let context = OperationContext::new(None, None);
    let error = analysis_validate(&model, &CoordinateFrame::Global, context)
        .expect_err("validation should fail");

    assert_eq!(error.error_code, "RM.FEA.VALIDATE.MISSING_MATERIALS");
    assert_eq!(error.operation, "fea.validate");
    assert_eq!(error.op_version, "fea.validate/v1");
}

#[test]
fn analysis_validate_maps_invalid_moment_signal_error_code() {
    let _guard = analysis_test_guard();
    let mut model = sample_model();
    replace_with_moment_excitation(&mut model, "load_tip");
    let PhysicalSignal::Moment(signal) = &mut model.signals[0] else {
        panic!("expected moment signal");
    };
    signal.representation = SignalRepresentation::Constant(SignalValue::Vector3([
        runmat_analysis_core::Moment::new(f64::INFINITY),
        runmat_analysis_core::Moment::new(0.0),
        runmat_analysis_core::Moment::new(1.0),
    ]));
    let context = OperationContext::new(None, None);
    let error = analysis_validate(&model, &CoordinateFrame::Global, context)
        .expect_err("validation should fail");

    assert_eq!(error.error_code, "RM.FEA.VALIDATE.INVALID_SIGNAL");
    assert_eq!(
        error.context.get("signal_id").map(String::as_str),
        Some("load_tip.signal")
    );
}

#[test]
fn analysis_run_linear_static_returns_typed_envelope() {
    let _guard = analysis_test_guard();
    let model = sample_model();
    let context =
        OperationContext::new(Some("trace-a2".to_string()), Some("request-a2".to_string()));
    let envelope = analysis_run_linear_static_with_options(
        &model,
        ComputeBackend::Cpu,
        AnalysisRunOptions {
            deterministic_mode: true,
            precision_mode: PrecisionMode::Fp64,
            preconditioner_mode: PreconditionerMode::Auto,
            quality_policy: QualityPolicy::Balanced,
            solver_mesh_artifact_path: None,
        },
        context,
    )
    .expect("run should pass");

    assert_eq!(envelope.operation, "fea.run_linear_static");
    assert_eq!(envelope.op_version, "fea.run_linear_static/v1");
    assert_eq!(envelope.data.run.requested_backend, ComputeBackend::Cpu);
    assert!(!envelope
        .data
        .run
        .field(FEA_FIELD_STRUCTURAL_DISPLACEMENT)
        .expect("structural displacement field should be present")
        .is_empty());
    assert_eq!(envelope.data.run_status, RunStatus::Completed);
    assert!(envelope.data.numerically_acceptable);
    assert_eq!(envelope.data.result_quality, QualityGate::Pass);
    assert!(envelope.data.quality_reasons.is_empty());
    assert!(envelope.data.modal_results.is_none());
    assert_eq!(envelope.data.solver_convergence, QualityGate::Pass);
    assert!(envelope.data.provenance.deterministic_mode);
    assert_eq!(envelope.data.provenance.precision_mode, "fp64");
    assert_eq!(
        envelope.data.provenance.solver_method,
        SolverMethod::SparseDirectCholesky
    );
    assert_eq!(
        envelope.data.provenance.preconditioner,
        SolverPreconditioner::None
    );
    assert!(envelope.data.provenance.fallback_events.is_empty());
    assert_eq!(envelope.data.provenance.quality_policy, "balanced");
    assert_eq!(envelope.data.provenance.solver_device_apply_k_ratio, 0.0);
    assert_eq!(envelope.data.provenance.solver_host_sync_count, 0);
}

#[test]
fn analysis_run_linear_static_with_thermo_mechanical_coupling_reports_fields() {
    let _guard = analysis_test_guard();
    let (mesh_root, mesh_path) = write_ready_minimal_analysis_mesh_artifact("linear-thermo-fields");
    let mut model = sample_solid_model();
    model.sequence.steps = vec![ProcedureStep::coupled(
        "linear_thermo_mechanical",
        CoupledPhysicsFamily::Thermomechanical,
        AnalysisProcedure::Static,
    )];
    set_model_thermo_coupling(
        &mut model,
        TestThermoMechanicalCoupling {
            reference_temperature_k: runmat_types::physical::Temperature::new(293.15),
            applied_temperature_delta_k: runmat_types::physical::TemperatureDifference::new(65.0),
            thermal_expansion_coefficient: runmat_types::physical::ThermalExpansion::new(1.2e-5),
            region_temperature_deltas: Vec::new(),
        },
    );

    let envelope = analysis_run_linear_static_with_options(
        &model,
        ComputeBackend::Cpu,
        AnalysisRunOptions {
            deterministic_mode: true,
            precision_mode: PrecisionMode::Fp64,
            preconditioner_mode: PreconditionerMode::Auto,
            quality_policy: QualityPolicy::Balanced,
            solver_mesh_artifact_path: Some(mesh_path.display().to_string()),
        },
        OperationContext::new(Some("trace-linear-thermo-fields".to_string()), None),
    )
    .expect("thermo-mechanical linear static run should pass");

    assert_eq!(
        envelope.data.run.identity.family(),
        PhysicsFamily::Coupled(CoupledPhysicsFamily::Thermomechanical)
    );

    assert!(envelope
        .data
        .run
        .field(&fea_thermo_mechanical_temperature_field_id(0))
        .is_some());
    assert!(envelope
        .data
        .run
        .field(&fea_thermo_mechanical_thermal_strain_field_id(0))
        .is_some());
    assert!(envelope
        .data
        .run
        .field(&fea_thermo_mechanical_thermal_stress_field_id(0))
        .is_some());
    assert!(envelope
        .data
        .run
        .field(&fea_thermo_mechanical_displacement_field_id(0))
        .is_some());
    assert!(envelope
        .data
        .run
        .field(&fea_thermo_mechanical_von_mises_field_id(0))
        .is_some());
    assert!(envelope
        .data
        .run
        .field(&fea_thermo_mechanical_coupling_residual_field_id(0))
        .is_some());

    let results = analysis_results_op(
        &envelope.data,
        AnalysisResultsQuery::default(),
        OperationContext::new(None, None),
    )
    .expect("thermo-mechanical linear static results should be queryable");
    let descriptor = |field_id: &str| {
        results
            .data
            .field_descriptors
            .iter()
            .find(|descriptor| descriptor.field_id == field_id)
            .expect("thermo-mechanical descriptor should be present")
    };
    for field_id in [
        fea_thermo_mechanical_temperature_field_id(0),
        fea_thermo_mechanical_von_mises_field_id(0),
        fea_thermo_mechanical_coupling_residual_field_id(0),
    ] {
        let descriptor = descriptor(&field_id);
        assert_eq!(descriptor.kind, AnalysisFieldKind::Scalar);
        assert_eq!(descriptor.component_count, None);
    }
    let displacement_descriptor = descriptor(&fea_thermo_mechanical_displacement_field_id(0));
    assert_eq!(displacement_descriptor.kind, AnalysisFieldKind::Vector);
    assert_eq!(displacement_descriptor.component_count, Some(3));
    for field_id in [
        fea_thermo_mechanical_thermal_strain_field_id(0),
        fea_thermo_mechanical_thermal_stress_field_id(0),
    ] {
        let descriptor = descriptor(&field_id);
        assert_eq!(descriptor.kind, AnalysisFieldKind::Tensor);
        assert_eq!(descriptor.component_count, Some(6));
    }
    let _ = fs::remove_dir_all(&mesh_root);
}

#[test]
fn analysis_run_linear_static_rejects_thermo_coupling_on_structural_step() {
    let _guard = analysis_test_guard();
    let (mesh_root, mesh_path) =
        write_ready_minimal_analysis_mesh_artifact("linear-structural-step-thermo-context");
    let mut model = sample_solid_model();
    set_model_thermo_coupling(
        &mut model,
        TestThermoMechanicalCoupling {
            reference_temperature_k: runmat_types::physical::Temperature::new(293.15),
            applied_temperature_delta_k: runmat_types::physical::TemperatureDifference::new(65.0),
            thermal_expansion_coefficient: runmat_types::physical::ThermalExpansion::new(1.2e-5),
            region_temperature_deltas: Vec::new(),
        },
    );

    let error = analysis_run_linear_static_with_options(
        &model,
        ComputeBackend::Cpu,
        AnalysisRunOptions {
            solver_mesh_artifact_path: Some(mesh_path.display().to_string()),
            ..AnalysisRunOptions::default()
        },
        OperationContext::new(None, None),
    )
    .expect_err("a structural step must not silently execute coupled assembly math");

    assert_eq!(
        error.error_code,
        "RM.FEA.RUN_LINEAR_STATIC.SOLVER_MODEL_INVALID"
    );
    assert!(error.message.contains("requests Structural"));
    assert!(error.message.contains("realizes Coupled(Thermomechanical)"));
    let _ = fs::remove_dir_all(&mesh_root);
}

#[test]
fn analysis_run_linear_static_persists_artifacts_through_runtime_filesystem_provider() {
    let _guard = analysis_test_guard();
    let _provider_lock = runmat_filesystem::provider_override_lock();
    storage::reset_artifact_store_for_tests();
    let sandbox_root = temp_artifact_root("artifact-provider");
    let _ = fs::remove_dir_all(&sandbox_root);
    let provider = Arc::new(
        runmat_filesystem::SandboxFsProvider::new(sandbox_root.clone())
            .expect("sandbox filesystem provider should be created"),
    );
    let _provider_guard = runmat_filesystem::replace_provider(provider);
    storage::configure_artifact_store(storage::AnalysisArtifactStoreConfig::Filesystem {
        root: PathBuf::from("/fea-artifacts"),
    })
    .expect("configure provider-backed filesystem artifact store");

    let model = sample_model();
    let envelope = analysis_run_linear_static_with_options(
        &model,
        ComputeBackend::Cpu,
        Default::default(),
        OperationContext::new(Some("trace-provider-artifact".to_string()), None),
    )
    .expect("run should pass");
    let artifact_path = PathBuf::from("/fea-artifacts")
        .join("runs")
        .join(format!("{}.json", envelope.data.run_id));

    let bytes =
        runmat_filesystem::read(&artifact_path).expect("artifact should be provider-readable");
    assert!(!bytes.is_empty());
    let persisted = storage::load_run_result(&envelope.data.run_id)
        .expect("provider-backed artifact load should succeed")
        .expect("run artifact should exist");
    assert_eq!(persisted.run_id, envelope.data.run_id);
    assert!(sandbox_root
        .join("fea-artifacts")
        .join("runs")
        .join(format!("{}.json", envelope.data.run_id))
        .exists());

    storage::reset_artifact_store_for_tests();
    let _ = fs::remove_dir_all(&sandbox_root);
}

#[test]
fn analysis_run_linear_static_cancellation_maps_normalized_error() {
    let _guard = analysis_test_guard();
    storage::reset_artifact_store_for_tests();
    let cancelled = Arc::new(AtomicBool::new(true));
    let _interrupt_guard = crate::interrupt::replace_interrupt(Some(
        runmat_execution::CancellationToken::from_shared(cancelled),
    ));
    let model = sample_model();

    let err = analysis_run_linear_static_with_options(
        &model,
        ComputeBackend::Cpu,
        Default::default(),
        OperationContext::new(Some("trace-cancel".to_string()), None),
    )
    .expect_err("cancelled run should fail");

    assert_eq!(err.error_code, "RM.FEA.RUN_LINEAR_STATIC.CANCELLED");
    assert_eq!(err.error_type, OperationErrorType::Cancelled);
    assert_eq!(err.severity, OperationErrorSeverity::Warning);
    assert!(!err.retryable);
    storage::reset_artifact_store_for_tests();
}

#[test]
fn analysis_run_linear_static_cancellation_after_preparation_stops_assembly() {
    let _guard = analysis_test_guard();
    let cancelled = Arc::new(AtomicBool::new(false));
    let _interrupt_guard = crate::interrupt::replace_interrupt(Some(
        runmat_execution::CancellationToken::from_shared(Arc::clone(&cancelled)),
    ));
    let cancellation_on_assembly = Arc::clone(&cancelled);
    let _progress_guard = replace_fea_progress_handler(Some(Arc::new(move |event| {
        if event.phase == FeaProgressPhase::ModelAssembly
            && event.status == FeaProgressStatus::Started
        {
            cancellation_on_assembly.store(true, Ordering::Relaxed);
        }
    })));

    let error = analysis_run_linear_static_with_options(
        &sample_model(),
        ComputeBackend::Cpu,
        AnalysisRunOptions::default(),
        OperationContext::new(Some("trace-cancel-after-preparation".to_string()), None),
    )
    .expect_err("cancellation at the assembly boundary must stop the inline solve");

    assert_eq!(error.error_code, "RM.FEA.RUN_LINEAR_STATIC.CANCELLED");
    assert_eq!(error.error_type, OperationErrorType::Cancelled);
    assert!(!error.retryable);
}

#[test]
fn analysis_run_linear_static_emits_solver_and_artifact_progress_events() {
    let _guard = analysis_test_guard();
    storage::reset_artifact_store_for_tests();
    let events = Arc::new(Mutex::new(Vec::new()));
    let events_for_handler = Arc::clone(&events);
    let _progress_guard = replace_fea_progress_handler(Some(Arc::new(move |event| {
        events_for_handler
            .lock()
            .expect("progress event lock should not be poisoned")
            .push(event);
    })));
    let model = sample_model();

    analysis_run_linear_static_with_options(
        &model,
        ComputeBackend::Cpu,
        Default::default(),
        OperationContext::new(Some("trace-progress".to_string()), None),
    )
    .expect("run should pass");

    let events = events
        .lock()
        .expect("progress event lock should not be poisoned");
    assert!(events.iter().any(|event| {
        event.operation == "fea.run_linear_static"
            && event.phase == FeaProgressPhase::Solve
            && event.status == FeaProgressStatus::Started
    }));
    assert!(events.iter().any(|event| {
        event.operation == "fea.run_linear_static"
            && event.phase == FeaProgressPhase::ArtifactPersistence
            && event.status == FeaProgressStatus::Completed
    }));
    assert!(events.iter().any(|event| {
        event.operation == "fea.run_linear_static"
            && event.phase == FeaProgressPhase::Complete
            && event.status == FeaProgressStatus::Completed
    }));

    storage::reset_artifact_store_for_tests();
}

#[test]
fn host_progress_handler_observes_direct_solver_progress() {
    let _guard = analysis_test_guard();
    let events = Arc::new(Mutex::new(Vec::new()));
    let events_for_handler = Arc::clone(&events);
    let _progress_guard = replace_fea_progress_handler(Some(Arc::new(move |event| {
        events_for_handler
            .lock()
            .expect("progress event lock should not be poisoned")
            .push(event);
    })));

    runmat_analysis_fea::emit_fea_progress_phase(
        "fea.run_checked_study",
        FeaProgressPhase::MeshPrep,
        FeaProgressStatus::Advanced,
        "Generated 42 elements",
        Some(42),
        Some(100),
    );

    let events = events
        .lock()
        .expect("progress event lock should not be poisoned");
    assert_eq!(events.len(), 1);
    assert_eq!(events[0].operation, "fea.run_checked_study");
    assert_eq!(events[0].phase, FeaProgressPhase::MeshPrep);
    assert_eq!(events[0].fraction, Some(0.42));
}

#[test]
fn native_sparse_fea_cancellation_is_terminal_and_releases_resources() {
    let _guard = analysis_test_guard();
    let provider: &'static runmat_accelerate::simple_provider::InProcessProvider = Box::leak(
        Box::new(runmat_accelerate::simple_provider::InProcessProvider::new()),
    );
    let _provider_guard = runmat_accelerate_api::ThreadProviderGuard::set(Some(provider));
    let cancellation = runmat_analysis_fea::FeaCancellationToken::new();
    cancellation.cancel();
    let _progress_guard = runmat_analysis_fea::replace_fea_progress_context(
        runmat_analysis_fea::FeaProgressContext {
            handler: None,
            cancellation,
            attempt_id: Some(runmat_execution::identity::AttemptId::derive(&[
                b"cancelled-fea-solve",
            ])),
        },
    );
    let dimension = 32;
    let stiffness = runmat_analysis_fea::algebra::RealSparseMatrix::diagonal(
        vec![2.0; dimension],
        runmat_analysis_fea::algebra::OperatorRole::Stiffness,
        runmat_analysis_fea::algebra::MatrixProperties::SYMMETRIC_POSITIVE_DEFINITE,
    )
    .expect("stiffness");
    let mass = runmat_analysis_fea::algebra::RealSparseMatrix::diagonal(
        vec![1.0; dimension],
        runmat_analysis_fea::algebra::OperatorRole::Mass,
        runmat_analysis_fea::algebra::MatrixProperties::SYMMETRIC_POSITIVE_DEFINITE,
    )
    .expect("mass");
    let damping = runmat_analysis_fea::algebra::RealSparseMatrix::diagonal(
        vec![0.0; dimension],
        runmat_analysis_fea::algebra::OperatorRole::Damping,
        runmat_analysis_fea::algebra::MatrixProperties::SYMMETRIC_POSITIVE_SEMIDEFINITE,
    )
    .expect("damping");
    let operators =
        runmat_analysis_fea::algebra::CanonicalRealOperators::try_new(stiffness, mass, damping)
            .expect("operators");
    let system = runmat_analysis_fea::algebra::AssembledRealSystem::try_new(
        operators,
        vec![false; dimension],
        vec![1.0; dimension],
    )
    .expect("system");

    let result = runmat_analysis_fea::solve::linear::solve_operator_system(
        &system,
        runmat_analysis_fea::solve::preconditioner::SpdPreconditionerKind::Jacobi,
        LinearAlgebraBackendKind::Accelerator,
        &runmat_analysis_fea::solve::backend::cpu_reference::CpuReferenceBackend,
        None,
    )
    .expect("cancelled coercivity fixture returns a diagnostic outcome");

    assert!(!result.converged);
    assert!(result.solution.iter().all(|value| *value == 0.0));
    // The coercivity proof runs on the host before any provider dispatch and observes the
    // cancellation first: the solve is terminal, the provider is never engaged, and no
    // provider evidence or resources exist for it.
    assert!(
        result
            .diagnostics
            .iter()
            .any(|diagnostic| diagnostic.code == "FEA_SOLVER_CANCELLED"),
        "diagnostics={:?}",
        result.diagnostics
    );
    assert!(result.accelerator_evidence.is_none());
    assert_eq!(result.host_sync_count, 0);
    assert_eq!(result.device_apply_k_count, 0);
    assert_eq!(provider.placement_resources().live_bytes, 0);
}

#[test]
fn native_sparse_provider_preserves_modal_fallback_and_transient_device_evidence() {
    let _guard = analysis_test_guard();
    let provider: &'static runmat_accelerate::simple_provider::InProcessProvider = Box::leak(
        Box::new(runmat_accelerate::simple_provider::InProcessProvider::new()),
    );
    let _provider_guard = runmat_accelerate_api::ThreadProviderGuard::set(Some(provider));

    let mut modal_model = sample_model();
    modal_model.sequence.steps = vec![ProcedureStep::structural_modal("modal-native-sparse")];
    let modal = analysis_run_modal_with_options_op(
        &modal_model,
        ComputeBackend::Gpu,
        Default::default(),
        OperationContext::new(Some("trace-modal-native-sparse".to_string()), None),
    )
    .expect("native sparse modal solve");
    assert_eq!(
        modal.data.run.solver_backend,
        LinearAlgebraBackendKind::CpuReference
    );
    assert!(modal.data.run.accelerator_evidence.is_empty());
    assert_eq!(
        modal
            .data
            .run
            .diagnostics
            .iter()
            .filter(|diagnostic| diagnostic.code
                == runmat_analysis_fea::accelerator_admission::ACCELERATOR_ADMISSION_FALLBACK_CODE)
            .count(),
        1
    );
    assert_eq!(provider.placement_resources().live_bytes, 0);

    let mut transient_model = sample_model();
    transient_model.sequence.steps = vec![ProcedureStep::structural_transient(
        "transient-native-sparse",
    )];
    let transient = analysis_run_transient_with_options_op(
        &transient_model,
        ComputeBackend::Gpu,
        Default::default(),
        OperationContext::new(Some("trace-transient-native-sparse".to_string()), None),
    )
    .expect("native sparse transient solve");
    assert_eq!(
        transient.data.run.solver_backend,
        LinearAlgebraBackendKind::Accelerator
    );
    assert!(!transient.data.run.accelerator_evidence.is_empty());
    assert!(transient
        .data
        .run
        .accelerator_evidence
        .iter()
        .all(|evidence| evidence.validate().is_ok() && evidence.satisfies_tolerance()));
    assert_eq!(provider.placement_resources().live_bytes, 0);
}

#[test]
fn analysis_results_returns_filtered_fields_and_metadata() {
    let _guard = analysis_test_guard();
    let model = sample_model();
    let run = analysis_run_linear_static_with_options(
        &model,
        ComputeBackend::Cpu,
        Default::default(),
        OperationContext::new(Some("trace-results-1".to_string()), None),
    )
    .expect("run should pass");

    let results = analysis_results_op(
        &run.data,
        AnalysisResultsQuery {
            include_fields: vec![FEA_FIELD_STRUCTURAL_DISPLACEMENT.to_string()],
            include_field_values: true,

            include_diagnostics: false,
            diagnostic_codes: Vec::new(),
            include_modal_results: true,
            include_buckling_results: true,
            mode_indices: Vec::new(),
            include_transient_results: true,
            transient_snapshot_indices: Vec::new(),
            include_nonlinear_results: true,
        },
        OperationContext::new(Some("trace-results-2".to_string()), None),
    )
    .expect("results should pass");

    assert_eq!(results.operation, "fea.results");
    assert_eq!(results.op_version, "fea.results/v1");
    assert_eq!(results.data.fields.len(), 1);
    assert_eq!(
        results.data.fields[0].field_id,
        FEA_FIELD_STRUCTURAL_DISPLACEMENT
    );
    assert!(results.data.diagnostics.is_none());
    assert_eq!(results.data.summary.field_count, 1);
    assert_eq!(results.data.summary.mode_count, 0);
    assert!(results.data.summary.available_mode_indices.is_empty());
    assert_eq!(results.data.summary.min_frequency_hz, None);
    assert_eq!(results.data.summary.max_frequency_hz, None);
    assert_eq!(results.data.summary.max_modal_residual_norm, None);
    assert_eq!(results.data.summary.first_mode_converged, None);
    assert_eq!(results.data.summary.snapshot_count, 0);
    assert_eq!(results.data.summary.time_start_s, None);
    assert_eq!(results.data.summary.time_end_s, None);
    assert_eq!(results.data.summary.max_transient_residual_norm, None);
    assert_eq!(results.data.summary.final_step_converged, None);
}

#[test]
fn analysis_field_descriptors_include_physics_units_and_locations() {
    let cases = vec![
        (
            FEA_FIELD_STRUCTURAL_DISPLACEMENT.to_string(),
            "structural",
            "displacement",
            Some("m"),
            AnalysisFieldLocation::Node,
        ),
        (
            FEA_FIELD_STRUCTURAL_STRESS.to_string(),
            "structural",
            "stress",
            Some("Pa"),
            AnalysisFieldLocation::Element,
        ),
        (
            FEA_FIELD_STRUCTURAL_STRAIN_ENERGY_DENSITY.to_string(),
            "structural",
            "strain_energy_density",
            Some("J/m^3"),
            AnalysisFieldLocation::Element,
        ),
        (
            FEA_FIELD_MODAL_FREQUENCY_HZ.to_string(),
            "modal",
            "frequency_hz",
            Some("Hz"),
            AnalysisFieldLocation::Mode,
        ),
        (
            fea_modal_mode_shape_field_id(1),
            "modal",
            "mode_shape",
            Some("m"),
            AnalysisFieldLocation::Node,
        ),
        (
            FEA_FIELD_ACOUSTIC_PRESSURE_REAL.to_string(),
            "acoustic",
            "pressure_real",
            Some("Pa"),
            AnalysisFieldLocation::Node,
        ),
        (
            FEA_FIELD_CFD_WALL_SHEAR_STRESS.to_string(),
            "cfd",
            "wall_shear_stress",
            Some("Pa"),
            AnalysisFieldLocation::BoundaryFace,
        ),
        (
            fea_thermal_temperature_gradient_field_id(0),
            "thermal",
            "temperature_gradient",
            Some("K/m"),
            AnalysisFieldLocation::Element,
        ),
        (
            FEA_FIELD_ELECTRO_THERMAL_JOULE_HEAT.to_string(),
            "electro_thermal",
            "joule_heat",
            Some("W/m^3"),
            AnalysisFieldLocation::Element,
        ),
        (
            fea_nonlinear_contact_pressure_field_id(0),
            "nonlinear",
            "contact_pressure",
            Some("Pa"),
            AnalysisFieldLocation::InterfaceFace,
        ),
        (
            fea_cht_interface_heat_flux_field_id(0),
            "cht",
            "interface_heat_flux",
            Some("W/m^2"),
            AnalysisFieldLocation::InterfaceFace,
        ),
        (
            fea_fsi_interface_traction_field_id(0),
            "fsi",
            "interface_traction",
            Some("Pa"),
            AnalysisFieldLocation::InterfaceFace,
        ),
        (
            fea_thermo_mechanical_thermal_strain_field_id(0),
            "thermo_mechanical",
            "thermal_strain",
            Some("1"),
            AnalysisFieldLocation::Element,
        ),
        (
            FEA_FIELD_STRUCTURAL_SHELL_MEMBRANE_FORCE.to_string(),
            "structural",
            "shell_membrane_force",
            Some("N"),
            AnalysisFieldLocation::Element,
        ),
        (
            FEA_FIELD_STRUCTURAL_SHELL_BENDING_MOMENT.to_string(),
            "structural",
            "shell_bending_moment",
            Some("N*m"),
            AnalysisFieldLocation::Element,
        ),
        (
            FEA_FIELD_STRUCTURAL_SHELL_TRANSVERSE_SHEAR.to_string(),
            "structural",
            "shell_transverse_shear",
            Some("N"),
            AnalysisFieldLocation::Element,
        ),
        (
            FEA_FIELD_STRUCTURAL_SHELL_VON_MISES.to_string(),
            "structural",
            "shell_von_mises",
            Some("Pa"),
            AnalysisFieldLocation::Element,
        ),
    ];

    // Family and unit follow the field id; location and quantity are the publisher's
    // declared layout (see `field_descriptors_project_the_declared_layout`).
    for (field_id, family, _published_name, unit, _location) in cases {
        let descriptor = AnalysisFieldDescriptor::from_field(&AnalysisField::host_f64(
            &field_id,
            runmat_analysis_core::AnalysisFieldLayout::scalar(
                runmat_analysis_core::ObservationQuantity::Residual,
                runmat_analysis_core::FieldLocation::Global,
            ),
            vec![1],
            vec![0.0],
        ));
        assert_eq!(descriptor.family, family, "{field_id} family");
        assert_eq!(descriptor.unit.as_deref(), unit, "{field_id} unit");
    }

    let membrane = AnalysisFieldDescriptor::from_field(&AnalysisField::host_f64(
        FEA_FIELD_STRUCTURAL_SHELL_MEMBRANE_FORCE,
        runmat_analysis_core::AnalysisFieldLayout::canonical(
            runmat_analysis_core::ObservationQuantity::ShellMembraneForce,
            runmat_analysis_core::FieldLocation::Element,
        ),
        vec![1, 3],
        vec![0.0, 0.0, 0.0],
    ));
    assert_eq!(membrane.kind, AnalysisFieldKind::Vector);
    assert_eq!(membrane.component_count, Some(3));
    let shear = AnalysisFieldDescriptor::from_field(&AnalysisField::host_f64(
        FEA_FIELD_STRUCTURAL_SHELL_TRANSVERSE_SHEAR,
        runmat_analysis_core::AnalysisFieldLayout::canonical(
            runmat_analysis_core::ObservationQuantity::ShellTransverseShear,
            runmat_analysis_core::FieldLocation::Element,
        ),
        vec![1, 2],
        vec![0.0, 0.0],
    ));
    assert_eq!(shear.kind, AnalysisFieldKind::Vector);
    assert_eq!(shear.component_count, Some(2));
    let von_mises = AnalysisFieldDescriptor::from_field(&AnalysisField::host_f64(
        FEA_FIELD_STRUCTURAL_SHELL_VON_MISES,
        runmat_analysis_core::AnalysisFieldLayout::scalar(
            runmat_analysis_core::ObservationQuantity::ShellVonMises,
            runmat_analysis_core::FieldLocation::Element,
        ),
        vec![1],
        vec![0.0],
    ));
    assert_eq!(von_mises.kind, AnalysisFieldKind::Scalar);
    assert_eq!(von_mises.component_count, Some(1));

    let displacement = AnalysisFieldDescriptor::from_field(&AnalysisField::host_f64(
        FEA_FIELD_STRUCTURAL_DISPLACEMENT,
        runmat_analysis_core::AnalysisFieldLayout::vector3(
            runmat_analysis_core::ObservationQuantity::Displacement,
            runmat_analysis_core::FieldLocation::Node,
        ),
        vec![4, 3],
        vec![0.0; 12],
    ));
    assert_eq!(displacement.topology_id.as_deref(), Some("analysis_mesh"));
    assert_eq!(displacement.element_kind, None);
    assert_eq!(displacement.location, AnalysisFieldLocation::Node);
    assert_eq!(displacement.entity_count, 4);
    assert_eq!(displacement.value_count, 12);
    assert_eq!(displacement.element_count, 12);
    assert_eq!(displacement.component_count, Some(3));
    let nodal_von_mises = AnalysisFieldDescriptor::from_field(&AnalysisField::host_f64(
        FEA_FIELD_STRUCTURAL_NODAL_VON_MISES,
        runmat_analysis_core::AnalysisFieldLayout::scalar(
            runmat_analysis_core::ObservationQuantity::VonMisesStress,
            runmat_analysis_core::FieldLocation::Node,
        ),
        vec![4],
        vec![0.0; 4],
    ));
    assert_eq!(
        nodal_von_mises.topology_id.as_deref(),
        Some("analysis_mesh")
    );
    assert_eq!(nodal_von_mises.element_kind, None);
    assert_eq!(nodal_von_mises.location, AnalysisFieldLocation::Node);
    assert_eq!(nodal_von_mises.entity_count, 4);
    assert_eq!(nodal_von_mises.value_count, 4);
    assert_eq!(nodal_von_mises.element_count, 4);
    assert_eq!(nodal_von_mises.component_count, Some(1));
    let solid_stress = AnalysisFieldDescriptor::from_field(&AnalysisField::host_f64(
        FEA_FIELD_STRUCTURAL_STRESS,
        runmat_analysis_core::AnalysisFieldLayout::symmetric_tensor3(
            runmat_analysis_core::ObservationQuantity::Stress,
            runmat_analysis_core::FieldLocation::Element,
        ),
        vec![1, 6],
        vec![0.0; 6],
    ));
    assert_eq!(solid_stress.topology_id.as_deref(), Some("analysis_mesh"));
    assert_eq!(solid_stress.element_kind.as_deref(), Some("tetrahedron4"));
    assert_eq!(solid_stress.location, AnalysisFieldLocation::Element);
    assert_eq!(solid_stress.entity_count, 1);
    assert_eq!(solid_stress.value_count, 6);
    assert_eq!(solid_stress.element_count, 6);
    assert_eq!(solid_stress.component_count, Some(6));
    let strain_energy_density = AnalysisFieldDescriptor::from_field(&AnalysisField::host_f64(
        FEA_FIELD_STRUCTURAL_STRAIN_ENERGY_DENSITY,
        runmat_analysis_core::AnalysisFieldLayout::scalar(
            runmat_analysis_core::ObservationQuantity::StrainEnergyDensity,
            runmat_analysis_core::FieldLocation::Element,
        ),
        vec![1],
        vec![0.0],
    ));
    assert_eq!(
        strain_energy_density.topology_id.as_deref(),
        Some("analysis_mesh")
    );
    assert_eq!(
        strain_energy_density.element_kind.as_deref(),
        Some("tetrahedron4")
    );
    assert_eq!(
        strain_energy_density.location,
        AnalysisFieldLocation::Element
    );
    assert_eq!(strain_energy_density.entity_count, 1);
    assert_eq!(strain_energy_density.value_count, 1);
    assert_eq!(membrane.topology_id.as_deref(), Some("analysis_mesh"));
    assert_eq!(membrane.element_kind.as_deref(), Some("tri3"));
    assert_eq!(membrane.entity_count, 1);
    assert_eq!(membrane.value_count, 3);
    let energy = AnalysisFieldDescriptor::from_field(&AnalysisField::host_f64(
        FEA_FIELD_STRUCTURAL_TOTAL_STRAIN_ENERGY,
        runmat_analysis_core::AnalysisFieldLayout::scalar(
            runmat_analysis_core::ObservationQuantity::StrainEnergy,
            runmat_analysis_core::FieldLocation::Global,
        ),
        vec![1],
        vec![0.0],
    ));
    assert_eq!(energy.topology_id, None);
    assert_eq!(energy.element_kind, None);
    assert_eq!(energy.entity_count, 1);
    assert_eq!(energy.value_count, 1);
}

#[test]
fn statistical_field_descriptors_preserve_statistic_units_and_locations() {
    let cases = [
        (
            "structural.response_spectrum.nodal.absolute_acceleration",
            "absolute_acceleration",
            Some(AnalysisFieldStatistic::ComponentMagnitudeEnvelope),
            Some("m/s^2"),
            AnalysisFieldLocation::Node,
        ),
        (
            "structural.response_spectrum.envelope.stress",
            "stress",
            Some(AnalysisFieldStatistic::ComponentMagnitudeEnvelope),
            Some("Pa"),
            AnalysisFieldLocation::Element,
        ),
        (
            "structural.random_response.rms.displacement",
            "displacement",
            Some(AnalysisFieldStatistic::RootMeanSquare),
            Some("m"),
            AnalysisFieldLocation::Node,
        ),
        (
            "structural.random_response.variance.stress",
            "stress",
            Some(AnalysisFieldStatistic::Variance),
            Some("Pa^2"),
            AnalysisFieldLocation::Element,
        ),
    ];
    for (field_id, _published_name, statistic, unit, location) in cases {
        let field_location = match location {
            AnalysisFieldLocation::Node => runmat_analysis_core::FieldLocation::Node,
            _ => runmat_analysis_core::FieldLocation::Element,
        };
        let descriptor = AnalysisFieldDescriptor::from_field(&AnalysisField::host_f64(
            field_id,
            runmat_analysis_core::AnalysisFieldLayout::vector3(
                runmat_analysis_core::ObservationQuantity::Displacement,
                field_location,
            ),
            vec![2, 3],
            vec![0.0; 6],
        ));
        assert_eq!(descriptor.statistic, statistic, "{field_id}");
        assert_eq!(descriptor.unit.as_deref(), unit, "{field_id}");
        assert_eq!(descriptor.location, location, "{field_id}");
    }
}

/// A descriptor reports exactly what the publisher declared: quantity, location, components,
/// representation and ordinal, and counts entities from the declared layout. The same field
/// id published with different layouts (element-constant vs nodal temperature) is described
/// by each publisher's declaration, never by its name.
#[test]
fn field_descriptors_project_the_declared_layout() {
    use runmat_analysis_core::{
        AnalysisFieldLayout, ComplexRepresentation, FieldLocation, FieldOrdinal,
        ObservationQuantity as Q,
    };
    let element_temperature = AnalysisFieldDescriptor::from_field(&AnalysisField::host_f64(
        "thermo_mechanical.temperature.0",
        AnalysisFieldLayout::scalar(Q::Temperature, FieldLocation::Element).step(0),
        vec![3],
        vec![300.0; 3],
    ));
    assert_eq!(element_temperature.location, AnalysisFieldLocation::Element);
    assert_eq!(element_temperature.entity_count, 3);
    assert_eq!(element_temperature.component_count, Some(1));
    assert_eq!(element_temperature.layout.ordinal, FieldOrdinal::Step(0));
    let nodal_temperature = AnalysisFieldDescriptor::from_field(&AnalysisField::host_f64(
        "thermo_mechanical.temperature.0",
        AnalysisFieldLayout::scalar(Q::Temperature, FieldLocation::Node).step(0),
        vec![5],
        vec![300.0; 5],
    ));
    assert_eq!(nodal_temperature.location, AnalysisFieldLocation::Node);
    assert_eq!(nodal_temperature.entity_count, 5);
    let acoustic_mode = AnalysisFieldDescriptor::from_field(&AnalysisField::host_f64(
        "acoustic.mode_shape.1",
        AnalysisFieldLayout::scalar(Q::AcousticPressure, FieldLocation::Node).mode(0),
        vec![4],
        vec![1.0; 4],
    ));
    assert_eq!(acoustic_mode.kind, AnalysisFieldKind::Scalar);
    assert_eq!(acoustic_mode.component_count, Some(1));
    assert_eq!(acoustic_mode.entity_count, 4);
    assert_eq!(acoustic_mode.layout.quantity, Q::AcousticPressure);
    let phasor = AnalysisFieldDescriptor::from_field(&AnalysisField::host_f64(
        "acoustic.pressure",
        AnalysisFieldLayout::scalar(Q::AcousticPressure, FieldLocation::Node)
            .representation(ComplexRepresentation::Complex),
        vec![4, 2],
        vec![1.0; 8],
    ));
    assert_eq!(phasor.entity_count, 4);
    assert_eq!(phasor.component_count, Some(1));
    let points = AnalysisFieldDescriptor::from_field(&AnalysisField::host_f64(
        "structural.integration_point_stress",
        AnalysisFieldLayout::symmetric_tensor3(Q::Stress, FieldLocation::IntegrationPoint),
        vec![8, 6],
        vec![0.0; 48],
    ));
    assert_eq!(points.location, AnalysisFieldLocation::IntegrationPoint);
    assert_eq!(points.entity_count, 8);
    assert_eq!(points.kind, AnalysisFieldKind::Tensor);
    let json = serde_json::to_value(&acoustic_mode).unwrap();
    assert_eq!(json["layout"]["quantity"], "acoustic_pressure");
    assert_eq!(json["layout"]["ordinal"]["kind"], "mode");
}

#[test]
fn analysis_results_describes_structural_l2_fields() {
    let _guard = analysis_test_guard();
    let (mesh_root, mesh_path) = write_ready_minimal_analysis_mesh_artifact("structural-l2-fields");
    let model = sample_solid_model();
    let run = analysis_run_linear_static_with_options(
        &model,
        ComputeBackend::Cpu,
        AnalysisRunOptions {
            solver_mesh_artifact_path: Some(mesh_path.display().to_string()),
            ..AnalysisRunOptions::default()
        },
        OperationContext::new(Some("trace-results-structural-fields-1".to_string()), None),
    )
    .expect("run should pass");

    let results = analysis_results_op(
        &run.data,
        AnalysisResultsQuery {
            include_fields: vec![
                FEA_FIELD_STRUCTURAL_STRAIN.to_string(),
                FEA_FIELD_STRUCTURAL_STRESS.to_string(),
                FEA_FIELD_STRUCTURAL_REACTION_FORCE.to_string(),
                FEA_FIELD_STRUCTURAL_TOTAL_STRAIN_ENERGY.to_string(),
                FEA_FIELD_STRUCTURAL_RESIDUAL_NORM.to_string(),
                FEA_FIELD_STRUCTURAL_EQUATION_SCALE.to_string(),
            ],
            include_field_values: false,
            include_diagnostics: false,
            diagnostic_codes: Vec::new(),
            include_modal_results: true,
            include_buckling_results: true,
            mode_indices: Vec::new(),
            include_transient_results: true,
            transient_snapshot_indices: Vec::new(),
            include_nonlinear_results: true,
        },
        OperationContext::new(Some("trace-results-structural-fields-2".to_string()), None),
    )
    .expect("results should pass");

    let descriptor = |field_id: &str| {
        results
            .data
            .field_descriptors
            .iter()
            .find(|descriptor| descriptor.field_id == field_id)
            .expect("descriptor should be present")
    };

    assert_eq!(
        descriptor(FEA_FIELD_STRUCTURAL_STRAIN).kind,
        AnalysisFieldKind::Tensor
    );
    assert_eq!(
        descriptor(FEA_FIELD_STRUCTURAL_STRAIN).component_count,
        Some(6)
    );
    assert_eq!(
        descriptor(FEA_FIELD_STRUCTURAL_STRESS).kind,
        AnalysisFieldKind::Tensor
    );
    assert_eq!(
        descriptor(FEA_FIELD_STRUCTURAL_STRESS).component_count,
        Some(6)
    );
    assert_eq!(
        descriptor(FEA_FIELD_STRUCTURAL_REACTION_FORCE).kind,
        AnalysisFieldKind::Vector
    );
    assert_eq!(
        descriptor(FEA_FIELD_STRUCTURAL_REACTION_FORCE).component_count,
        Some(3)
    );
    assert_eq!(
        descriptor(FEA_FIELD_STRUCTURAL_TOTAL_STRAIN_ENERGY).kind,
        AnalysisFieldKind::Scalar
    );
    assert_eq!(
        descriptor(FEA_FIELD_STRUCTURAL_RESIDUAL_NORM).kind,
        AnalysisFieldKind::Scalar
    );
    assert_eq!(
        descriptor(FEA_FIELD_STRUCTURAL_EQUATION_SCALE).kind,
        AnalysisFieldKind::Scalar
    );
    let _ = fs::remove_dir_all(mesh_root);
}

#[test]
fn analysis_results_describes_beam_resultant_fields() {
    let _guard = analysis_test_guard();
    let model = runmat_analysis_fea::fixtures::fixture_model(
        runmat_analysis_fea::fixtures::FixtureId::StructuralBeamCantileverEndMomentReference,
    );
    let run = analysis_run_linear_static_with_options(
        &model,
        ComputeBackend::Cpu,
        Default::default(),
        OperationContext::new(Some("trace-results-beam-fields-1".to_string()), None),
    )
    .expect("beam run should pass");

    assert_eq!(
        run.data
            .run
            .field(FEA_FIELD_STRUCTURAL_BEAM_BENDING_MOMENT)
            .expect("beam bending moment field should be present")
            .shape,
        vec![1, 2]
    );
    assert_eq!(
        run.data
            .run
            .field(FEA_FIELD_STRUCTURAL_BEAM_TORSION_STRESS)
            .expect("beam torsion stress field should be present")
            .shape,
        vec![1]
    );

    let results = analysis_results_op(
        &run.data,
        AnalysisResultsQuery {
            include_fields: vec![
                FEA_FIELD_STRUCTURAL_BEAM_AXIAL_FORCE.to_string(),
                FEA_FIELD_STRUCTURAL_BEAM_SHEAR_FORCE.to_string(),
                FEA_FIELD_STRUCTURAL_BEAM_TORSION_MOMENT.to_string(),
                FEA_FIELD_STRUCTURAL_BEAM_BENDING_MOMENT.to_string(),
                FEA_FIELD_STRUCTURAL_BEAM_BENDING_STRESS.to_string(),
                FEA_FIELD_STRUCTURAL_BEAM_TORSION_STRESS.to_string(),
            ],
            include_field_values: false,
            include_diagnostics: false,
            diagnostic_codes: Vec::new(),
            include_modal_results: true,
            include_buckling_results: true,
            mode_indices: Vec::new(),
            include_transient_results: true,
            transient_snapshot_indices: Vec::new(),
            include_nonlinear_results: true,
        },
        OperationContext::new(Some("trace-results-beam-fields-2".to_string()), None),
    )
    .expect("beam results should pass");

    let descriptor = |field_id: &str| {
        results
            .data
            .field_descriptors
            .iter()
            .find(|descriptor| descriptor.field_id == field_id)
            .expect("descriptor should be present")
    };

    assert_eq!(
        descriptor(FEA_FIELD_STRUCTURAL_BEAM_AXIAL_FORCE)
            .unit
            .as_deref(),
        Some("N")
    );
    assert_eq!(
        descriptor(FEA_FIELD_STRUCTURAL_BEAM_AXIAL_FORCE).kind,
        AnalysisFieldKind::Scalar
    );
    assert_eq!(
        descriptor(FEA_FIELD_STRUCTURAL_BEAM_SHEAR_FORCE).component_count,
        Some(2)
    );
    assert_eq!(
        descriptor(FEA_FIELD_STRUCTURAL_BEAM_TORSION_MOMENT)
            .unit
            .as_deref(),
        Some("N*m")
    );
    assert_eq!(
        descriptor(FEA_FIELD_STRUCTURAL_BEAM_BENDING_MOMENT).kind,
        AnalysisFieldKind::Vector
    );
    assert_eq!(
        descriptor(FEA_FIELD_STRUCTURAL_BEAM_BENDING_STRESS)
            .unit
            .as_deref(),
        Some("Pa")
    );
    assert_eq!(
        descriptor(FEA_FIELD_STRUCTURAL_BEAM_BENDING_STRESS).component_count,
        Some(2)
    );
    assert_eq!(
        descriptor(FEA_FIELD_STRUCTURAL_BEAM_TORSION_STRESS).kind,
        AnalysisFieldKind::Scalar
    );
    assert_eq!(
        descriptor(FEA_FIELD_STRUCTURAL_BEAM_TORSION_STRESS).location,
        AnalysisFieldLocation::Element
    );
}

#[test]
fn analysis_results_unavailable_beam_continuum_field_maps_typed_error() {
    let _guard = analysis_test_guard();
    let model = sample_model();
    let run = analysis_run_linear_static_with_options(
        &model,
        ComputeBackend::Cpu,
        Default::default(),
        OperationContext::new(Some("trace-results-3".to_string()), None),
    )
    .expect("run should pass");

    let err = analysis_results_op(
        &run.data,
        AnalysisResultsQuery {
            include_fields: vec![FEA_FIELD_STRUCTURAL_STRAIN.to_string()],
            include_field_values: true,

            include_diagnostics: true,
            diagnostic_codes: Vec::new(),
            include_modal_results: true,
            include_buckling_results: true,
            mode_indices: Vec::new(),
            include_transient_results: true,
            transient_snapshot_indices: Vec::new(),
            include_nonlinear_results: true,
        },
        OperationContext::new(Some("trace-results-4".to_string()), None),
    )
    .expect_err("a beam run must not invent continuum strain");

    assert_eq!(err.operation, "fea.results");
    assert_eq!(err.op_version, "fea.results/v1");
    assert_eq!(err.error_code, "RM.FEA.RESULTS.FIELD_NOT_FOUND");
    assert_eq!(
        err.context.get("requested_field"),
        Some(&FEA_FIELD_STRUCTURAL_STRAIN.to_string())
    );
}

#[test]
fn explicit_field_values_are_returned_beyond_the_default_inline_budget() {
    let _guard = analysis_test_guard();
    storage::reset_artifact_store_for_tests();
    let model = sample_model();
    let mut run = analysis_run_linear_static_with_options(
        &model,
        ComputeBackend::Cpu,
        Default::default(),
        OperationContext::new(Some("trace-results-explicit-values".to_string()), None),
    )
    .expect("run should pass")
    .data;
    // One field past the default inline budget, as a large mesh produces.
    let entries = (RESULTS_INLINE_FIELD_BYTES / 8 + 1) as usize;
    run.run_id = format!("{}-large", run.run_id);
    run.run.fields.push(AnalysisField::host_f64(
        "test.large_nodal_field",
        runmat_analysis_core::AnalysisFieldLayout::scalar(
            runmat_analysis_core::ObservationQuantity::Temperature,
            runmat_analysis_core::FieldLocation::Node,
        ),
        vec![entries],
        (0..entries).map(|index| index as f64).collect(),
    ));
    storage::persist_run_result(&run).expect("large run should persist");
    let field_count = run.run.fields.len();

    let bounded =
        bounded_default_results_query(&run.run_id, AnalysisResultsQuery::default()).unwrap();
    assert!(
        !bounded.include_field_values,
        "the default view opens bounded"
    );
    let default_view =
        analysis_results_by_run_id_op(&run.run_id, bounded, OperationContext::new(None, None))
            .expect("bounded results");
    assert!(default_view.data.fields.is_empty());
    assert_eq!(default_view.data.field_descriptors.len(), field_count);

    // A selection that fits the budget keeps its values in the default view.
    let small = bounded_default_results_query(
        &run.run_id,
        AnalysisResultsQuery {
            include_fields: vec![FEA_FIELD_STRUCTURAL_DISPLACEMENT.to_string()],
            ..AnalysisResultsQuery::default()
        },
    )
    .unwrap();
    assert!(small.include_field_values);

    let explicit = analysis_results_by_run_id_op(
        &run.run_id,
        AnalysisResultsQuery {
            include_field_values: true,
            ..AnalysisResultsQuery::default()
        },
        OperationContext::new(None, None),
    )
    .expect("explicit field values");
    assert_eq!(explicit.data.fields.len(), field_count);
    let large = explicit
        .data
        .fields
        .iter()
        .find(|field| field.field_id == "test.large_nodal_field")
        .expect("the large field is returned");
    let AnalysisFieldValues::HostF64(values) = &large.values else {
        panic!("host values expected");
    };
    assert_eq!(values.len(), entries);
    assert_eq!(values[entries - 1], (entries - 1) as f64);
}

#[test]
fn analysis_results_by_run_id_roundtrip_works() {
    let _guard = analysis_test_guard();
    storage::reset_artifact_store_for_tests();
    let model = sample_model();
    let run = analysis_run_linear_static_with_options(
        &model,
        ComputeBackend::Cpu,
        Default::default(),
        OperationContext::new(Some("trace-results-by-id-run".to_string()), None),
    )
    .expect("run should pass");

    let fetched = analysis_results_by_run_id_op(
        &run.data.run_id,
        AnalysisResultsQuery::default(),
        OperationContext::new(Some("trace-results-by-id-fetch".to_string()), None),
    )
    .expect("results by id should pass");

    assert_eq!(fetched.operation, "fea.results");
    assert_eq!(fetched.op_version, "fea.results/v1");
    assert_eq!(fetched.data.summary.field_count, run.data.run.fields.len());
    assert_eq!(
        fetched.data.field_descriptors.len(),
        run.data.run.fields.len()
    );
    let displacement_descriptor = fetched
        .data
        .field_descriptors
        .iter()
        .find(|descriptor| descriptor.field_id == FEA_FIELD_STRUCTURAL_DISPLACEMENT)
        .expect("structural displacement descriptor should replay");
    assert_eq!(displacement_descriptor.unit.as_deref(), Some("m"));
    assert_eq!(
        displacement_descriptor.location,
        AnalysisFieldLocation::Node
    );
    let rotation_descriptor = fetched
        .data
        .field_descriptors
        .iter()
        .find(|descriptor| descriptor.field_id == FEA_FIELD_STRUCTURAL_ROTATION)
        .expect("structural rotation descriptor should replay");
    assert_eq!(rotation_descriptor.unit.as_deref(), Some("rad"));
    let reaction_moment_descriptor = fetched
        .data
        .field_descriptors
        .iter()
        .find(|descriptor| descriptor.field_id == FEA_FIELD_STRUCTURAL_REACTION_MOMENT)
        .expect("structural reaction moment descriptor should replay");
    assert_eq!(reaction_moment_descriptor.unit.as_deref(), Some("N*m"));
    assert_eq!(fetched.data.summary.mode_count, 0);
    assert!(fetched.data.summary.available_mode_indices.is_empty());
    assert_eq!(fetched.data.summary.min_frequency_hz, None);
    assert_eq!(fetched.data.summary.max_frequency_hz, None);
    assert_eq!(fetched.data.summary.max_modal_residual_norm, None);
    assert_eq!(fetched.data.summary.first_mode_converged, None);
    assert_eq!(fetched.data.summary.snapshot_count, 0);
    assert_eq!(fetched.data.summary.time_start_s, None);
    assert_eq!(fetched.data.summary.time_end_s, None);
    assert_eq!(fetched.data.summary.max_transient_residual_norm, None);
    assert_eq!(fetched.data.summary.final_step_converged, None);

    storage::reset_artifact_store_for_tests();
}

#[test]
fn analysis_results_by_run_id_missing_maps_typed_error() {
    let _guard = analysis_test_guard();
    storage::reset_artifact_store_for_tests();
    let err = analysis_results_by_run_id_op(
        "run_missing",
        AnalysisResultsQuery::default(),
        OperationContext::new(Some("trace-results-by-id-missing".to_string()), None),
    )
    .expect_err("missing run id should fail");

    assert_eq!(err.error_code, "RM.FEA.RESULTS.RUN_NOT_FOUND");
    storage::reset_artifact_store_for_tests();
}

#[test]
fn analysis_results_compare_reports_typed_deltas_rejects_unsupported_envelope() {
    let _guard = analysis_test_guard();
    let mut model = sample_model();
    model.sequence.steps = vec![ProcedureStep::structural_nonlinear("unsupported_nonlinear")];
    let error = analysis_run_nonlinear_with_options_op(
        &model,
        ComputeBackend::Cpu,
        AnalysisNonlinearRunOptions::balanced(),
        OperationContext::new(None, None),
    )
    .expect_err("an unsupported envelope must not publish nonlinear snapshots or quality metrics");
    assert_eq!(
        error.error_code,
        "RM.FEA.RUN_NONLINEAR.SOLVER_MODEL_INVALID"
    );
    assert!(!error.retryable);
}

#[test]
fn analysis_trends_summarizes_recent_nonlinear_runs_rejects_unsupported_envelope() {
    let _guard = analysis_test_guard();
    let mut model = sample_model();
    model.sequence.steps = vec![ProcedureStep::structural_nonlinear("unsupported_nonlinear")];
    let error = analysis_run_nonlinear_with_options_op(
        &model,
        ComputeBackend::Cpu,
        AnalysisNonlinearRunOptions::balanced(),
        OperationContext::new(None, None),
    )
    .expect_err("an unsupported envelope must not publish nonlinear snapshots or quality metrics");
    assert_eq!(
        error.error_code,
        "RM.FEA.RUN_NONLINEAR.SOLVER_MODEL_INVALID"
    );
    assert!(!error.retryable);
}

#[test]
fn analysis_trends_classifies_acoustic_runs_separately() {
    let _guard = analysis_test_guard();
    storage::reset_artifact_store_for_tests();

    let geometry = sample_geometry_asset();
    let acoustic_model = test_model_for_geometry(
        &geometry,
        TestModelSpec {
            model_id: "acoustic_trend_model".to_string(),
            kind: TestModelKind::AcousticHarmonic,
        },
        OperationContext::new(None, None),
    )
    .expect("acoustic model should be created");
    for _ in 0..3 {
        let _ = analysis_run_acoustic_with_options_op(
            &acoustic_model.data,
            ComputeBackend::Cpu,
            Default::default(),
            OperationContext::new(None, None),
        )
        .expect("acoustic run should persist for trends");
    }

    let trends = analysis_trends_op(
        AnalysisTrendsQuery { window_size: 2 },
        OperationContext::new(None, None),
    )
    .expect("trends should succeed");

    let acoustic = trends
        .data
        .summaries
        .iter()
        .find(|summary| summary.identity.family() == PhysicsFamily::Acoustic)
        .expect("acoustic trend summary should exist");
    assert_eq!(acoustic.sample_count, 2);
    assert!(acoustic.median_solve_ms.is_some());
    assert!(acoustic.p95_solve_ms.is_some());

    let modal = trends
        .data
        .summaries
        .iter()
        .find(|summary| summary.identity.procedure() == AnalysisProcedure::ModalExtraction);
    assert!(
        modal.is_none(),
        "acoustic runs should not classify as modal"
    );

    storage::reset_artifact_store_for_tests();
}

#[test]
fn analysis_trends_never_records_rejected_reduced_cfd_as_a_run() {
    let _guard = analysis_test_guard();
    storage::reset_artifact_store_for_tests();

    let mut model = sample_model();
    model.sequence.steps = vec![ProcedureStep::fluid_static("cfd_1".to_string())];
    model.cfd = Some(sample_cfd_domain());
    for _ in 0..3 {
        let error = analysis_run_cfd_with_options_op(
            &model,
            ComputeBackend::Cpu,
            Default::default(),
            OperationContext::new(None, None),
        )
        .expect_err("the removed reduced CFD endpoint must not persist a fabricated run");
        assert_eq!(error.error_code, "RM.FEA.RUN_CFD.AUTHORED_STUDY_REQUIRED");
    }

    let trends = analysis_trends_op(
        AnalysisTrendsQuery { window_size: 2 },
        OperationContext::new(None, None),
    )
    .expect("trends should succeed");

    assert!(trends
        .data
        .summaries
        .iter()
        .all(|summary| summary.identity.family() != PhysicsFamily::Fluid));

    storage::reset_artifact_store_for_tests();
}

#[test]
fn analysis_trends_omit_rejected_coupled_flow_runs() {
    let _guard = analysis_test_guard();
    storage::reset_artifact_store_for_tests();
    for _ in 0..3 {
        analysis_run_cht_op(&sample_cht_model(), OperationContext::new(None, None))
            .expect_err("rejected CHT runs are not persisted");
        analysis_run_fsi_op(&sample_fsi_model(), OperationContext::new(None, None))
            .expect_err("rejected FSI runs are not persisted");
    }
    let trends = analysis_trends_op(
        AnalysisTrendsQuery { window_size: 2 },
        OperationContext::new(None, None),
    )
    .expect("trends should succeed");
    assert!(trends
        .data
        .summaries
        .iter()
        .all(|summary| !matches!(summary.identity.family(), PhysicsFamily::Coupled(_))));
    storage::reset_artifact_store_for_tests();
}

#[test]
fn analysis_run_transient_rejects_configured_thermomechanical_step() {
    let _guard = analysis_test_guard();
    let mut model = sample_solid_model();
    model.sequence.steps = vec![ProcedureStep::coupled(
        "transient_thermo_mechanical",
        CoupledPhysicsFamily::Thermomechanical,
        AnalysisProcedure::TransientResponse,
    )];
    set_model_thermo_coupling(
        &mut model,
        TestThermoMechanicalCoupling {
            reference_temperature_k: runmat_types::physical::Temperature::new(293.15),
            applied_temperature_delta_k: runmat_types::physical::TemperatureDifference::new(65.0),
            thermal_expansion_coefficient: runmat_types::physical::ThermalExpansion::new(1.2e-5),
            region_temperature_deltas: Vec::new(),
        },
    );
    let error = analysis_run_transient_with_options_op(
        &model,
        ComputeBackend::Cpu,
        AnalysisTransientRunOptions::default(),
        OperationContext::new(None, None),
    )
    .expect_err("thermomechanical transient response is not an admitted solver family");

    assert_eq!(error.error_code, "RM.FEA.RUN_TRANSIENT.INVALID_MODEL");
}

#[test]
fn analysis_run_transient_rejects_invalid_electro_thermal_voltage() {
    let _guard = analysis_test_guard();
    let mut model = sample_model();
    model.sequence.steps = vec![ProcedureStep::structural_transient(
        "transient_invalid_electro_voltage".to_string(),
    )];
    set_model_electro_coupling(
        &mut model,
        TestElectroThermalCoupling {
            reference_temperature_k: runmat_types::physical::Temperature::new(293.15),
            applied_voltage_v: f64::NAN,
            base_electrical_conductivity_s_per_m: 3.5e7,
            resistive_heating_coefficient: runmat_types::physical::InverseTemperature::new(4.0e-4),
        },
    );

    let err = analysis_run_transient_with_options_op(
        &model,
        ComputeBackend::Cpu,
        Default::default(),
        OperationContext::new(None, None),
    )
    .expect_err("transient run should reject invalid electro-thermal voltage");

    assert_eq!(err.error_code, "RM.FEA.RUN_TRANSIENT.SOLVER_MODEL_INVALID");
    assert!(err.message.contains("signal"));
}

#[test]
fn analysis_run_transient_rejects_electro_thermal_context_on_structural_step() {
    let _guard = analysis_test_guard();
    let mut model = sample_solid_model();
    model.sequence.steps = vec![ProcedureStep::structural_transient(
        "transient_with_unsupported_electro_thermal_context".to_string(),
    )];
    set_model_electro_coupling(
        &mut model,
        TestElectroThermalCoupling {
            reference_temperature_k: runmat_types::physical::Temperature::new(293.15),
            applied_voltage_v: 36.0,
            base_electrical_conductivity_s_per_m: 3.5e7,
            resistive_heating_coefficient: runmat_types::physical::InverseTemperature::new(4.0e-4),
        },
    );

    let error = analysis_run_transient_with_options_op(
        &model,
        ComputeBackend::Cpu,
        Default::default(),
        OperationContext::new(None, None),
    )
    .expect_err("structural transient execution must reject electro-thermal context");

    assert_eq!(
        error.error_code,
        "RM.FEA.RUN_TRANSIENT.SOLVER_MODEL_INVALID"
    );
    assert!(
        error
            .message
            .contains("structural pipelines do not implement an electrothermal"),
        "{}",
        error.message
    );
}

#[test]
fn analysis_results_summary_surfaces_thermo_nonlinear_metrics_rejects_unsupported_envelope() {
    let _guard = analysis_test_guard();
    let mut model = sample_solid_model();
    model.sequence.steps = vec![ProcedureStep::coupled(
        "unsupported_coupled",
        CoupledPhysicsFamily::Thermomechanical,
        AnalysisProcedure::Nonlinear,
    )];
    let error = analysis_run_nonlinear_with_options_op(
        &model,
        ComputeBackend::Cpu,
        AnalysisNonlinearRunOptions::balanced(),
        OperationContext::new(None, None),
    )
    .expect_err("an unsupported envelope must not publish nonlinear snapshots or quality metrics");
    assert_eq!(
        error.error_code,
        "RM.FEA.RUN_NONLINEAR.FORMULATION_NOT_ADMITTED"
    );
    assert!(!error.retryable);
}

#[test]
fn analysis_run_nonlinear_rejects_explicit_connectivity_that_drops_thermo_coupling() {
    let _guard = analysis_test_guard();
    let published = storage::list_run_metadata().unwrap().len();
    let mut model = sample_model();
    model.sequence.steps = vec![ProcedureStep::coupled(
        "nonlinear_explicit_thermo_mechanical",
        CoupledPhysicsFamily::Thermomechanical,
        AnalysisProcedure::Nonlinear,
    )];
    set_model_thermo_coupling(
        &mut model,
        TestThermoMechanicalCoupling {
            reference_temperature_k: runmat_types::physical::Temperature::new(293.15),
            applied_temperature_delta_k: runmat_types::physical::TemperatureDifference::new(80.0),
            thermal_expansion_coefficient: runmat_types::physical::ThermalExpansion::new(1.2e-5),
            region_temperature_deltas: Vec::new(),
        },
    );

    let error = analysis_run_nonlinear_with_options_op(
        &model,
        ComputeBackend::Cpu,
        AnalysisNonlinearRunOptions::solid_recommended(),
        OperationContext::new(None, None),
    )
    .expect_err("explicit connectivity must not silently discard thermo coupling");

    assert_eq!(
        error.error_code,
        "RM.FEA.RUN_NONLINEAR.FORMULATION_NOT_ADMITTED"
    );
    // This authored nonlinear coupled family has no available formulation.
    // It must fail before attempting to substitute the structural realization.
    assert!(!error.retryable);
    assert_eq!(storage::list_run_metadata().unwrap().len(), published);
}

#[test]
fn analysis_trends_ignore_noncanonical_artifacts() {
    let _guard = analysis_test_guard();
    storage::reset_artifact_store_for_tests();
    let root = temp_artifact_root("trends-mixed-schema");
    let _ = fs::remove_dir_all(&root);
    storage::configure_artifact_store(storage::AnalysisArtifactStoreConfig::Filesystem {
        root: root.clone(),
    })
    .expect("configure filesystem artifact store");

    let (model, options) = nonlinear_physical::artifact_seed(&root);
    let run = analysis_run_nonlinear_with_options_op(
        &model,
        ComputeBackend::Cpu,
        options,
        OperationContext::new(None, None),
    )
    .expect("seed nonlinear run should succeed");

    // Deliberately omit the canonical persisted-artifact envelope. Derive this
    // invalid legacy record from the typed result, not the current envelope's
    // implementation fields, and leave the valid canonical record untouched.
    let mut legacy = serde_json::to_value(&run.data).expect("encode unwrapped run result");
    legacy["run_id"] = serde_json::json!(format!("{}_legacy", run.data.run_id));
    fs::write(
        root.join("runs")
            .join(format!("{}_legacy.json", run.data.run_id)),
        serde_json::to_vec_pretty(&legacy).expect("encode legacy artifact"),
    )
    .expect("write legacy artifact");

    let trends = analysis_trends_op(
        AnalysisTrendsQuery { window_size: 8 },
        OperationContext::new(None, None),
    )
    .expect("trends should ignore noncanonical artifacts");
    let nonlinear = trends
        .data
        .summaries
        .iter()
        .find(|summary| summary.identity.procedure() == AnalysisProcedure::Nonlinear)
        .expect("nonlinear summary should be present");
    assert_eq!(nonlinear.sample_count, 1);
    assert!(nonlinear.p95_solve_ms.unwrap_or(0.0) >= nonlinear.median_solve_ms.unwrap_or(0.0));

    storage::reset_artifact_store_for_tests();
    let _ = fs::remove_dir_all(&root);
}

fn temp_artifact_root(test_name: &str) -> PathBuf {
    std::env::temp_dir().join(format!(
        "runmat-analysis-tests-{}-{}",
        test_name,
        Utc::now().timestamp_nanos_opt().unwrap_or_default()
    ))
}

struct EnvVarRestoreGuard {
    key: &'static str,
    previous: Option<String>,
}

impl Drop for EnvVarRestoreGuard {
    fn drop(&mut self) {
        if let Some(previous) = self.previous.as_ref() {
            std::env::set_var(self.key, previous);
        } else {
            std::env::remove_var(self.key);
        }
    }
}

pub(super) struct FeaRuntimeConfigRestoreGuard {
    previous: FeaRuntimeConfig,
}

impl Drop for FeaRuntimeConfigRestoreGuard {
    fn drop(&mut self) {
        configure_fea_runtime(self.previous.clone()).expect("restore FEA runtime config");
    }
}

pub(super) fn scoped_study_artifact_root(root: &Path) -> FeaRuntimeConfigRestoreGuard {
    let previous = current_fea_runtime_config();
    configure_fea_runtime(FeaRuntimeConfig {
        study_artifact_root: Some(root.to_path_buf()),
        ..previous.clone()
    })
    .expect("configure FEA study artifact root");
    FeaRuntimeConfigRestoreGuard { previous }
}

fn scoped_fea_artifact_roots(root: &Path) -> FeaRuntimeConfigRestoreGuard {
    let previous = current_fea_runtime_config();
    configure_fea_runtime(FeaRuntimeConfig {
        artifact_root: Some(root.to_path_buf()),
        study_artifact_root: Some(root.join("studies")),
        ..Default::default()
    })
    .expect("configure FEA artifact roots");
    FeaRuntimeConfigRestoreGuard { previous }
}

fn write_ready_minimal_analysis_mesh_artifact(test_name: &str) -> (PathBuf, PathBuf) {
    write_ready_analysis_mesh_artifact(test_name, &ready_minimal_analysis_mesh())
}

/// The minimal tetrahedron with canonical CAD face provenance (`root`, `tip`), shaped like the
/// output of the runner-native exact meshing provider.
fn ready_minimal_analysis_mesh() -> SolverMeshArtifact {
    let mut mesh =
        runmat_meshing_core::fixtures::canonical_tetrahedron_solver_mesh(ElementOrder::Tet4);
    mesh.topology.boundary_faces[0]
        .provenance
        .push(persistent_face("root"));
    mesh.topology.boundary_faces[1]
        .provenance
        .push(persistent_face("tip"));
    for face in &mut mesh.topology.boundary_faces {
        face.provenance.sort();
    }
    mesh.canonical_digest = StableDigest::ZERO;
    mesh.seal_canonical_digest()
        .expect("seal canonical solver mesh fixture");
    mesh
}

fn write_ready_analysis_mesh_artifact(
    test_name: &str,
    mesh: &SolverMeshArtifact,
) -> (PathBuf, PathBuf) {
    let root = temp_artifact_root(test_name);
    let _ = fs::remove_dir_all(&root);
    fs::create_dir_all(&root).expect("create solver mesh artifact root");
    let mesh_path = root.join("solver_mesh.cbor");
    fs::write(
        &mesh_path,
        mesh.canonical_encode()
            .expect("encode canonical solver mesh artifact"),
    )
    .expect("write canonical solver mesh artifact");
    (root, mesh_path)
}

fn persistent_face(source_topology_id: &str) -> PersistentEntityId {
    PersistentEntityId::new(
        PersistentEntityKind::Face,
        source_topology_id.to_owned(),
        vec!["root".to_owned()],
    )
}

fn run_study_through_meshing_provider(
    spec: &AnalysisStudySpec,
) -> Result<AnalysisStudyRunData, String> {
    let mesh = ready_minimal_analysis_mesh();
    let evidence = runmat_meshing_core::fixtures::canonical_tetrahedron_meshing_evidence(&mesh);
    let _provider = crate::meshing::replace_meshing_execution_provider(Some(std::sync::Arc::new(
        move |_request: &crate::meshing::MeshingExecutionRequest| {
            Ok(crate::meshing::MeshingExecutionOutput {
                artifact: mesh.clone(),
                evidence: evidence.clone(),
            })
        },
    )))
    .expect("install meshing provider");
    analysis_run_study_op(spec, OperationContext::new(None, None))
        .map(|envelope| envelope.data)
        .map_err(|error| format!("{error:?}"))
}

#[test]
fn provider_meshing_output_runs_the_study_by_canonical_selectors_only() {
    let _guard = analysis_test_guard();
    storage::reset_artifact_store_for_tests();
    let root = temp_artifact_root("provider-meshing-canonical-selectors");
    let _ = fs::remove_dir_all(&root);
    let _runtime_guard = scoped_fea_artifact_roots(&root);
    let mut spec = sample_linear_static_study_spec();
    spec.geometry.source.sha256 = "ab".repeat(32);
    let canonical = |target: &mut RegionTarget| {
        if matches!(target.region_id.as_str(), "root" | "tip") {
            target.region_id = persistent_face(&target.region_id)
                .analysis_selection_id()
                .unwrap();
        }
    };
    for constraint in &mut spec.model.constraints {
        constraint.visit_targets_mut(canonical);
    }
    for action in &mut spec.model.excitations {
        if let ExcitationTarget::Region(target) = &mut action.excitation_mut().target {
            canonical(target);
        }
    }
    assert!(spec.solver_mesh_artifact_path.is_none());
    assert!(spec.meshing_settings.is_some());

    // The provider's mesh carries canonical CAD identities; the model names them directly.
    let run = run_study_through_meshing_provider(&spec)
        .expect("canonical selectors resolve against the provider mesh");
    assert!(run.solver_mesh_artifact_path.is_some());

    // A legacy importer ordinal is not an identity: nothing maps it, so the study is rejected.
    let mut legacy = spec.clone();
    legacy.model.constraints[0].visit_targets_mut(|target| {
        *target = RegionTarget::boundary("face_000001");
    });
    let error = run_study_through_meshing_provider(&legacy)
        .expect_err("a legacy face_000001 selector must be rejected");
    assert!(error.contains("face_000001"), "{error}");
    let _ = fs::remove_dir_all(&root);
}

#[test]
fn one_analysis_material_requires_and_uses_an_explicit_solid_assignment() {
    let model = sample_solid_model();
    let mesh = runmat_meshing_core::fixtures::canonical_tetrahedron_solver_mesh(ElementOrder::Tet4);

    assert_eq!(
        linear_static_result::property_compilation_quality_reason(&model, &mesh),
        None
    );
}

#[test]
fn solid_material_quality_uses_resolved_occurrence_provenance() {
    let mut model = sample_solid_model();
    let mesh = runmat_meshing_core::fixtures::canonical_tetrahedron_solver_mesh(ElementOrder::Tet4);
    let source_solid = mesh.topology.volume_elements[0].provenance[0].clone();
    model.material_assignments[0].target = runmat_analysis_core::PropertyRegionTarget::solid(
        runmat_meshing_core::fixtures::fixture_region("authored_body"),
    );
    model.material_assignments[0]
        .target
        .bind_resolved_entities(vec![source_solid.clone()])
        .expect("resolve authored solid");
    assert_eq!(
        linear_static_result::property_compilation_quality_reason(&model, &mesh),
        None
    );

    let mut other_occurrence = source_solid;
    other_occurrence
        .assembly_path
        .push("other-instance".to_string());
    model.material_assignments[0]
        .target
        .bind_resolved_entities(vec![other_occurrence])
        .expect("resolve another occurrence");
    let reason = linear_static_result::property_compilation_quality_reason(&model, &mesh)
        .expect("different occurrence must fail compiled property coverage");
    assert_eq!(
        reason.code,
        QualityReasonCode::StructuralPropertyCompilationFailed
    );
    assert!(reason
        .detail
        .contains("requires exactly one compatible material assignment"));
}

#[test]
fn compiled_solid_property_quality_rejects_gaps_and_overlaps() {
    let mut model = sample_solid_model();
    model.material_assignments.clear();
    let mesh = runmat_meshing_core::fixtures::canonical_tetrahedron_solver_mesh(ElementOrder::Tet4);

    let gap = linear_static_result::property_compilation_quality_reason(&model, &mesh)
        .expect("compiled property coverage is required");
    assert_eq!(
        gap.code,
        QualityReasonCode::StructuralPropertyCompilationFailed
    );
    assert!(gap.detail.contains("found 0"));

    model.material_assignments = vec![MaterialAssignment {
        target: runmat_analysis_core::PropertyRegionTarget::solid(
            runmat_meshing_core::fixtures::fixture_region("region"),
        ),
        expected_material_id: "mat_steel".to_string(),
        assigned_material_id: "mat_steel".to_string(),
        confidence: EvidenceConfidence::Verified,
    }];
    assert_eq!(
        linear_static_result::property_compilation_quality_reason(&model, &mesh),
        None
    );
    model.material_assignments.push(MaterialAssignment {
        target: runmat_analysis_core::PropertyRegionTarget::all_solids(),
        expected_material_id: "mat_steel".to_string(),
        assigned_material_id: "mat_steel".to_string(),
        confidence: EvidenceConfidence::Verified,
    });
    let overlap = linear_static_result::property_compilation_quality_reason(&model, &mesh)
        .expect("overlapping compiled property owners must fail");
    assert_eq!(
        overlap.code,
        QualityReasonCode::StructuralPropertyCompilationFailed
    );
    assert!(overlap.detail.contains("found 2"));
}

#[test]
fn static_completion_rejects_result_with_unowned_solid_elements() {
    let _guard = analysis_test_guard();
    let (mesh_root, mesh_path) =
        write_ready_minimal_analysis_mesh_artifact("completion-property-coverage");
    let model = sample_solid_model();
    let options = AnalysisRunOptions {
        solver_mesh_artifact_path: Some(mesh_path.display().to_string()),
        ..AnalysisRunOptions::default()
    };
    let solved = analysis_run_linear_static_with_options(
        &model,
        ComputeBackend::Cpu,
        options.clone(),
        OperationContext::new(Some("trace-completion-property-coverage".to_string()), None),
    )
    .expect("valid solid solve");
    assert_eq!(solved.data.run_status, RunStatus::Completed);
    let mesh = runmat_meshing_core::SolverMeshArtifact::canonical_decode(
        &fs::read(&mesh_path).expect("read solver mesh"),
    )
    .expect("decode solver mesh");
    let mut unowned = model.clone();
    unowned.material_assignments.clear();
    let published = linear_static_result::admit_linear_static_result(
        linear_static_result::LinearStaticResultInput {
            model: &unowned,
            solver_mesh: Some(&mesh),
            requested_backend: ComputeBackend::Cpu,
            options: &options,
            assigned_run_id: Some("rejected-property-coverage"),
            placement: Default::default(),
        },
        solved.data.run.clone(),
    )
    .expect("completion must publish an attributable rejection");
    assert_eq!(published.run_status, RunStatus::Rejected);
    assert!(!published.numerically_acceptable);
    assert!(published.quality_reasons.iter().any(|reason| {
        reason.code == QualityReasonCode::StructuralPropertyCompilationFailed
            && reason.detail.contains("found 0")
    }));

    let mut thermal_only = model;
    thermal_only.materials[0].elasticity = None;
    let published = linear_static_result::admit_linear_static_result(
        linear_static_result::LinearStaticResultInput {
            model: &thermal_only,
            solver_mesh: Some(&mesh),
            requested_backend: ComputeBackend::Cpu,
            options: &options,
            assigned_run_id: Some("rejected-structural-capability"),
            placement: Default::default(),
        },
        solved.data.run,
    )
    .expect("completion must reject a solid without structural capabilities");
    assert_eq!(published.run_status, RunStatus::Rejected);
    assert!(!published.numerically_acceptable);
    assert!(published.quality_reasons.iter().any(|reason| {
        reason.code == QualityReasonCode::StructuralPropertyCompilationFailed
            && reason.detail.contains("IsotropicLinearElasticity")
    }));
    let _ = fs::remove_dir_all(mesh_root);
}

#[test]
fn analysis_results_by_run_id_rejects_unwrapped_artifacts() {
    let _guard = analysis_test_guard();
    storage::reset_artifact_store_for_tests();
    let root = temp_artifact_root("legacy-loadable");
    let _ = fs::remove_dir_all(&root);
    storage::configure_artifact_store(storage::AnalysisArtifactStoreConfig::Filesystem {
        root: root.clone(),
    })
    .expect("configure filesystem artifact store");

    let (model, options) = nonlinear_physical::artifact_seed(&root);
    let run = analysis_run_nonlinear_with_options_op(
        &model,
        ComputeBackend::Cpu,
        options,
        OperationContext::new(None, None),
    )
    .expect("nonlinear run should succeed");
    let run_id = run.data.run_id.clone();
    let run_path = root.join("runs").join(format!("{run_id}.json"));

    let mut legacy_value = serde_json::to_value(&run.data).expect("serialize nonlinear run");
    let nonlinear = legacy_value
        .get_mut("nonlinear_results")
        .and_then(|value| value.as_object_mut())
        .expect("nonlinear results should be object");
    nonlinear.remove("increment_norms");
    nonlinear.remove("iteration_counts");
    nonlinear.remove("failed_increments");
    nonlinear.remove("line_search_backtracks");
    nonlinear.remove("max_line_search_backtracks_per_increment");
    nonlinear.remove("tangent_rebuild_count");
    nonlinear.remove("iteration_spike_count");
    nonlinear.remove("convergence_stall_count");
    nonlinear.remove("backtrack_burst_count");
    fs::write(
        &run_path,
        serde_json::to_vec_pretty(&legacy_value).expect("encode legacy artifact"),
    )
    .expect("write legacy artifact");

    let error = analysis_results_by_run_id_op(
        &run_id,
        AnalysisResultsQuery::default(),
        OperationContext::new(None, None),
    )
    .expect_err("unwrapped analysis artifacts must not enter the live schema");
    assert_eq!(error.error_type, OperationErrorType::Internal);

    storage::reset_artifact_store_for_tests();
    let _ = fs::remove_dir_all(&root);
}

#[test]
fn analysis_results_by_run_id_rejects_unknown_artifact_schema() {
    let _guard = analysis_test_guard();
    storage::reset_artifact_store_for_tests();
    let root = temp_artifact_root("future-extra-fields");
    let _ = fs::remove_dir_all(&root);
    storage::configure_artifact_store(storage::AnalysisArtifactStoreConfig::Filesystem {
        root: root.clone(),
    })
    .expect("configure filesystem artifact store");

    let (model, options) = nonlinear_physical::artifact_seed(&root);
    let run = analysis_run_nonlinear_with_options_op(
        &model,
        ComputeBackend::Cpu,
        options,
        OperationContext::new(None, None),
    )
    .expect("nonlinear run should succeed");
    let run_id = run.data.run_id.clone();
    let run_path = root.join("runs").join(format!("{run_id}.json"));

    let wrapped = serde_json::json!({
        "schema_version": "analysis_run_artifact/unknown",
        "created_at": Utc::now().to_rfc3339(),
        "op_version": "fea.run_nonlinear/v1",
        "run": run.data,
    });
    fs::write(
        &run_path,
        serde_json::to_vec_pretty(&wrapped).expect("encode future artifact"),
    )
    .expect("write future artifact");

    let error = analysis_results_by_run_id_op(
        &run_id,
        AnalysisResultsQuery::default(),
        OperationContext::new(None, None),
    )
    .expect_err("unknown analysis artifact schemas must be rejected");
    assert_eq!(error.error_type, OperationErrorType::Internal);

    storage::reset_artifact_store_for_tests();
    let _ = fs::remove_dir_all(&root);
}

// witness: FEA-SHARED-RESOURCES-001/phase_lifetimes @
//   shared_infrastructure/canonical_mixed_topology/numerical_kernel; coverage: partial;
//   boundary: a thermal steady run with no observations, staged in a publication (its field
//   chunks stream to the spill area), reads back through the staged-record reader with every
//   temperature and heat-flux field unchanged; this is the read thermal observation binding uses.
#[test]
fn staged_thermal_steady_run_reads_back_its_streamed_fields() {
    let _guard = analysis_test_guard();
    storage::reset_artifact_store_for_tests();
    let (mesh_root, mesh_path) = write_ready_minimal_analysis_mesh_artifact("staged-thermal");
    let mut model = sample_solid_model();
    model.sequence.steps = vec![ProcedureStep::thermal_steady("staged_thermal".to_string())];
    model.thermal = Some(runmat_analysis_core::ThermalDomain {
        reference_temperature: Temperature::new(293.15),
    });
    set_model_thermo_coupling(
        &mut model,
        TestThermoMechanicalCoupling {
            reference_temperature_k: runmat_types::physical::Quantity::new(293.15),
            applied_temperature_delta_k: runmat_types::physical::Quantity::new(20.0),
            thermal_expansion_coefficient: runmat_types::physical::Quantity::new(1.2e-5),
            region_temperature_deltas: Vec::new(),
        },
    );
    let run = analysis_run_thermal_with_options_op(
        &model,
        ComputeBackend::Cpu,
        AnalysisThermalRunOptions {
            solver_mesh_artifact_path: Some(mesh_path.display().to_string()),
            ..Default::default()
        },
        OperationContext::new(None, None),
    )
    .expect("thermal steady run should succeed")
    .data;
    assert!(run.run.observations.is_empty() && run.run.series_reductions.is_empty());
    let fields = crate::analysis::operations::collect_analysis_result_fields(&run);
    assert!(
        !fields.is_empty(),
        "a thermal steady run carries temperature fields"
    );

    let (restored, spilled) = storage::publication::stage_and_read_back_for_tests(&run)
        .expect("a staged thermal run reads back with its streamed field chunks");
    assert!(
        spilled > 0,
        "staging streams the thermal field chunks to the spill area"
    );
    assert_eq!(
        crate::analysis::operations::collect_analysis_result_fields(&restored),
        fields
    );
    assert_eq!(restored.run.observations, run.run.observations);

    storage::reset_artifact_store_for_tests();
    let _ = fs::remove_dir_all(&mesh_root);
}

#[test]
fn analysis_artifacts_record_family_specific_op_versions() {
    let _guard = analysis_test_guard();
    storage::reset_artifact_store_for_tests();
    let root = temp_artifact_root("family-op-version");
    let _ = fs::remove_dir_all(&root);
    storage::configure_artifact_store(storage::AnalysisArtifactStoreConfig::Filesystem {
        root: root.clone(),
    })
    .expect("configure filesystem artifact store");
    let (mesh_root, mesh_path) = write_ready_minimal_analysis_mesh_artifact("family-op-version");

    let acoustic_model = test_model_for_geometry(
        &sample_geometry_asset(),
        TestModelSpec {
            model_id: "acoustic_op_version_model".to_string(),
            kind: TestModelKind::AcousticHarmonic,
        },
        OperationContext::new(None, None),
    )
    .expect("acoustic model should be created");
    let acoustic = analysis_run_acoustic_with_options_op(
        &acoustic_model.data,
        ComputeBackend::Cpu,
        Default::default(),
        OperationContext::new(None, None),
    )
    .expect("acoustic run should succeed");
    let mut thermal_model = sample_solid_model();
    thermal_model.sequence.steps = vec![ProcedureStep::thermal_transient(
        "thermal_op_version".to_string(),
    )];
    thermal_model.thermal = Some(runmat_analysis_core::ThermalDomain {
        reference_temperature: Temperature::new(293.15),
    });
    set_model_thermo_coupling(
        &mut thermal_model,
        TestThermoMechanicalCoupling {
            reference_temperature_k: runmat_types::physical::Temperature::new(293.15),
            applied_temperature_delta_k: runmat_types::physical::TemperatureDifference::new(20.0),
            thermal_expansion_coefficient: runmat_types::physical::ThermalExpansion::new(1.2e-5),
            region_temperature_deltas: Vec::new(),
        },
    );
    let thermal = analysis_run_thermal_with_options_op(
        &thermal_model,
        ComputeBackend::Cpu,
        AnalysisThermalRunOptions {
            solver_mesh_artifact_path: Some(mesh_path.display().to_string()),
            ..Default::default()
        },
        OperationContext::new(None, None),
    )
    .expect("thermal run should succeed");

    let read_op_version = |run_id: &str| -> String {
        let path = root.join("runs").join(format!("{run_id}.json"));
        let raw = fs::read_to_string(path).expect("read wrapped artifact");
        let wrapped: serde_json::Value =
            serde_json::from_str(&raw).expect("parse wrapped artifact");
        wrapped
            .get("op_version")
            .and_then(|value| value.as_str())
            .expect("wrapped artifact should include op_version")
            .to_string()
    };

    assert_eq!(read_op_version(&thermal.data.run_id), "fea.run_thermal/v1");
    assert_eq!(
        read_op_version(&acoustic.data.run_id),
        "fea.run_acoustic/v1"
    );

    storage::reset_artifact_store_for_tests();
    let _ = fs::remove_dir_all(&root);
    let _ = fs::remove_dir_all(&mesh_root);
}

#[test]
fn analysis_artifact_retention_prunes_old_runs_per_kind() {
    let _guard = analysis_test_guard();
    storage::reset_artifact_store_for_tests();
    let root = temp_artifact_root("retention-prune");
    let _ = fs::remove_dir_all(&root);
    storage::configure_artifact_store(storage::AnalysisArtifactStoreConfig::Filesystem {
        root: root.clone(),
    })
    .expect("configure filesystem artifact store");
    std::env::set_var("RUNMAT_FEA_ARTIFACT_MAX_RUNS_PER_KIND", "2");
    std::env::remove_var("RUNMAT_FEA_ARTIFACT_MAX_RUNS");

    let (model, options) = nonlinear_physical::artifact_seed(&root);
    let mut run_ids = Vec::new();
    for _ in 0..5 {
        let run = analysis_run_nonlinear_with_options_op(
            &model,
            ComputeBackend::Cpu,
            options.clone(),
            OperationContext::new(None, None),
        )
        .expect("nonlinear run should succeed");
        run_ids.push(run.data.run_id.clone());
    }

    let run_dir = root.join("runs");
    let kept_files = fs::read_dir(&run_dir)
        .expect("read run dir")
        .filter_map(|entry| entry.ok())
        .filter(|entry| entry.path().extension().and_then(|ext| ext.to_str()) == Some("json"))
        .count();
    assert!(kept_files <= 2);
    assert!(storage::load_run_result(&run_ids[0])
        .expect("load pruned result")
        .is_none());
    assert!(
        storage::load_run_result(run_ids.last().expect("latest run id"))
            .expect("load latest result")
            .is_some()
    );

    std::env::remove_var("RUNMAT_FEA_ARTIFACT_MAX_RUNS_PER_KIND");
    storage::reset_artifact_store_for_tests();
    let _ = fs::remove_dir_all(&root);
}

#[test]
fn analysis_results_by_run_id_filesystem_replay_is_stable() {
    let _guard = analysis_test_guard();
    storage::reset_artifact_store_for_tests();
    let root = temp_artifact_root("filesystem-replay");
    let _ = fs::remove_dir_all(&root);
    storage::configure_artifact_store(storage::AnalysisArtifactStoreConfig::Filesystem {
        root: root.clone(),
    })
    .expect("configure filesystem artifact store");

    let (model, options) = nonlinear_physical::artifact_seed(&root);
    let run = analysis_run_nonlinear_with_options_op(
        &model,
        ComputeBackend::Cpu,
        options,
        OperationContext::new(None, None),
    )
    .expect("nonlinear run should succeed");

    let first = analysis_results_by_run_id_op(
        &run.data.run_id,
        AnalysisResultsQuery::default(),
        OperationContext::new(None, None),
    )
    .expect("load first replay");
    let second = analysis_results_by_run_id_op(
        &run.data.run_id,
        AnalysisResultsQuery::default(),
        OperationContext::new(None, None),
    )
    .expect("load second replay");

    assert_eq!(first.data.summary, second.data.summary);
    assert_eq!(first.data.run_status, second.data.run_status);
    assert_eq!(
        first.data.numerically_acceptable,
        second.data.numerically_acceptable
    );
    assert_eq!(first.data.quality_reasons, second.data.quality_reasons);

    storage::reset_artifact_store_for_tests();
    let _ = fs::remove_dir_all(&root);
}

#[test]
fn requested_amg_preconditioner_records_execution_without_a_false_fallback() {
    let _guard = analysis_test_guard();
    let (mesh_root, mesh_path) =
        write_ready_minimal_analysis_mesh_artifact("preconditioner-fallback");
    let model = sample_solid_model();
    let envelope = analysis_run_linear_static_with_options(
        &model,
        ComputeBackend::Cpu,
        AnalysisRunOptions {
            deterministic_mode: true,
            precision_mode: PrecisionMode::Fp64,
            preconditioner_mode: PreconditionerMode::Amg,
            quality_policy: QualityPolicy::Balanced,
            solver_mesh_artifact_path: Some(mesh_path.display().to_string()),
        },
        OperationContext::new(Some("trace-preconditioner-fallback".to_string()), None),
    )
    .expect("run should succeed");

    assert_eq!(
        envelope.data.provenance.preconditioner,
        SolverPreconditioner::Amg
    );
    assert!(!envelope
        .data
        .provenance
        .fallback_events
        .iter()
        .any(|event| event.reason == FallbackReason::RequestedPreconditionerUnavailable));
    let _ = fs::remove_dir_all(&mesh_root);
}

#[test]
fn solver_mesh_geometry_revision_must_match_the_analysis_model() {
    let _guard = analysis_test_guard();
    let (mesh_root, mesh_path) =
        write_ready_minimal_analysis_mesh_artifact("geometry-revision-mismatch");
    let model = sample_solid_model();
    let mut mesh = SolverMeshArtifact::canonical_decode(
        &fs::read(&mesh_path).expect("read solver mesh fixture"),
    )
    .expect("decode solver mesh fixture");
    mesh.geometry.geometry_revision = u64::from(model.geometry_revision) + 1;
    mesh.canonical_digest = StableDigest::ZERO;
    mesh.seal_canonical_digest()
        .expect("reseal mismatched solver mesh");
    fs::write(
        &mesh_path,
        mesh.canonical_encode()
            .expect("encode mismatched solver mesh"),
    )
    .expect("write mismatched solver mesh");

    let error = analysis_run_linear_static_with_options(
        &model,
        ComputeBackend::Cpu,
        AnalysisRunOptions {
            solver_mesh_artifact_path: Some(mesh_path.display().to_string()),
            ..AnalysisRunOptions::default()
        },
        OperationContext::new(None, None),
    )
    .expect_err("stale solver mesh should be rejected");

    assert_eq!(
        error.error_code,
        "RM.FEA.RUN_LINEAR_STATIC.SOLVER_MESH_INVALID"
    );
    assert_eq!(
        error.context.get("model_geometry_revision"),
        Some(&model.geometry_revision.to_string())
    );
    assert_eq!(
        error.context.get("solver_mesh_geometry_revision"),
        Some(&(u64::from(model.geometry_revision) + 1).to_string())
    );
    let _ = fs::remove_dir_all(mesh_root);
}

#[test]
fn ilu_preconditioner_provenance_reports_the_executed_ilu_solver() {
    let _guard = analysis_test_guard();
    let (mesh_root, mesh_path) = write_ready_minimal_analysis_mesh_artifact("preconditioner-ilu");
    let model = sample_solid_model();
    let envelope = analysis_run_linear_static_with_options(
        &model,
        ComputeBackend::Cpu,
        AnalysisRunOptions {
            deterministic_mode: true,
            precision_mode: PrecisionMode::Fp64,
            preconditioner_mode: PreconditionerMode::Ilu,
            quality_policy: QualityPolicy::Balanced,
            solver_mesh_artifact_path: Some(mesh_path.display().to_string()),
        },
        OperationContext::new(Some("trace-preconditioner-ilu".to_string()), None),
    )
    .expect("run should succeed");

    assert_eq!(
        envelope.data.provenance.preconditioner,
        SolverPreconditioner::Ilu0
    );
    assert_eq!(
        envelope.data.provenance.solver_method,
        SolverMethod::AssembledPcg
    );
    assert!(envelope.data.provenance.fallback_events.is_empty());
    assert!(envelope.data.run.diagnostics.iter().any(|diagnostic| {
        diagnostic.code == "FEA_SOLVER_METHOD"
            && diagnostic
                .message
                .contains("solver=pcg preconditioner=ilu0")
    }));
    let _ = fs::remove_dir_all(&mesh_root);
}

#[test]
fn quality_policy_exploratory_allows_numerically_acceptable_warn_path() {
    let _guard = analysis_test_guard();
    let model = sample_model_with_material_assignment_mismatch();
    let envelope = analysis_run_linear_static_with_options(
        &model,
        ComputeBackend::Cpu,
        AnalysisRunOptions {
            deterministic_mode: true,
            precision_mode: PrecisionMode::Fp64,
            preconditioner_mode: PreconditionerMode::Auto,
            quality_policy: QualityPolicy::Exploratory,
            solver_mesh_artifact_path: None,
        },
        OperationContext::new(Some("trace-quality-policy-exploratory".to_string()), None),
    )
    .expect("run should succeed");

    assert!(envelope.data.numerically_acceptable);
    assert_eq!(envelope.data.run_status, RunStatus::Completed);
    assert!(envelope
        .data
        .quality_reasons
        .iter()
        .any(|reason| reason.code == QualityReasonCode::MaterialAssignmentConflict));
    assert_eq!(envelope.data.provenance.quality_policy, "exploratory");
}

#[test]
fn direct_solid_run_without_solver_mesh_fails_closed() {
    let _guard = analysis_test_guard();
    let err = analysis_run_linear_static_with_options(
        &sample_solid_model(),
        ComputeBackend::Cpu,
        AnalysisRunOptions {
            deterministic_mode: true,
            precision_mode: PrecisionMode::Fp64,
            preconditioner_mode: PreconditionerMode::Auto,
            quality_policy: QualityPolicy::Balanced,
            solver_mesh_artifact_path: None,
        },
        OperationContext::new(
            Some("trace-direct-solid-run-requires-solver-mesh".to_string()),
            None,
        ),
    )
    .expect_err("direct solid solve must require a solver mesh");

    assert_eq!(
        err.error_code,
        "RM.FEA.RUN_LINEAR_STATIC.SOLVER_MODEL_INVALID"
    );
    assert!(err.message.contains("require a canonical solver mesh"));
}

#[test]
fn analysis_run_modal_rejects_models_without_modal_step() {
    let _guard = analysis_test_guard();
    let model = sample_model();
    let err = analysis_run_modal_with_options_op(
        &model,
        ComputeBackend::Cpu,
        Default::default(),
        OperationContext::new(None, None),
    )
    .expect_err("modal run should fail for missing modal step");

    assert_eq!(err.operation, "fea.run_modal");
    assert_eq!(err.op_version, "fea.run_modal/v1");
    assert_eq!(err.error_code, "RM.FEA.RUN_MODAL.INVALID_MODEL");
}

#[test]
fn analysis_run_acoustic_rejects_models_without_acoustic_harmonic_step() {
    let _guard = analysis_test_guard();
    let model = sample_model();
    let err = analysis_run_acoustic_with_options_op(
        &model,
        ComputeBackend::Cpu,
        Default::default(),
        OperationContext::new(None, None),
    )
    .expect_err("acoustic run should fail for missing acoustic harmonic step marker");

    assert_eq!(err.operation, "fea.run_acoustic");
    assert_eq!(err.op_version, "fea.run_acoustic/v1");
    assert_eq!(err.error_code, "RM.FEA.RUN_ACOUSTIC.INVALID_MODEL");
}

#[test]
fn analysis_run_acoustic_rejects_models_without_acoustic_source() {
    let _guard = analysis_test_guard();
    let geometry = sample_geometry_asset();
    let mut acoustic_model = test_model_for_geometry(
        &geometry,
        TestModelSpec {
            model_id: "acoustic_model_missing_source".to_string(),
            kind: TestModelKind::AcousticHarmonic,
        },
        OperationContext::new(None, None),
    )
    .expect("acoustic model should be created")
    .data;
    acoustic_model.signals.clear();
    acoustic_model.excitations.clear();
    acoustic_model.excitation_cases.clear();
    append_constant_excitation(
        &mut acoustic_model,
        "load_structural_force_not_acoustic_source",
        "region_default",
        ConstantExcitation::Force([1.0, 0.0, 0.0]),
    );

    let err = analysis_run_acoustic_with_options_op(
        &acoustic_model,
        ComputeBackend::Cpu,
        Default::default(),
        OperationContext::new(None, None),
    )
    .expect_err("acoustic run should reject models without acoustic pressure sources");

    assert_eq!(
        err.error_code,
        "RM.FEA.RUN_ACOUSTIC.MISSING_ACOUSTIC_SOURCE"
    );
}

#[test]
fn analysis_run_acoustic_rejects_moment_excitations() {
    let _guard = analysis_test_guard();
    let geometry = sample_geometry_asset();
    let mut acoustic_model = test_model_for_geometry(
        &geometry,
        TestModelSpec {
            model_id: "acoustic_model_with_moment".to_string(),
            kind: TestModelKind::AcousticHarmonic,
        },
        OperationContext::new(None, None),
    )
    .expect("acoustic model should be created")
    .data;
    replace_with_moment_excitation(&mut acoustic_model, "acoustic_moment");

    let err = analysis_run_acoustic_with_options_op(
        &acoustic_model,
        ComputeBackend::Cpu,
        Default::default(),
        OperationContext::new(None, None),
    )
    .expect_err("acoustic run should reject structural moment excitations");

    assert_eq!(
        err.error_code,
        "RM.FEA.RUN_ACOUSTIC.INVALID_ACOUSTIC_SOURCE"
    );
}

#[test]
fn analysis_run_acoustic_rejects_models_without_acoustic_boundary() {
    let _guard = analysis_test_guard();
    let geometry = sample_geometry_asset();
    let mut acoustic_model = test_model_for_geometry(
        &geometry,
        TestModelSpec {
            model_id: "acoustic_model_missing_boundary".to_string(),
            kind: TestModelKind::AcousticHarmonic,
        },
        OperationContext::new(None, None),
    )
    .expect("acoustic model should be created")
    .data;
    acoustic_model.boundaries.clear();
    acoustic_model.constraints = vec![fixed_constraint(
        "bc_structural_fixed_not_acoustic_boundary",
        "region_default",
    )];

    let err = analysis_run_acoustic_with_options_op(
        &acoustic_model,
        ComputeBackend::Cpu,
        Default::default(),
        OperationContext::new(None, None),
    )
    .expect_err("acoustic run should reject models without acoustic boundary conditions");

    assert_eq!(
        err.error_code,
        "RM.FEA.RUN_ACOUSTIC.MISSING_ACOUSTIC_BOUNDARY"
    );
}

#[test]
fn analysis_run_transient_rejects_models_without_transient_step() {
    let _guard = analysis_test_guard();
    let model = sample_model();
    let err = analysis_run_transient_with_options_op(
        &model,
        ComputeBackend::Cpu,
        Default::default(),
        OperationContext::new(None, None),
    )
    .expect_err("transient run should fail for missing transient step");

    assert_eq!(err.operation, "fea.run_transient");
    assert_eq!(err.op_version, "fea.run_transient/v1");
    assert_eq!(err.error_code, "RM.FEA.RUN_TRANSIENT.INVALID_MODEL");
}

#[test]
fn analysis_run_cfd_rejects_models_without_cfd_step() {
    let _guard = analysis_test_guard();
    let model = sample_model();
    let err = analysis_run_cfd_with_options_op(
        &model,
        ComputeBackend::Cpu,
        Default::default(),
        OperationContext::new(None, None),
    )
    .expect_err("cfd run should fail for missing cfd step");

    assert_eq!(err.operation, "fea.run_cfd");
    assert_eq!(err.op_version, "fea.run_cfd/v1");
    assert_eq!(err.error_code, "RM.FEA.RUN_CFD.AUTHORED_STUDY_REQUIRED");
}

#[test]
fn analysis_run_cfd_rejects_model_without_cfd_domain() {
    let _guard = analysis_test_guard();
    let mut model = sample_model();
    model.sequence.steps[0] = ProcedureStep::fluid_static("cfd_test");
    let err = analysis_run_cfd_with_options_op(
        &model,
        ComputeBackend::Cpu,
        Default::default(),
        OperationContext::new(None, None),
    )
    .expect_err("cfd run should fail when cfd domain is missing");

    assert_eq!(err.operation, "fea.run_cfd");
    assert_eq!(err.op_version, "fea.run_cfd/v1");
    assert_eq!(err.error_code, "RM.FEA.RUN_CFD.AUTHORED_STUDY_REQUIRED");
}

#[test]
fn analysis_run_cfd_rejects_moment_excitations() {
    let _guard = analysis_test_guard();
    let mut model = sample_model();
    model.sequence.steps[0] = ProcedureStep::fluid_static("cfd_test");
    model.cfd = Some(sample_cfd_domain());
    replace_with_moment_excitation(&mut model, "cfd_moment");

    let err = analysis_run_cfd_with_options_op(
        &model,
        ComputeBackend::Cpu,
        Default::default(),
        OperationContext::new(None, None),
    )
    .expect_err("cfd run should reject structural moment excitations");

    assert_eq!(err.operation, "fea.run_cfd");
    assert_eq!(err.op_version, "fea.run_cfd/v1");
    assert_eq!(err.error_code, "RM.FEA.RUN_CFD.AUTHORED_STUDY_REQUIRED");
}

#[test]
fn analysis_run_thermal_rejects_models_without_thermal_step() {
    let _guard = analysis_test_guard();
    let model = sample_model();
    let err = analysis_run_thermal_with_options_op(
        &model,
        ComputeBackend::Cpu,
        Default::default(),
        OperationContext::new(None, None),
    )
    .expect_err("thermal run should fail for missing thermal step");

    assert_eq!(err.operation, "fea.run_thermal");
    assert_eq!(err.op_version, "fea.run_thermal/v1");
    assert_eq!(err.error_code, "RM.FEA.RUN_THERMAL.INVALID_MODEL");
}

#[test]
fn analysis_run_thermal_rejects_invalid_thermal_material_values() {
    let _guard = analysis_test_guard();
    let mut model = sample_model();
    model.sequence.steps = vec![ProcedureStep::thermal_transient(
        "thermal_invalid_material".to_string(),
    )];
    model.materials[0]
        .thermal
        .as_mut()
        .expect("thermal fixture material")
        .conductivity_w_per_mk = runmat_types::physical::ThermalConductivity::new(0.0);

    let err = analysis_run_thermal_with_options_op(
        &model,
        ComputeBackend::Cpu,
        Default::default(),
        OperationContext::new(None, None),
    )
    .expect_err("thermal run should reject nonpositive thermal material values");

    assert_eq!(
        err.error_code,
        "RM.FEA.RUN_THERMAL.INVALID_THERMAL_MATERIAL"
    );
}

#[test]
fn analysis_run_thermal_rejects_invalid_heat_source_values() {
    let _guard = analysis_test_guard();
    let mut model = sample_model();
    model.sequence.steps = vec![ProcedureStep::thermal_transient(
        "thermal_invalid_source".to_string(),
    )];
    model.signals.clear();
    model.excitations.clear();
    model.excitation_cases.clear();
    append_constant_excitation(
        &mut model,
        "load_invalid_heat_source",
        "thermal_core",
        ConstantExcitation::HeatGeneration(f64::INFINITY),
    );

    let err = analysis_run_thermal_with_options_op(
        &model,
        ComputeBackend::Cpu,
        Default::default(),
        OperationContext::new(None, None),
    )
    .expect_err("thermal run should reject non-finite heat source values");

    assert_eq!(err.error_code, "RM.FEA.RUN_THERMAL.SOLVER_MODEL_INVALID");
    assert!(err.message.contains("signal"));
}

#[test]
fn analysis_run_thermal_rejects_moment_excitations() {
    let _guard = analysis_test_guard();
    let mut model = sample_model();
    model.sequence.steps = vec![ProcedureStep::thermal_transient(
        "thermal_moment".to_string(),
    )];
    replace_with_moment_excitation(&mut model, "thermal_moment");

    let err = analysis_run_thermal_with_options_op(
        &model,
        ComputeBackend::Cpu,
        Default::default(),
        OperationContext::new(None, None),
    )
    .expect_err("thermal run should reject structural moment excitations");

    assert_eq!(err.error_code, "RM.FEA.RUN_THERMAL.INVALID_THERMAL_SOURCE");
}

#[test]
fn analysis_run_thermal_rejects_invalid_thermal_boundary_values() {
    let _guard = analysis_test_guard();
    let mut model = sample_model();
    model.sequence.steps = vec![ProcedureStep::thermal_transient(
        "thermal_invalid_boundary".to_string(),
    )];
    model.constraints.clear();
    model.boundaries = vec![runmat_analysis_core::AnalysisBoundary {
        boundary_id: "bc_invalid_convection".to_string(),
        target: runmat_analysis_core::RegionTarget::boundary(
            runmat_meshing_core::fixtures::fixture_face("thermal_open_wall"),
        ),
        operator: runmat_analysis_core::BoundaryOperator::ThermalConvection {
            ambient_temperature: runmat_analysis_core::PrescribedValue::Constant(
                runmat_analysis_core::Temperature::new(293.15),
            ),
            coefficient: runmat_analysis_core::HeatTransferCoefficient::new(f64::NAN),
        },
    }];

    let err = analysis_run_thermal_with_options_op(
        &model,
        ComputeBackend::Cpu,
        Default::default(),
        OperationContext::new(None, None),
    )
    .expect_err("thermal run should reject non-finite thermal boundary values");

    assert_eq!(
        err.error_code,
        "RM.FEA.RUN_THERMAL.INVALID_THERMAL_BOUNDARY"
    );
}

#[test]
fn analysis_run_cht_requires_an_authored_study() {
    let _guard = analysis_test_guard();
    let mut thermal_only = sample_model();
    thermal_only.sequence.steps = vec![ProcedureStep::thermal_transient("thermal_1".to_string())];
    let mut moment = sample_cht_model();
    replace_with_moment_excitation(&mut moment, "cht_moment");
    for model in [sample_cht_model(), thermal_only, moment] {
        let err = analysis_run_cht_op(&model, OperationContext::new(None, None))
            .expect_err("the removed reduced CHT endpoint must not fabricate a coupled run");
        assert_eq!(err.operation, "fea.run_cht");
        assert_eq!(err.op_version, "fea.run_cht/v1");
        assert_eq!(err.error_code, "RM.FEA.RUN_CHT.AUTHORED_STUDY_REQUIRED");
    }
}

#[test]
fn analysis_run_fsi_requires_an_authored_study() {
    let _guard = analysis_test_guard();
    let mut fluid_only = sample_model();
    fluid_only.sequence.steps = vec![ProcedureStep::fluid_static("cfd_1".to_string())];
    fluid_only.cfd = Some(sample_cfd_domain());
    let mut moment = sample_fsi_model();
    replace_with_moment_excitation(&mut moment, "fsi_moment");
    for model in [sample_fsi_model(), fluid_only, moment] {
        let err = analysis_run_fsi_op(&model, OperationContext::new(None, None))
            .expect_err("the removed reduced FSI endpoint must not fabricate a coupled run");
        assert_eq!(err.operation, "fea.run_fsi");
        assert_eq!(err.op_version, "fea.run_fsi/v1");
        assert_eq!(err.error_code, "RM.FEA.RUN_FSI.AUTHORED_STUDY_REQUIRED");
    }
}

#[test]
fn analysis_run_thermal_returns_temperature_payload() {
    let _guard = analysis_test_guard();
    let (mesh_root, mesh_path) = write_ready_minimal_analysis_mesh_artifact("thermal-payload");
    let mut model = sample_solid_model();
    model.sequence.steps = vec![ProcedureStep::thermal_transient("thermal_1".to_string())];
    model.thermal = Some(runmat_analysis_core::ThermalDomain {
        reference_temperature: Temperature::new(293.15),
    });
    let run = analysis_run_thermal_with_options_op(
        &model,
        ComputeBackend::Cpu,
        AnalysisThermalRunOptions {
            snapshot_count: 6,
            solver_mesh_artifact_path: Some(mesh_path.display().to_string()),
            ..AnalysisThermalRunOptions::default()
        },
        OperationContext::new(None, None),
    )
    .expect("thermal run should succeed");

    assert_eq!(run.operation, "fea.run_thermal");
    assert_eq!(run.op_version, "fea.run_thermal/v1");
    assert!(run.data.thermal_results.is_some());
    assert!(run.data.transient_results.is_none());
    assert!(!run
        .data
        .run
        .diagnostics
        .iter()
        .any(|diag| diag.code == "FEA_THERMAL_ENERGY_STORAGE_CONSISTENCY"));
    assert!(run.data.run.diagnostics.iter().any(|diag| {
        diag.code == "FEA_THERMAL_FIELD_RECOVERY"
            && diag.message.contains("basis=physical_quadrature")
            && diag.message.contains("boundary_face_count=4")
    }));
    assert!(!run
        .data
        .run
        .diagnostics
        .iter()
        .any(|diag| diag.code == "FEA_THERMAL_KNOWN_ANSWER"));
    let results = analysis_results_op(
        &run.data,
        AnalysisResultsQuery::default(),
        OperationContext::new(None, None),
    )
    .expect("fea.results should return thermal payload");
    let thermal = results
        .data
        .thermal_results
        .as_ref()
        .expect("thermal results must be present");
    assert_eq!(thermal.time_points_s.len(), 6);
    assert_eq!(thermal.temperature_snapshots.len(), 6);
    assert_eq!(thermal.temperature_gradient_snapshots.len(), 6);
    assert_eq!(thermal.heat_flux_snapshots.len(), 6);
    assert_eq!(thermal.energy_storage_rate_snapshots.len(), 6);
    let initial_state = &thermal.initial_state;
    assert_eq!(initial_state.procedure_step_id, "thermal_1");
    assert_eq!(initial_state.reference_node_count, 4);
    assert_eq!(thermal.temperature_gradient_snapshots[0].shape.len(), 2);
    assert_eq!(thermal.temperature_gradient_snapshots[0].shape[1], 3);
    assert_eq!(
        thermal.heat_flux_snapshots[0].shape,
        thermal.temperature_gradient_snapshots[0].shape
    );
    assert_eq!(
        thermal.energy_storage_rate_snapshots[0].shape,
        vec![thermal.temperature_gradient_snapshots[0].shape[0]]
    );
    let field_ids = results
        .data
        .field_descriptors
        .iter()
        .map(|descriptor| descriptor.field_id.as_str())
        .collect::<Vec<_>>();
    assert!(field_ids.contains(&fea_thermal_temperature_gradient_field_id(0).as_str()));
    assert!(field_ids.contains(&fea_thermal_heat_flux_field_id(0).as_str()));
    assert!(field_ids.contains(&fea_thermal_energy_storage_rate_field_id(0).as_str()));
    let descriptor = |field_id: &str| {
        results
            .data
            .field_descriptors
            .iter()
            .find(|descriptor| descriptor.field_id == field_id)
            .expect("thermal field descriptor should be present")
    };
    assert_eq!(
        descriptor(&fea_thermal_heat_flux_field_id(0)).kind,
        AnalysisFieldKind::Vector
    );
    assert_eq!(
        descriptor(&fea_thermal_heat_flux_field_id(0)).component_count,
        Some(3)
    );
    assert_eq!(results.data.summary.snapshot_count, 6);
    let _ = fs::remove_dir_all(mesh_root);
}

#[test]
fn analysis_run_thermal_reports_host_execution_for_gpu_request() {
    let _guard = analysis_test_guard();
    let _provider_guard = runmat_accelerate_api::ThreadProviderGuard::set(None);
    let (mesh_root, mesh_path) =
        write_ready_minimal_analysis_mesh_artifact("thermal-gpu-provenance");
    let mut model = sample_solid_model();
    model.sequence.steps = vec![ProcedureStep::thermal_transient(
        "thermal_execution_provenance",
    )];
    model.thermal = Some(runmat_analysis_core::ThermalDomain {
        reference_temperature: Temperature::new(293.15),
    });

    let run = analysis_run_thermal_with_options_op(
        &model,
        ComputeBackend::Gpu,
        AnalysisThermalRunOptions {
            solver_mesh_artifact_path: Some(mesh_path.display().to_string()),
            ..AnalysisThermalRunOptions::default()
        },
        OperationContext::new(None, None),
    )
    .expect("thermal run should complete");

    assert_eq!(run.data.provenance.requested_backend, ComputeBackend::Gpu);
    assert_eq!(run.data.provenance.execution_backend, ExecutionBackend::Cpu);
    assert_eq!(run.data.run.execution_backend, ExecutionBackend::Cpu);
    assert_eq!(
        run.data.provenance.solver_backend,
        LinearAlgebraBackendKind::CpuReference
    );
    assert!(run
        .data
        .provenance
        .fallback_events
        .iter()
        .any(FallbackEvent::is_solver_backend_fallback));
    let _ = fs::remove_dir_all(mesh_root);
}

#[test]
fn analysis_run_thermal_balanced_degrades_on_high_constitutive_spread() {
    let _guard = analysis_test_guard();
    let mesh = runmat_meshing_core::fixtures::three_region_tetrahedron_solver_mesh();
    let (mesh_root, mesh_path) =
        write_ready_analysis_mesh_artifact("thermal-constitutive-spread", &mesh);
    let mut model = sample_solid_model();
    model.sequence.steps = vec![ProcedureStep::thermal_transient("thermal_1".to_string())];
    model.thermal = Some(runmat_analysis_core::ThermalDomain {
        reference_temperature: Temperature::new(293.15),
    });
    model.materials.push(MaterialModel {
        fluid: None,
        material_id: "mat_poly_high_k".to_string(),
        name: "High K Composite".to_string(),
        provenance: runmat_analysis_core::MaterialProvenance::Authored,
        mass_density_kg_per_m3: Some(runmat_types::physical::MassDensity::new(1800.0)),
        elasticity: Some(MaterialElasticityModel::IsotropicLinear(
            MaterialMechanicalModel {
                youngs_modulus_pa: runmat_types::physical::YoungsModulus::new(5.0e9),
                poisson_ratio: runmat_types::physical::PoissonRatio::new(0.33),
            },
        )),
        damping: None,
        thermal: Some(MaterialThermalModel {
            reference_temperature_k: runmat_types::physical::Temperature::new(293.15),
            modulus_temp_coeff_per_k: runmat_types::physical::InverseTemperature::new(-2.5e-4),
            conductivity_w_per_mk: runmat_types::physical::ThermalConductivity::new(1200.0),
            specific_heat_j_per_kgk: runmat_types::physical::SpecificHeat::new(160.0),
            expansion_coefficient_per_k: runmat_types::physical::ThermalExpansion::new(1.2e-5),
        }),
        acoustic: None,
        electrical: None,
        plastic: None,
    });
    model.material_assignments = [
        ("mid_aluminum", "mat_steel"),
        ("polymer_segment", "mat_poly_high_k"),
        ("tip_steel", "mat_steel"),
    ]
    .into_iter()
    .map(|(region, material)| MaterialAssignment {
        target: runmat_analysis_core::PropertyRegionTarget::solid(
            runmat_meshing_core::fixtures::fixture_region(region),
        ),
        expected_material_id: material.into(),
        assigned_material_id: material.into(),
        confidence: EvidenceConfidence::Verified,
    })
    .collect();

    let run = analysis_run_thermal_with_options_op(
        &model,
        ComputeBackend::Cpu,
        AnalysisThermalRunOptions {
            quality_policy: QualityPolicy::Balanced,
            snapshot_count: 8,
            solver_mesh_artifact_path: Some(mesh_path.display().to_string()),
            ..AnalysisThermalRunOptions::default()
        },
        OperationContext::new(None, None),
    )
    .expect("thermal run should execute");

    assert_eq!(mesh.topology.volume_elements.len(), 3);
    let conductivity_spread = diagnostic_metric(
        &run.data.run.diagnostics,
        "FEA_THERMAL_CONSTITUTIVE",
        "conductivity_spread_ratio",
    )
    .expect("active heterogeneous materials should report conductivity spread");
    assert!((conductivity_spread - 1200.0 / 45.0).abs() < 1.0e-12);
    assert!(!run.data.numerically_acceptable);
    assert_eq!(run.data.run_status, RunStatus::Degraded);
    assert!(run
        .data
        .quality_reasons
        .iter()
        .any(|reason| reason.code == QualityReasonCode::ThermalConstitutiveSpreadHigh));
    let _ = fs::remove_dir_all(mesh_root);
}

#[test]
fn analysis_run_nonlinear_rejects_models_without_nonlinear_step() {
    let _guard = analysis_test_guard();
    let model = sample_model();
    let err = analysis_run_nonlinear_with_options_op(
        &model,
        ComputeBackend::Cpu,
        Default::default(),
        OperationContext::new(None, None),
    )
    .expect_err("nonlinear run should reject models without nonlinear step");
    assert_eq!(err.operation, "fea.run_nonlinear");
    assert_eq!(err.op_version, "fea.run_nonlinear/v1");
    assert_eq!(err.error_code, "RM.FEA.RUN_NONLINEAR.INVALID_MODEL");
}

#[test]
fn analysis_run_nonlinear_returns_native_nonlinear_result_rejects_unsupported_envelope() {
    let _guard = analysis_test_guard();
    let mut model = sample_model();
    model.sequence.steps = vec![ProcedureStep::structural_nonlinear("unsupported_nonlinear")];
    let error = analysis_run_nonlinear_with_options_op(
        &model,
        ComputeBackend::Cpu,
        AnalysisNonlinearRunOptions::balanced(),
        OperationContext::new(None, None),
    )
    .expect_err("an unsupported envelope must not publish nonlinear snapshots or quality metrics");
    assert_eq!(
        error.error_code,
        "RM.FEA.RUN_NONLINEAR.SOLVER_MODEL_INVALID"
    );
    assert!(!error.retryable);
}

#[test]
fn analysis_run_nonlinear_exposes_beam_rotation_snapshots_rejects_unsupported_envelope() {
    let _guard = analysis_test_guard();
    let mut model = runmat_analysis_fea::fixtures::fixture_model(
        runmat_analysis_fea::fixtures::FixtureId::StructuralBeamCantileverEndMomentReference,
    );
    model.sequence.steps = vec![ProcedureStep::structural_nonlinear("unsupported_nonlinear")];
    let error = analysis_run_nonlinear_with_options_op(
        &model,
        ComputeBackend::Cpu,
        AnalysisNonlinearRunOptions::balanced(),
        OperationContext::new(None, None),
    )
    .expect_err("an unsupported envelope must not publish nonlinear snapshots or quality metrics");
    assert_eq!(
        error.error_code,
        "RM.FEA.RUN_NONLINEAR.SOLVER_MODEL_INVALID"
    );
    assert!(!error.retryable);
}

#[test]
fn nonlinear_strict_iteration_controls_do_not_admit_an_unsupported_envelope() {
    let _guard = analysis_test_guard();
    let mut model = sample_model();
    model.sequence.steps = vec![ProcedureStep::structural_nonlinear("unsupported_nonlinear")];
    let error = analysis_run_nonlinear_with_options_op(
        &model,
        ComputeBackend::Cpu,
        AnalysisNonlinearRunOptions {
            quality_policy: QualityPolicy::Strict,
            max_newton_iters: 1,
            ..AnalysisNonlinearRunOptions::balanced()
        },
        OperationContext::new(None, None),
    )
    .expect_err("an unsupported envelope must not publish nonlinear snapshots or quality metrics");
    assert_eq!(
        error.error_code,
        "RM.FEA.RUN_NONLINEAR.SOLVER_MODEL_INVALID"
    );
    assert!(!error.retryable);
}

#[test]
fn nonlinear_quality_policy_cannot_override_formulation_admission() {
    let _guard = analysis_test_guard();
    let mut model = sample_model();
    model.sequence.steps = vec![ProcedureStep::structural_nonlinear("unsupported_nonlinear")];
    for quality_policy in [
        QualityPolicy::Strict,
        QualityPolicy::Balanced,
        QualityPolicy::Exploratory,
    ] {
        let error = analysis_run_nonlinear_with_options_op(
            &model,
            ComputeBackend::Cpu,
            AnalysisNonlinearRunOptions {
                quality_policy,
                ..AnalysisNonlinearRunOptions::balanced()
            },
            OperationContext::new(None, None),
        )
        .expect_err(
            "an unsupported envelope must not publish nonlinear snapshots or quality metrics",
        );
        assert_eq!(
            error.error_code,
            "RM.FEA.RUN_NONLINEAR.SOLVER_MODEL_INVALID"
        );
        assert!(!error.retryable);
    }
}

#[test]
fn nonlinear_balanced_degrades_when_thermo_mechanical_severity_is_high_rejects_unsupported_envelope(
) {
    let _guard = analysis_test_guard();
    let mut model = sample_solid_model();
    model.sequence.steps = vec![ProcedureStep::coupled(
        "unsupported_coupled",
        CoupledPhysicsFamily::Thermomechanical,
        AnalysisProcedure::Nonlinear,
    )];
    let error = analysis_run_nonlinear_with_options_op(
        &model,
        ComputeBackend::Cpu,
        AnalysisNonlinearRunOptions::balanced(),
        OperationContext::new(None, None),
    )
    .expect_err("an unsupported envelope must not publish nonlinear snapshots or quality metrics");
    assert_eq!(
        error.error_code,
        "RM.FEA.RUN_NONLINEAR.FORMULATION_NOT_ADMITTED"
    );
    assert!(!error.retryable);
}

#[test]
fn nonlinear_balanced_does_not_report_thermo_stress_below_severity_threshold_rejects_unsupported_envelope(
) {
    let _guard = analysis_test_guard();
    let mut model = sample_solid_model();
    model.sequence.steps = vec![ProcedureStep::coupled(
        "unsupported_coupled",
        CoupledPhysicsFamily::Thermomechanical,
        AnalysisProcedure::Nonlinear,
    )];
    let error = analysis_run_nonlinear_with_options_op(
        &model,
        ComputeBackend::Cpu,
        AnalysisNonlinearRunOptions::balanced(),
        OperationContext::new(None, None),
    )
    .expect_err("an unsupported envelope must not publish nonlinear snapshots or quality metrics");
    assert_eq!(
        error.error_code,
        "RM.FEA.RUN_NONLINEAR.FORMULATION_NOT_ADMITTED"
    );
    assert!(!error.retryable);
}

#[test]
fn analysis_results_query_can_exclude_nonlinear_payload_rejects_unsupported_envelope() {
    let _guard = analysis_test_guard();
    let mut model = sample_model();
    model.sequence.steps = vec![ProcedureStep::structural_nonlinear("unsupported_nonlinear")];
    let error = analysis_run_nonlinear_with_options_op(
        &model,
        ComputeBackend::Cpu,
        AnalysisNonlinearRunOptions::balanced(),
        OperationContext::new(None, None),
    )
    .expect_err("an unsupported envelope must not publish nonlinear snapshots or quality metrics");
    assert_eq!(
        error.error_code,
        "RM.FEA.RUN_NONLINEAR.SOLVER_MODEL_INVALID"
    );
    assert!(!error.retryable);
}

#[test]
fn nonlinear_results_deserialize_with_missing_new_fields() {
    let payload = serde_json::json!({
        "nonlinear_payload_version": "nonlinear_results/v1",
        "load_factors": [0.5, 1.0],
        "displacement_snapshots": [],
        "residual_norms": [1.0e-6, 5.0e-7],
        "method": "incremental_newton_raphson"
    });
    let parsed: NonlinearResultsData =
        serde_json::from_value(payload).expect("legacy nonlinear payload should deserialize");

    assert_eq!(parsed.increment_norms.len(), 0);
    assert_eq!(parsed.iteration_counts.len(), 0);
    assert_eq!(parsed.failed_increments, 0);
    assert_eq!(parsed.line_search_backtracks, 0);
    assert_eq!(parsed.max_line_search_backtracks_per_increment, 0);
    assert_eq!(parsed.tangent_rebuild_count, 0);
    assert_eq!(parsed.iteration_spike_count, 0);
    assert_eq!(parsed.convergence_stall_count, 0);
    assert_eq!(parsed.backtrack_burst_count, 0);
    assert!(parsed.von_mises_snapshots.is_empty());
    assert!(parsed.plastic_strain_snapshots.is_empty());
    assert!(parsed.equivalent_plastic_strain_snapshots.is_empty());
    assert!(parsed.contact_pressure_snapshots.is_empty());
    assert!(parsed.contact_gap_snapshots.is_empty());
}

#[test]
fn analysis_run_transient_returns_native_transient_result() {
    let _guard = analysis_test_guard();
    let mut model = sample_model();
    model.sequence.steps = vec![ProcedureStep::structural_transient(
        "transient_1".to_string(),
    )];

    let envelope = analysis_run_transient_with_options_op(
        &model,
        ComputeBackend::Cpu,
        Default::default(),
        OperationContext::new(None, None),
    )
    .expect("transient run should return envelope");

    assert_eq!(envelope.operation, "fea.run_transient");
    assert_eq!(envelope.op_version, "fea.run_transient/v1");
    assert_eq!(
        envelope.data.run.solver_method,
        SolverMethod::NewmarkAverageAccelerationPcg
    );
    assert_eq!(
        envelope.data.provenance.solver_method,
        SolverMethod::NewmarkAverageAccelerationPcg
    );
    assert_eq!(envelope.data.provenance.precision_mode, "fp64");
    assert_eq!(envelope.data.run_status, RunStatus::Completed);
    assert!(envelope.data.numerically_acceptable);
    assert!(envelope
        .data
        .run
        .diagnostics
        .iter()
        .any(|diag| diag.code == "FEA_TRANSIENT_CONVERGENCE"));
    let transient = envelope
        .data
        .transient_results
        .as_ref()
        .expect("transient payload should exist");
    assert_eq!(
        transient.integration_method,
        TransientIntegrationMethod::NewmarkAverageAcceleration
    );
    assert!(!transient.time_points_s.is_empty());
    assert_eq!(
        transient.time_points_s.len(),
        transient.displacement_snapshots.len()
    );
    assert_eq!(
        transient.time_points_s.len(),
        transient.velocity_snapshots.len()
    );
    assert_eq!(
        transient.time_points_s.len(),
        transient.acceleration_snapshots.len()
    );
    assert!(transient.strain_snapshots.is_empty());
    assert!(transient.stress_snapshots.is_empty());
    assert!(transient.strain_energy_density_snapshots.is_empty());
    assert!(transient.von_mises_snapshots.is_empty());
    assert_eq!(
        transient.time_points_s.len(),
        transient.kinetic_energy_snapshots.len()
    );
    assert_eq!(
        transient.time_points_s.len(),
        transient.strain_energy_snapshots.len()
    );
    assert_eq!(
        transient.time_points_s.len(),
        transient.residual_norm_snapshots.len()
    );
    assert_eq!(
        transient.time_points_s.len(),
        transient.reaction_force_snapshots.len()
    );
    assert_eq!(
        transient.time_points_s.len(),
        transient.external_work_snapshots.len()
    );
    assert_eq!(
        transient.time_points_s.len(),
        transient.support_work_snapshots.len()
    );
    assert_eq!(
        transient.time_points_s.len(),
        transient.damping_dissipation_snapshots.len()
    );
    assert_eq!(
        transient.time_points_s.len(),
        transient.energy_balance_residual_snapshots.len()
    );
    assert_eq!(
        transient.time_points_s.len(),
        transient.relative_energy_balance_error_snapshots.len()
    );
    assert_eq!(
        transient.time_points_s.len(),
        transient.dynamic_balance_snapshots.len()
    );
    assert_eq!(
        transient.velocity_snapshots[1].field_id,
        fea_transient_velocity_field_id(1)
    );
    assert_eq!(
        transient.acceleration_snapshots[1].field_id,
        fea_transient_acceleration_field_id(1)
    );
    assert_eq!(
        transient.kinetic_energy_snapshots[1].field_id,
        fea_transient_kinetic_energy_field_id(1)
    );
    assert_eq!(
        transient.strain_energy_snapshots[1].field_id,
        fea_transient_strain_energy_field_id(1)
    );
    assert_eq!(
        transient.residual_norm_snapshots[1].field_id,
        fea_transient_residual_norm_field_id(1)
    );
    assert_eq!(
        transient.reaction_force_snapshots[1].field_id,
        fea_transient_reaction_force_field_id(1)
    );
    let results = analysis_results_op(
        &envelope.data,
        AnalysisResultsQuery::default(),
        OperationContext::new(None, None),
    )
    .expect("transient results should be queryable");
    let descriptor = |field_id: &str| {
        results
            .data
            .field_descriptors
            .iter()
            .find(|descriptor| descriptor.field_id == field_id)
            .expect("transient descriptor should be present")
    };
    for field_id in [
        fea_transient_velocity_field_id(1),
        fea_transient_acceleration_field_id(1),
        fea_transient_reaction_force_field_id(1),
    ] {
        let descriptor = descriptor(&field_id);
        assert_eq!(descriptor.kind, AnalysisFieldKind::Vector);
        assert_eq!(descriptor.component_count, Some(3));
    }
    for field_id in [
        fea_transient_kinetic_energy_field_id(1),
        fea_transient_strain_energy_field_id(1),
        fea_transient_residual_norm_field_id(1),
        fea_transient_external_work_field_id(1),
        fea_transient_support_work_field_id(1),
        fea_transient_damping_dissipation_field_id(1),
        fea_transient_energy_balance_residual_field_id(1),
        fea_transient_relative_energy_balance_error_field_id(1),
    ] {
        let descriptor = descriptor(&field_id);
        assert_eq!(descriptor.kind, AnalysisFieldKind::Scalar);
        assert_eq!(descriptor.component_count, None);
    }
}

#[test]
fn analysis_run_transient_exposes_beam_rotational_snapshots() {
    let _guard = analysis_test_guard();
    let mut model = runmat_analysis_fea::fixtures::fixture_model(
        runmat_analysis_fea::fixtures::FixtureId::StructuralBeamCantileverEndMomentReference,
    );
    model.sequence.steps = vec![ProcedureStep::structural_transient(
        "beam_moment_transient".to_string(),
    )];

    let envelope = analysis_run_transient_with_options_op(
        &model,
        ComputeBackend::Cpu,
        Default::default(),
        OperationContext::new(None, None),
    )
    .expect("beam transient run should return envelope");
    let transient = envelope
        .data
        .transient_results
        .as_ref()
        .expect("transient payload should exist");

    assert_eq!(
        transient.time_points_s.len(),
        transient.rotation_snapshots.len()
    );
    assert_eq!(
        transient.time_points_s.len(),
        transient.angular_velocity_snapshots.len()
    );
    assert_eq!(
        transient.time_points_s.len(),
        transient.angular_acceleration_snapshots.len()
    );
    assert_eq!(
        transient.rotation_snapshots[1].field_id,
        fea_transient_rotation_field_id(1)
    );
    assert_eq!(
        transient.angular_velocity_snapshots[1].field_id,
        fea_transient_angular_velocity_field_id(1)
    );
    assert_eq!(
        transient.angular_acceleration_snapshots[1].field_id,
        fea_transient_angular_acceleration_field_id(1)
    );

    let results = analysis_results_op(
        &envelope.data,
        AnalysisResultsQuery::default(),
        OperationContext::new(None, None),
    )
    .expect("transient beam results should be queryable");
    for field_id in [
        fea_transient_rotation_field_id(1),
        fea_transient_angular_velocity_field_id(1),
        fea_transient_angular_acceleration_field_id(1),
    ] {
        let descriptor = results
            .data
            .field_descriptors
            .iter()
            .find(|descriptor| descriptor.field_id == field_id)
            .expect("rotational transient descriptor should be present");
        assert_eq!(descriptor.kind, AnalysisFieldKind::Vector);
        assert_eq!(descriptor.component_count, Some(3));
    }
}

#[test]
fn analysis_run_cfd_rejects_vector_boundaries_without_fabricated_payload() {
    let _guard = analysis_test_guard();
    let mut model = sample_model();
    model.sequence.steps[0] = ProcedureStep::fluid_static("cfd_test");
    model.boundaries.clear();
    model.constraints = sample_cfd_constraints([4.25, 0.0, 0.0]);
    model.cfd = Some(sample_cfd_domain());
    let runmat_analysis_core::AnalysisConstraint::Fluid(inlet) = &model.constraints[0] else {
        panic!("explicit vector inlet fixture");
    };
    assert!(matches!(&inlet.kind,
        runmat_analysis_core::FluidConstraintKind::PrescribedVelocity {
            value: runmat_analysis_core::PrescribedValue::Constant(velocity), ..
        } if velocity.map(|component| component.canonical_value()) == [4.25, 0.0, 0.0]
    ));
    let err = analysis_run_cfd_with_options_op(
        &model,
        ComputeBackend::Cpu,
        AnalysisCfdRunOptions::default(),
        OperationContext::new(None, None),
    )
    .expect_err("reduced channel execution cannot publish an exact vector CFD solution");
    assert_eq!(err.operation, "fea.run_cfd");
    assert_eq!(err.op_version, "fea.run_cfd/v1");
    assert_eq!(err.error_code, "RM.FEA.RUN_CFD.AUTHORED_STUDY_REQUIRED");
    assert!(err
        .message
        .contains("canonical finite-volume execution requires an authored steady fluid study"));
}

#[test]
fn canonical_cfd_wire_options_reject_channel_controls_and_nonfluid_procedures() {
    let options = AnalysisCfdRunOptions::default();
    let encoded = serde_json::to_value(&options).unwrap();
    assert_eq!(
        serde_json::from_value::<AnalysisCfdRunOptions>(encoded.clone()).unwrap(),
        options
    );
    for removed in [
        "sample_count",
        "max_linear_iters",
        "tolerance",
        "residual_warn_threshold",
        "solver_mesh_artifact_path",
    ] {
        let mut invalid = encoded.clone();
        invalid[removed] = serde_json::json!(1);
        assert!(serde_json::from_value::<AnalysisCfdRunOptions>(invalid).is_err());
    }
    let mut step = ProcedureStep::fluid_static("flow");
    assert!(AnalysisSolverOptions::defaults_for(&step).is_ok());
    step.procedure = AnalysisProcedure::TransientResponse;
    assert!(AnalysisSolverOptions::defaults_for(&step).is_ok());
    assert!(AnalysisSolverOptions::IncompressibleFlow(options.clone()).supports(&step));
    let structural = ProcedureStep::structural_modal("not-fluid");
    assert!(!AnalysisSolverOptions::IncompressibleFlow(options).supports(&structural));
}

#[test]
fn analysis_run_cfd_mesh_counts_do_not_qualify_a_reduced_vector_solve() {
    let _guard = analysis_test_guard();
    let (mesh_root, mesh_path) = write_ready_minimal_analysis_mesh_artifact("cfd-solver-mesh");
    // Independently verify the actual source artifact's topology. Aggregate source counts
    // are not evidence that a conservative pressure-velocity operator was assembled.
    let mesh = SolverMeshArtifact::canonical_decode(&fs::read(&mesh_path).unwrap()).unwrap();
    let source = CfdDomainTopology::from_solver_mesh(&mesh);
    assert_eq!(source.control_volume_count, 1);
    assert_eq!(source.control_volume_boundary_face_count, 4);
    assert_eq!(source.control_volume_internal_face_count, 0);
    let mut model = sample_model();
    model.sequence.steps[0] = ProcedureStep::fluid_static("cfd_test");
    model.boundaries.clear();
    model.constraints = sample_cfd_constraints([4.25, 0.0, 0.0]);
    model.cfd = Some(sample_cfd_domain());
    let err = analysis_run_cfd_with_options_op(
        &model,
        ComputeBackend::Cpu,
        AnalysisCfdRunOptions::default(),
        OperationContext::new(None, None),
    )
    .expect_err("having a mesh must not turn reduced channel physics into canonical CFD");
    assert_eq!(err.error_code, "RM.FEA.RUN_CFD.AUTHORED_STUDY_REQUIRED");
    assert!(err
        .message
        .contains("canonical finite-volume execution requires an authored steady fluid study"));
    let _ = fs::remove_dir_all(mesh_root);
}

#[test]
fn canonical_boundary_faces_define_complete_tetrahedral_interface_connectivity() {
    let mesh = runmat_meshing_core::fixtures::canonical_tetrahedron_solver_mesh(ElementOrder::Tet4);
    let topology = CfdDomainTopology::from_solver_mesh(&mesh);

    let (edges, mesh_coverage) =
        runmat_analysis_fea::physics::cfd::interface_graph_edges(&topology, 4);

    assert_eq!(edges.len(), 6);
    assert_eq!(mesh_coverage, 1.0);
    assert!(edges.contains(&(0, 1)));
    assert!(edges.contains(&(2, 3)));
}

#[test]
fn analysis_run_cfd_rejects_partial_authored_boundary_conditions() {
    let _guard = analysis_test_guard();
    let mut model = sample_model();
    model.sequence.steps[0] = ProcedureStep::fluid_static("cfd_test");
    model.cfd = Some(sample_cfd_domain());
    model.boundaries.clear();
    model.constraints = vec![runmat_analysis_core::AnalysisConstraint::Fluid(
        runmat_analysis_core::FluidConstraint {
            constraint_id: "bc_cfd_inlet_only".to_string(),
            kind: runmat_analysis_core::FluidConstraintKind::PrescribedVelocity {
                target: runmat_analysis_core::RegionTarget::boundary(
                    runmat_meshing_core::fixtures::fixture_face("fluid_inlet"),
                ),
                value: runmat_analysis_core::PrescribedValue::Constant(
                    [4.25, -1.0, 0.0].map(runmat_analysis_core::Velocity::new),
                ),
            },
        },
    )];

    let err = analysis_run_cfd_with_options_op(
        &model,
        ComputeBackend::Cpu,
        Default::default(),
        OperationContext::new(None, None),
    )
    .expect_err("partial authored cfd boundaries should fail validation");

    assert_eq!(err.operation, "fea.run_cfd");
    assert_eq!(err.op_version, "fea.run_cfd/v1");
    assert_eq!(err.error_code, "RM.FEA.RUN_CFD.AUTHORED_STUDY_REQUIRED");
}

#[test]
fn analysis_run_transient_with_options_controls_timeline() {
    let _guard = analysis_test_guard();
    let mut model = sample_model();
    model.sequence.steps = vec![ProcedureStep::structural_transient(
        "transient_1".to_string(),
    )];

    let envelope = analysis_run_transient_with_options_op(
        &model,
        ComputeBackend::Cpu,
        AnalysisTransientRunOptions {
            deterministic_mode: true,
            quality_policy: QualityPolicy::Balanced,
            time_step_policy: AnalysisStructuralTimeStepPolicy::Fixed { interval_count: 3 },
            residual_target: 1.0e-6,
            maximum_intervals_per_block: 256,
            solver_mesh_artifact_path: None,
        },
        OperationContext::new(None, None),
    )
    .expect("transient run should succeed with options");

    let transient = envelope
        .data
        .transient_results
        .as_ref()
        .expect("transient payload should exist");
    assert_eq!(transient.time_points_s.len(), 4);
    assert_eq!(transient.time_points_s[0], 0.0);
    assert!((transient.time_points_s[3] - 1.0).abs() < 1.0e-12);
    assert!(envelope.data.provenance.deterministic_mode);
}

#[test]
fn analysis_run_linear_static_thermomechanical_rejects_unknown_temperature_field_region() {
    let _guard = analysis_test_guard();
    let (mesh_root, mesh_path) =
        write_ready_minimal_analysis_mesh_artifact("linear-thermo-temperature-target");
    let mut model = sample_solid_model();
    model.sequence.steps = vec![ProcedureStep::coupled(
        "linear_thermo_temperature_target",
        CoupledPhysicsFamily::Thermomechanical,
        AnalysisProcedure::Static,
    )];
    set_model_thermo_coupling(
        &mut model,
        TestThermoMechanicalCoupling {
            reference_temperature_k: runmat_types::physical::Temperature::new(293.15),
            applied_temperature_delta_k: runmat_types::physical::TemperatureDifference::new(80.0),
            thermal_expansion_coefficient: runmat_types::physical::ThermalExpansion::new(1.2e-5),
            region_temperature_deltas: vec![TestThermoRegionTemperatureDelta {
                region_id: "missing_region".to_string(),
                temperature_delta_k: 80.0,
            }],
        },
    );

    let published = storage::list_run_metadata()
        .expect("list existing runs")
        .len();
    let err = analysis_run_linear_static_with_options(
        &model,
        ComputeBackend::Cpu,
        AnalysisRunOptions {
            solver_mesh_artifact_path: Some(mesh_path.display().to_string()),
            ..Default::default()
        },
        OperationContext::new(None, None),
    )
    .expect_err("unknown thermo expected region should be rejected");

    assert_eq!(
        err.error_code,
        "RM.FEA.RUN_LINEAR_STATIC.SOLVER_MODEL_INVALID"
    );
    assert_eq!(err.error_type, OperationErrorType::Validation);
    assert!(
        err.message
            .contains("temperature field targets unknown model region missing_region"),
        "expected unknown temperature region diagnostic, got: {err:?}"
    );
    assert_eq!(
        storage::list_run_metadata()
            .expect("list runs after rejection")
            .len(),
        published,
        "invalid temperature region must not publish a run"
    );
    let _ = fs::remove_dir_all(mesh_root);
}

#[test]
fn analysis_run_nonlinear_rejects_invalid_plastic_material() {
    let _guard = analysis_test_guard();
    let mut model = sample_model();
    model.sequence.steps = vec![ProcedureStep::structural_nonlinear(
        "nonlinear_1".to_string(),
    )];
    set_model_plasticity(
        &mut model,
        runmat_analysis_core::MaterialPlasticModel::J2IsotropicLinearHardening {
            yield_stress_pa: runmat_types::physical::YieldStress::new(-1.0),
            isotropic_hardening_modulus_pa: runmat_types::physical::HardeningModulus::new(1e9),
        },
    );

    let err = analysis_run_nonlinear_with_options_op(
        &model,
        ComputeBackend::Cpu,
        AnalysisNonlinearRunOptions::default(),
        OperationContext::new(None, None),
    )
    .expect_err("nonlinear run should reject invalid authored plastic material");

    assert_eq!(err.error_type, OperationErrorType::Validation);
}

#[test]
fn nonlinear_solver_rejects_invalid_contact_enforcement_options() {
    let _guard = analysis_test_guard();
    let error = validate_contact_interface_options(&runmat_analysis_fea::FeaContactEnforcement {
        penalty_stiffness_scale: 0.0,
        max_penetration_ratio: 0.01,
    })
    .expect_err("nonlinear solver should reject an invalid contact penalty");
    assert!(error.0.contains("penalty_stiffness_scale"));
}

#[test]
fn nonlinear_policy_wire_contract_rejects_removed_proxy_controls() {
    let options = AnalysisNonlinearRunOptions::default();
    let encoded = serde_json::to_value(&options).unwrap();
    assert_eq!(encoded["linear_tolerance"], 1e-10);
    assert_eq!(encoded["max_load_cutbacks"], 8);
    assert_eq!(encoded["minimum_load_increment"], 1e-8);
    assert_eq!(encoded["max_accepted_increments"], 4096);
    assert_eq!(
        serde_json::from_value::<AnalysisNonlinearRunOptions>(encoded.clone()).unwrap(),
        options
    );
    for removed in [
        "residual_convergence_factor",
        "increment_norm_tolerance",
        "tangent_refresh_interval",
    ] {
        let mut invalid = encoded.clone();
        invalid[removed] = serde_json::json!(1);
        assert!(serde_json::from_value::<AnalysisNonlinearRunOptions>(invalid).is_err());
    }
}

#[test]
fn nonlinear_plasticity_rejects_noncanonical_synthetic_model() {
    let _guard = analysis_test_guard();
    let mut model = sample_model();
    model.sequence.steps = vec![ProcedureStep::structural_nonlinear(
        "nonlinear_1".to_string(),
    )];
    set_model_plasticity(
        &mut model,
        runmat_analysis_core::MaterialPlasticModel::J2IsotropicLinearHardening {
            yield_stress_pa: runmat_types::physical::YieldStress::new(250e6),
            isotropic_hardening_modulus_pa: runmat_types::physical::HardeningModulus::new(1e9),
        },
    );

    let error = analysis_run_nonlinear_with_options_op(
        &model,
        ComputeBackend::Cpu,
        AnalysisNonlinearRunOptions {
            quality_policy: QualityPolicy::Balanced,
            ..AnalysisNonlinearRunOptions::balanced()
        },
        OperationContext::new(None, None),
    )
    .expect_err("physical J2 must not run through a synthetic model");
    assert!(!error.retryable);
}

#[test]
fn nonlinear_contact_is_rejected_outside_the_j2_envelope() {
    let _guard = analysis_test_guard();
    let mut model = sample_model();
    model.sequence.steps = vec![ProcedureStep::structural_nonlinear(
        "nonlinear_1".to_string(),
    )];
    set_model_contact(&mut model, 0.9);

    let error = analysis_run_nonlinear_with_options_op(
        &model,
        ComputeBackend::Cpu,
        AnalysisNonlinearRunOptions {
            quality_policy: QualityPolicy::Balanced,
            contact_enforcement: Some(ContactEnforcementOptions {
                penalty_stiffness_scale: 0.15,
                max_penetration_ratio: 0.035,
            }),
            ..AnalysisNonlinearRunOptions::balanced()
        },
        OperationContext::new(None, None),
    )
    .expect_err("contact requires a separately qualified physical formulation");
    assert!(!error.retryable);
}

#[test]
fn analysis_run_modal_returns_native_modal_result() {
    let _guard = analysis_test_guard();
    let geometry = sample_geometry_asset();
    let modal_model = test_model_for_geometry(
        &geometry,
        TestModelSpec {
            model_id: "modal_model_run".to_string(),
            kind: TestModelKind::ModalStructural,
        },
        OperationContext::new(None, None),
    )
    .expect("modal model should be created");

    let envelope = analysis_run_modal_with_options_op(
        &modal_model.data,
        ComputeBackend::Cpu,
        Default::default(),
        OperationContext::new(None, None),
    )
    .expect("modal run should produce modal result");

    assert_eq!(envelope.operation, "fea.run_modal");
    assert_eq!(envelope.op_version, "fea.run_modal/v1");
    assert_eq!(
        envelope.data.run.solver_method,
        SolverMethod::ShiftInvertBlockLanczos
    );
    assert_eq!(
        envelope.data.provenance.solver_method,
        SolverMethod::ShiftInvertBlockLanczos
    );
    assert_eq!(envelope.data.run_status, RunStatus::Completed);
    assert!(envelope.data.numerically_acceptable);
    let modal = envelope
        .data
        .modal_results
        .as_ref()
        .expect("modal payload should exist");
    assert!(!modal.frequencies_hz().is_empty());
    assert_eq!(modal.frequencies_hz().len(), modal.mode_shapes.len());
    assert_eq!(
        modal.mode_shapes[0].field_id,
        fea_modal_mode_shape_field_id(1)
    );
    assert_eq!(modal.mode_shapes[0].shape.len(), 2);
    assert_eq!(modal.mode_shapes[0].shape[1], 3);
    assert_eq!(modal.frequencies_hz().len(), modal.residual_norms().len());
    assert!(modal.residual_norms().iter().all(|value| value.is_finite()));
    assert_eq!(modal.mode_indices.len(), modal.basis.modes().len());
    assert_eq!(modal.mode_units, ModalFrequencyUnits::Hz);
    assert_eq!(modal.frequency_basis, ModalFrequencyBasis::NativeEigenSolve);
    let declaration = &modal_model.data.sequence.steps[0].state_outputs[0];
    let payload = semantic_state_payload(&envelope.data, declaration)
        .expect("modal result should materialize its typed basis state");
    assert_eq!(
        payload.modal_basis().unwrap().identity(),
        modal.basis.identity()
    );
    let results = analysis_results_op(
        &envelope.data,
        AnalysisResultsQuery::default(),
        OperationContext::new(None, None),
    )
    .expect("modal results should be queryable");
    let field_ids = results
        .data
        .field_descriptors
        .iter()
        .map(|descriptor| descriptor.field_id.as_str())
        .collect::<Vec<_>>();
    assert!(field_ids.contains(&FEA_FIELD_MODAL_FREQUENCY_HZ));
    assert!(field_ids.contains(&FEA_FIELD_MODAL_EIGENVALUE));
    assert!(field_ids.contains(&FEA_FIELD_MODAL_MODAL_MASS));
    assert!(field_ids.contains(&FEA_FIELD_MODAL_MODAL_STIFFNESS));
    assert!(field_ids.contains(&FEA_FIELD_MODAL_PARTICIPATION_FACTOR));
    assert!(field_ids.contains(&FEA_FIELD_MODAL_RESIDUAL_NORM));
    assert!(field_ids.contains(&FEA_FIELD_MODAL_RELATIVE_FREQUENCY_SEPARATION));
    let orthogonality_descriptor = results
        .data
        .field_descriptors
        .iter()
        .find(|descriptor| descriptor.field_id == FEA_FIELD_MODAL_M_ORTHOGONALITY)
        .expect("modal orthogonality descriptor should be present");
    assert_eq!(orthogonality_descriptor.kind, AnalysisFieldKind::Tensor);
    assert_eq!(
        orthogonality_descriptor.shape,
        vec![modal.frequencies_hz().len(), modal.frequencies_hz().len()]
    );
    assert_eq!(orthogonality_descriptor.component_count, None);
    assert!(!envelope
        .data
        .quality_reasons
        .iter()
        .any(|reason| reason.code == QualityReasonCode::ModalResidualExceeded));
    assert!(envelope
        .data
        .run
        .diagnostics
        .iter()
        .any(|diag| diag.code == "FEA_MODAL_CONVERGENCE"));
}

#[test]
fn analysis_run_acoustic_returns_acoustic_fields_and_diagnostics() {
    let _guard = analysis_test_guard();
    let geometry = sample_geometry_asset();
    let acoustic_model = test_model_for_geometry(
        &geometry,
        TestModelSpec {
            model_id: "acoustic_model_run".to_string(),
            kind: TestModelKind::AcousticHarmonic,
        },
        OperationContext::new(None, None),
    )
    .expect("acoustic model should be created");

    let envelope = analysis_run_acoustic_with_options_op(
        &acoustic_model.data,
        ComputeBackend::Cpu,
        Default::default(),
        OperationContext::new(None, None),
    )
    .expect("acoustic run should produce acoustic fields");

    assert_eq!(envelope.operation, "fea.run_acoustic");
    assert_eq!(envelope.op_version, "fea.run_acoustic/v1");
    assert_eq!(
        envelope.data.run.solver_method,
        SolverMethod::AcousticDomainGraphHelmholtzHarmonic
    );
    assert!(envelope.data.modal_results.is_none());
    assert!(envelope
        .data
        .run
        .diagnostics
        .iter()
        .any(|diag| diag.code == "FEA_ACOUSTIC_HARMONIC_RESPONSE"));
    assert!(envelope
        .data
        .run
        .diagnostics
        .iter()
        .any(|diag| diag.code == "FEA_ACOUSTIC_DOMAIN_ASSEMBLY"));
    assert!(envelope.data.run.diagnostics.iter().any(|diag| {
        diag.code == "FEA_ACOUSTIC_DOMAIN_ASSEMBLY"
            && diag.message.contains("domain_edge_count=2")
            && diag.message.contains("domain_active_dimension_count=2")
    }));
    assert!(envelope
        .data
        .run
        .diagnostics
        .iter()
        .any(|diag| diag.code == "FEA_ACOUSTIC_HELMHOLTZ_RESIDUAL"));
    assert!(envelope
        .data
        .run
        .diagnostics
        .iter()
        .any(|diag| diag.code == "FEA_ACOUSTIC_FREQUENCY_RESPONSE"));
    assert!(envelope.data.run.diagnostics.iter().any(|diag| {
        diag.code == "FEA_ACOUSTIC_KNOWN_ANSWER"
            && diag.message.contains("tube_mode_alignment_error_ratio=")
            && diag.message.contains("tube_pressure_variation_ratio=")
            && diag.message.contains("cavity_mode_spacing_ratio=")
            && diag.message.contains("known_answer_coverage_ratio=1")
    }));
    assert!(envelope.data.run.diagnostics.iter().any(|diag| {
        diag.code == "FEA_ACOUSTIC_HARMONIC_RESPONSE"
            && diag.message.contains("acoustic_material_coverage_ratio=1")
    }));
    assert!(envelope.data.run.diagnostics.iter().any(|diag| {
        diag.code == "FEA_ACOUSTIC_BOUNDARY_MODEL"
            && diag.message.contains("acoustic_boundary_coverage_ratio=1")
            && diag.message.contains("rigid_wall_count=1")
    }));
    assert!(envelope
        .data
        .run
        .field(FEA_FIELD_ACOUSTIC_PRESSURE_REAL)
        .is_some());
    assert!(envelope
        .data
        .run
        .field(FEA_FIELD_ACOUSTIC_PRESSURE_MAGNITUDE)
        .is_some());
    assert!(envelope
        .data
        .run
        .field(FEA_FIELD_ACOUSTIC_PARTICLE_VELOCITY)
        .is_some());
    let response_field_count = envelope
        .data
        .run
        .fields
        .iter()
        .filter(|field| field.field_id.starts_with("acoustic.frequency_response."))
        .count();
    assert_eq!(response_field_count, 3);
    let expected_first_response_hz = 125.0_f64 * 3.0 * 3.0_f64.sqrt() * 0.75;
    assert!(envelope
        .data
        .run
        .field(&fea_acoustic_frequency_response_field_id(
            expected_first_response_hz
        ))
        .is_some());
    let results = analysis_results_op(
        &envelope.data,
        AnalysisResultsQuery::default(),
        OperationContext::new(None, None),
    )
    .expect("acoustic results should be queryable");
    let field_ids = results
        .data
        .field_descriptors
        .iter()
        .map(|descriptor| descriptor.field_id.as_str())
        .collect::<Vec<_>>();
    assert!(field_ids.contains(&FEA_FIELD_ACOUSTIC_PRESSURE_REAL));
    assert!(field_ids.contains(&FEA_FIELD_ACOUSTIC_PRESSURE_MAGNITUDE));
    assert!(field_ids.contains(&FEA_FIELD_ACOUSTIC_PARTICLE_VELOCITY));
    assert!(field_ids
        .iter()
        .any(|field_id| field_id.starts_with("acoustic.frequency_response.")));
}

#[test]
fn analysis_run_acoustic_realizes_complex_frequency_pressure() {
    let _guard = analysis_test_guard();
    let geometry = sample_geometry_asset();
    let mut complex_model = test_model_for_geometry(
        &geometry,
        TestModelSpec {
            model_id: "acoustic_complex_frequency_source".to_string(),
            kind: TestModelKind::AcousticHarmonic,
        },
        OperationContext::new(None, None),
    )
    .expect("acoustic model should be created")
    .data;
    let PhysicalSignal::Pressure(signal) = &mut complex_model.signals[0] else {
        panic!("acoustic template should author a pressure signal");
    };
    signal.representation = SignalRepresentation::Tabulated(TabulatedSignal {
        domain: TabulatedSignalDomain::FrequencyHertz,
        knots: vec![
            SignalKnot::continuous(
                50.0,
                SignalOrdinate::Complex {
                    real: SignalValue::Scalar(runmat_analysis_core::Pressure::new(1.0)),
                    imaginary: SignalValue::Scalar(runmat_analysis_core::Pressure::new(0.25)),
                },
            ),
            SignalKnot::continuous(
                20_000.0,
                SignalOrdinate::Complex {
                    real: SignalValue::Scalar(runmat_analysis_core::Pressure::new(2.0)),
                    imaginary: SignalValue::Scalar(runmat_analysis_core::Pressure::new(0.5)),
                },
            ),
        ],
        interpolation: SignalInterpolation::Linear,
        extrapolation: SignalExtrapolation::HoldEndpoint,
    });
    signal.precision = SignalPrecision::Complex128;
    signal.evidence.sample_count = 2;

    let mut real_model = complex_model.clone();
    let PhysicalSignal::Pressure(signal) = &mut real_model.signals[0] else {
        panic!("acoustic template should author a pressure signal");
    };
    let SignalRepresentation::Tabulated(table) = &mut signal.representation else {
        panic!("test should have authored a tabulated pressure signal");
    };
    for knot in &mut table.knots {
        let runmat_analysis_core::SignalKnotValue::Continuous(SignalOrdinate::Complex {
            imaginary,
            ..
        }) = &mut knot.value
        else {
            panic!("test should have authored complex pressure knots");
        };
        *imaginary = SignalValue::Scalar(runmat_analysis_core::Pressure::new(0.0));
    }

    let run = |model: &AnalysisModel| {
        analysis_run_acoustic_with_options_op(
            model,
            ComputeBackend::Cpu,
            Default::default(),
            OperationContext::new(None, None),
        )
        .expect("frequency-domain acoustic source should be realized")
        .data
        .run
    };
    let complex_run = run(&complex_model);
    let real_run = run(&real_model);
    let complex_imag = complex_run
        .field(FEA_FIELD_ACOUSTIC_PRESSURE_IMAG)
        .and_then(AnalysisField::as_host_f64)
        .expect("complex acoustic solve should return imaginary pressure");
    let real_imag = real_run
        .field(FEA_FIELD_ACOUSTIC_PRESSURE_IMAG)
        .and_then(AnalysisField::as_host_f64)
        .expect("real acoustic solve should return imaginary pressure");
    assert!(complex_imag
        .iter()
        .zip(real_imag)
        .any(|(complex, real)| (complex - real).abs() > 1.0e-10));
}

#[test]
fn analysis_run_modal_with_options_controls_requested_mode_count() {
    let _guard = analysis_test_guard();
    let geometry = sample_geometry_asset();
    let modal_model = test_model_for_geometry(
        &geometry,
        TestModelSpec {
            model_id: "modal_model_run_opts".to_string(),
            kind: TestModelKind::ModalStructural,
        },
        OperationContext::new(None, None),
    )
    .expect("modal model should be created");

    let envelope = analysis_run_modal_with_options_op(
        &modal_model.data,
        ComputeBackend::Cpu,
        AnalysisModalRunOptions {
            deterministic_mode: true,
            precision_mode: PrecisionMode::Fp64,
            quality_policy: QualityPolicy::Balanced,
            truncation: runmat_analysis_fea::ModalTruncationPolicy::requested(2),
            convergence: runmat_analysis_fea::ModalConvergencePolicy::production(),
            residual_vector_directions: Vec::new(),
            solver_mesh_artifact_path: None,
        },
        OperationContext::new(None, None),
    )
    .expect("modal run should succeed with options");

    let modal = envelope
        .data
        .modal_results
        .as_ref()
        .expect("modal payload should exist");
    assert!(!modal.frequencies_hz().is_empty());
    assert!(modal.frequencies_hz().len() >= 2);
    assert!(envelope.data.provenance.deterministic_mode);
}

#[test]
fn analysis_results_include_modal_payload_for_modal_runs() {
    let _guard = analysis_test_guard();
    let geometry = sample_geometry_asset();
    let modal_model = test_model_for_geometry(
        &geometry,
        TestModelSpec {
            model_id: "modal_model_results".to_string(),
            kind: TestModelKind::ModalStructural,
        },
        OperationContext::new(None, None),
    )
    .expect("modal model should be created");

    let run = analysis_run_modal_with_options_op(
        &modal_model.data,
        ComputeBackend::Cpu,
        Default::default(),
        OperationContext::new(None, None),
    )
    .expect("modal run should succeed");

    let results = analysis_results_op(
        &run.data,
        AnalysisResultsQuery::default(),
        OperationContext::new(None, None),
    )
    .expect("results should succeed");

    let modal = results
        .data
        .modal_results
        .as_ref()
        .expect("modal payload should propagate to results");
    assert!(!modal.frequencies_hz().is_empty());
    assert_eq!(modal.frequencies_hz().len(), modal.mode_shapes.len());
    assert_eq!(modal.mode_shapes[0].shape.len(), 2);
    assert_eq!(modal.mode_shapes[0].shape[1], 3);
    assert_eq!(modal.frequencies_hz().len(), modal.residual_norms().len());
    assert_eq!(modal.mode_indices.len(), modal.basis.modes().len());
    assert_eq!(modal.mode_units, ModalFrequencyUnits::Hz);
    assert_eq!(modal.frequency_basis, ModalFrequencyBasis::NativeEigenSolve);
    assert!(results.data.summary.mode_count > 0);
    assert_eq!(
        results.data.summary.mode_count,
        results.data.summary.available_mode_indices.len()
    );
    assert!(results.data.summary.min_frequency_hz.is_some());
    assert!(results.data.summary.max_frequency_hz.is_some());
    assert!(results.data.summary.max_modal_residual_norm.is_some());
    assert!(results.data.summary.first_mode_converged.is_some());
}

#[test]
fn analysis_results_query_can_exclude_modal_payload() {
    let _guard = analysis_test_guard();
    let geometry = sample_geometry_asset();
    let modal_model = test_model_for_geometry(
        &geometry,
        TestModelSpec {
            model_id: "modal_model_results_filter".to_string(),
            kind: TestModelKind::ModalStructural,
        },
        OperationContext::new(None, None),
    )
    .expect("modal model should be created");
    let run = analysis_run_modal_with_options_op(
        &modal_model.data,
        ComputeBackend::Cpu,
        Default::default(),
        OperationContext::new(None, None),
    )
    .expect("modal run should succeed");

    let results = analysis_results_op(
        &run.data,
        AnalysisResultsQuery {
            include_fields: Vec::new(),
            include_field_values: true,

            include_diagnostics: true,
            diagnostic_codes: Vec::new(),
            include_modal_results: false,
            include_buckling_results: false,
            mode_indices: Vec::new(),
            include_transient_results: true,
            transient_snapshot_indices: Vec::new(),
            include_nonlinear_results: true,
        },
        OperationContext::new(None, None),
    )
    .expect("results should succeed");

    assert!(results.data.modal_results.is_none());
}

#[test]
fn analysis_results_query_rejects_unknown_modal_mode_index() {
    let _guard = analysis_test_guard();
    let geometry = sample_geometry_asset();
    let modal_model = test_model_for_geometry(
        &geometry,
        TestModelSpec {
            model_id: "modal_model_results_index".to_string(),
            kind: TestModelKind::ModalStructural,
        },
        OperationContext::new(None, None),
    )
    .expect("modal model should be created");
    let run = analysis_run_modal_with_options_op(
        &modal_model.data,
        ComputeBackend::Cpu,
        Default::default(),
        OperationContext::new(None, None),
    )
    .expect("modal run should succeed");

    let err = analysis_results_op(
        &run.data,
        AnalysisResultsQuery {
            include_fields: Vec::new(),
            include_field_values: true,

            include_diagnostics: true,
            diagnostic_codes: Vec::new(),
            include_modal_results: true,
            include_buckling_results: true,
            mode_indices: vec![10],
            include_transient_results: true,
            transient_snapshot_indices: Vec::new(),
            include_nonlinear_results: true,
        },
        OperationContext::new(None, None),
    )
    .expect_err("results should fail for unknown mode index");

    assert_eq!(err.error_code, "RM.FEA.RESULTS.MODE_NOT_FOUND");
    assert_eq!(err.operation, "fea.results");
    assert_eq!(err.op_version, "fea.results/v1");
}

#[test]
fn analysis_results_include_transient_payload_for_transient_runs() {
    let _guard = analysis_test_guard();
    let (mesh_root, mesh_path) =
        write_ready_minimal_analysis_mesh_artifact("transient-results-physical-fields");
    let mut model = sample_solid_model();
    model.sequence.steps = vec![ProcedureStep::structural_transient(
        "transient_1".to_string(),
    )];
    let run = analysis_run_transient_with_options_op(
        &model,
        ComputeBackend::Cpu,
        AnalysisTransientRunOptions {
            solver_mesh_artifact_path: Some(mesh_path.display().to_string()),
            ..Default::default()
        },
        OperationContext::new(None, None),
    )
    .expect("transient run should succeed");

    let results = analysis_results_op(
        &run.data,
        AnalysisResultsQuery::default(),
        OperationContext::new(None, None),
    )
    .expect("results should succeed");

    let transient = results
        .data
        .transient_results
        .as_ref()
        .expect("transient payload should propagate");
    assert_eq!(
        transient.integration_method,
        TransientIntegrationMethod::NewmarkAverageAcceleration
    );
    assert!(!transient.time_points_s.is_empty());
    assert_eq!(
        transient.time_points_s.len(),
        transient.displacement_snapshots.len()
    );
    assert_eq!(
        transient.time_points_s.len(),
        transient.strain_snapshots.len()
    );
    assert_eq!(
        transient.time_points_s.len(),
        transient.stress_snapshots.len()
    );
    assert_eq!(
        transient.time_points_s.len(),
        transient.strain_energy_density_snapshots.len()
    );

    let snapshot_query = AnalysisResultsQuery {
        transient_snapshot_indices: vec![1],
        ..AnalysisResultsQuery::default()
    };
    let selected = analysis_results_by_run_id_op(
        &run.data.run_id,
        snapshot_query,
        OperationContext::new(None, None),
    )
    .expect("persisted transient physical fields should support indexed hydration");
    let selected = selected
        .data
        .transient_results
        .expect("selected transient payload");
    assert_eq!(selected.time_points_s.len(), 1);
    assert_eq!(
        selected.strain_snapshots[0].field_id,
        fea_transient_strain_field_id(1)
    );
    assert_eq!(
        selected.stress_snapshots[0].field_id,
        fea_transient_stress_field_id(1)
    );
    assert_eq!(
        selected.strain_energy_density_snapshots[0].field_id,
        fea_transient_strain_energy_density_field_id(1)
    );
    assert_eq!(
        selected.strain_energy_snapshots[0].field_id,
        fea_transient_strain_energy_field_id(1)
    );
    let _ = fs::remove_dir_all(mesh_root);
}

#[cfg(feature = "plot-core")]
#[test]
fn analysis_generate_study_run_mesh_figure_requires_typed_maxwell_completion() {
    let _guard = analysis_test_guard();
    let spec = sample_electromagnetic_study_spec();
    let error = analysis_run_study_op(&spec, OperationContext::new(None, None))
        .expect_err("no EM plot may be synthesized from the retired scalar proxy");
    assert_eq!(
        error.error_code,
        "RM.FEA.RUN_STUDY.STATIC_EM_TYPED_EXECUTION_REQUIRED"
    );
}

#[test]
fn analysis_results_query_can_exclude_transient_payload() {
    let _guard = analysis_test_guard();
    let mut model = sample_model();
    model.sequence.steps = vec![ProcedureStep::structural_transient(
        "transient_1".to_string(),
    )];
    let run = analysis_run_transient_with_options_op(
        &model,
        ComputeBackend::Cpu,
        Default::default(),
        OperationContext::new(None, None),
    )
    .expect("transient run should succeed");

    let results = analysis_results_op(
        &run.data,
        AnalysisResultsQuery {
            include_fields: Vec::new(),
            include_field_values: true,

            include_diagnostics: true,
            diagnostic_codes: Vec::new(),
            include_modal_results: true,
            include_buckling_results: true,
            mode_indices: Vec::new(),
            include_transient_results: false,
            transient_snapshot_indices: Vec::new(),
            include_nonlinear_results: true,
        },
        OperationContext::new(None, None),
    )
    .expect("results should succeed");

    assert!(results.data.transient_results.is_none());
    assert!(results.data.summary.snapshot_count > 0);
    assert_eq!(results.data.summary.time_start_s, Some(0.0));
    assert!(results.data.summary.time_end_s.unwrap_or(0.0) > 0.0);
    assert!(results.data.summary.max_transient_residual_norm.is_some());
    assert!(results.data.summary.final_step_converged.is_some());
}

#[test]
fn analysis_results_query_rejects_unknown_transient_snapshot_index() {
    let _guard = analysis_test_guard();
    let mut model = sample_model();
    model.sequence.steps = vec![ProcedureStep::structural_transient(
        "transient_1".to_string(),
    )];
    let run = analysis_run_transient_with_options_op(
        &model,
        ComputeBackend::Cpu,
        Default::default(),
        OperationContext::new(None, None),
    )
    .expect("transient run should succeed");

    let err = analysis_results_op(
        &run.data,
        AnalysisResultsQuery {
            include_fields: Vec::new(),
            include_field_values: true,

            include_diagnostics: true,
            diagnostic_codes: Vec::new(),
            include_modal_results: true,
            include_buckling_results: true,
            mode_indices: Vec::new(),
            include_transient_results: true,
            transient_snapshot_indices: vec![999],
            include_nonlinear_results: true,
        },
        OperationContext::new(None, None),
    )
    .expect_err("results should fail for unknown transient snapshot index");

    assert_eq!(
        err.error_code,
        "RM.FEA.RESULTS.TRANSIENT_SNAPSHOT_NOT_FOUND"
    );
    assert_eq!(err.operation, "fea.results");
    assert_eq!(err.op_version, "fea.results/v1");
}

#[test]
fn retired_electromagnetic_proxy_solver_contract_is_rejected() {
    let old = serde_json::json!({
        "solver": "electromagnetic_potential",
        "options": {
            "deterministic_mode": true, "precision_mode": "fp64",
            "quality_policy": "strict", "residual_target": 1e-6,
            "harmonic_tolerance": 1e-7, "harmonic_max_iterations": 96,
            "sweep_enabled": false, "sweep_frequency_hz": []
        }
    });
    assert!(serde_json::from_value::<AnalysisSolverOptions>(old).is_err());
}

#[test]
fn scalar_study_execution_requires_authenticated_maxwell_completion() {
    let _guard = analysis_test_guard();
    let spec = sample_electromagnetic_study_spec();
    assert!(matches!(
        spec.execution.steps[0].solver_options,
        AnalysisSolverOptions::MaxwellHarmonic(_)
    ));
    let error = analysis_run_study_op(&spec, OperationContext::new(None, None))
        .expect_err("scalar model path must not synthesize Maxwell fields");
    assert_eq!(
        error.error_code,
        "RM.FEA.RUN_STUDY.STATIC_EM_TYPED_EXECUTION_REQUIRED"
    );
}

#[test]
fn electromagnetic_harmonic_defaults_select_genuine_maxwell_without_vector_potential() {
    let step = ProcedureStep::electromagnetic_harmonic("phasor");
    assert!(matches!(
        AnalysisSolverOptions::defaults_for(&step).unwrap(),
        AnalysisSolverOptions::MaxwellHarmonic(_)
    ));
}

mod study_step_execution;
