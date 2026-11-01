use super::*;
use runmat_analysis_core::*;
use runmat_analysis_fea::{
    execution::MagnetostaticWorkload,
    fixtures::{fixture_model, FixtureId},
};
use runmat_execution::ProgramRevision;
use runmat_execution_artifact::{object::ObjectInventoryLimits, ArtifactAccess};
use runmat_meshing_core::{
    fixtures::{quadratic_slab_mesh, slab_mesh},
    SolverMeshArtifact,
};

pub(super) fn task(
    order: ElementOrder,
    cells: usize,
    current: f64,
    permeability: f64,
    chunk: usize,
) -> (
    MagnetostaticWorkload,
    SolverMeshArtifact,
    PreparedMagnetostaticTask,
) {
    let mut model = fixture_model(FixtureId::CantileverLinearStatic);
    model.constraints.clear();
    model.boundaries.clear();
    model.excitations.clear();
    model.excitation_cases.clear();
    model.signals.clear();
    model.initial_conditions.clear();
    model.structural = None;
    model.thermal = None;
    model.thermo_mechanical = None;
    model.electro_thermal = None;
    model.cfd = None;
    model.electromagnetic = Some(ElectromagneticDomain::Magnetostatic {
        formulation: MagnetostaticFormulation::MagneticVectorPotential,
    });
    model.sequence = AnalysisSequence::from_steps(
        "native-magnetic",
        vec![ProcedureStep::electromagnetic_static("potential")],
    );
    model.sequence.steps[0].state_outputs = ["retained", "second"]
        .into_iter()
        .map(|id| StateOutput {
            state_id: StateId(id.into()),
            kind: StateKind::ElectromagneticStatic,
            completeness: StateCompleteness::Complete,
            persistence: StatePersistence::Canonical,
        })
        .collect();
    model.material_assignments[0].target =
        PropertyRegionTarget::solid(runmat_meshing_core::fixtures::fixture_solid("slab"));
    model.materials[0].electrical = Some(MaterialElectricalModel {
        reference_temperature_k: runmat_types::physical::Temperature::new(293.15),
        conductivity_s_per_m: runmat_types::physical::ElectricalConductivity::new(0.0),
        relative_permittivity: runmat_types::physical::RelativePermittivity::new(1.0),
        relative_permeability: runmat_types::physical::RelativePermeability::new(permeability),
        resistive_heating_coefficient: runmat_types::physical::InverseTemperature::new(0.0),
        conductivity_frequency_response: Vec::new(),
    });
    for boundary in ["slab_left", "slab_right", "slab_insulated"] {
        model
            .constraints
            .push(AnalysisConstraint::Magnetic(MagneticConstraint {
                constraint_id: boundary.into(),
                kind: MagneticConstraintKind::VectorPotentialGround {
                    target: RegionTarget::boundary(
                        runmat_meshing_core::fixtures::fixture_boundary_selection(boundary),
                    ),
                },
            }));
    }
    model
        .constraints
        .push(AnalysisConstraint::Magnetic(MagneticConstraint {
            constraint_id: "gauge".into(),
            kind: MagneticConstraintKind::CoulombGauge {
                target: RegionTarget::volume(runmat_meshing_core::fixtures::fixture_solid("slab")),
            },
        }));
    model
        .signals
        .push(PhysicalSignal::ElectricCurrentDensity(Signal {
            signal_id: SignalId("J".into()),
            shape: SignalShape::Vector3,
            frame: Some(CoordinateFrame::Global),
            representation: SignalRepresentation::Constant(SignalValue::Vector3(
                [0.0, 0.0, current].map(ElectricCurrentDensity::new),
            )),
            precision: SignalPrecision::Real64,
            provenance: SignalProvenance::Authored,
            evidence: SignalApproximationEvidence {
                max_absolute_error: Some(0.0),
                max_relative_error: Some(0.0),
                derivative_max_absolute_error: None,
                sample_count: 1,
            },
        }));
    model
        .excitations
        .push(ExcitationAction::MagneticVolumeCurrentDensity(
            AnalysisExcitation {
                excitation_id: ExcitationId("coil".into()),
                target: ExcitationTarget::WholeModel,
                frame: CoordinateFrame::Global,
                direction: None,
                distribution: ExcitationDistribution::IntensityPerVolume,
                kinematics: ExcitationKinematics::FixedFrame,
                signal_id: SignalId("J".into()),
                active_step_ids: vec!["potential".into()],
                combination_group: None,
                projection: ProjectionRequirement {
                    preserve_resultant: false,
                    preserve_moment: false,
                    relative_tolerance: 1e-12,
                },
            },
        ));
    model.excitation_cases.push(ExcitationCase {
        case_id: ExcitationCaseId("coil".into()),
        action_ids: vec![ExcitationId("coil".into())],
        activation: ExcitationCaseActivation::Always,
    });
    let workload = MagnetostaticWorkload {
        model,
        step_id: ProcedureStepId("potential".into()),
        relative_residual_target: 1e-10,
    };
    let mesh = match order {
        ElementOrder::Tet4 => slab_mesh(cells),
        ElementOrder::Tet10 => quadratic_slab_mesh(cells),
    };
    let d = Digest::sha256(b"native magnetic fixture");
    let prepared = prepare_magnetostatic_task(MagnetostaticTaskPreparation {
        local_execution: runmat_analysis_fea::planning::resource::LocalExecutionPlan::serial(),
        authored_study_digest: d,
        frozen_project_digest: None,
        // The process conformance harness runs this task on native workers, whose host
        // identity is this binary's program environment; the fixture uses the same one.
        program_revision: ProgramRevision::new(
            d,
            d,
            runmat_core::program_environment(runmat_core::CompatMode::Matlab),
        )
        .unwrap(),
        artifact_access: ArtifactAccess {
            authorization_scope: "native-magnetic".into(),
            encryption_context: d,
        },
        workload: workload.clone(),
        mesh: mesh.clone(),
        output_policy: AnalysisOutputPolicy::new(chunk, ObjectInventoryLimits::default()).unwrap(),
        required_capabilities: Default::default(),
    })
    .unwrap();
    (workload, mesh, prepared)
}
