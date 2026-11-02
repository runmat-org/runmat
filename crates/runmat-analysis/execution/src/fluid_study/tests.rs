//! Lifecycle and source-plan rejection tests; governed cold execution is native evidence.
use super::*;
use runmat_analysis_core::*;
use runmat_analysis_fea::{
    execution::FluidSteadyWorkload,
    fixtures::{fixture_model, FixtureId},
    physics::cfd::canonical::CanonicalFluidPolicy,
    properties::compile_property_map,
    topology::FiniteVolumeProjectionPolicy,
};
use runmat_meshing_core::{fixtures, SolverMeshArtifact};
fn authored() -> (FluidSteadyWorkload, SolverMeshArtifact) {
    let mesh = fixtures::slab_mesh(1);
    let mut model = fixture_model(FixtureId::CantileverLinearStatic);
    model.structural = None;
    model.thermal = None;
    model.thermo_mechanical = None;
    model.electro_thermal = None;
    model.electromagnetic = None;
    model.constraints.clear();
    model.boundaries.clear();
    model.signals.clear();
    model.excitations.clear();
    model.excitation_cases.clear();
    model.initial_conditions.clear();
    model.interfaces.clear();
    model.connectors.clear();
    model.response_spectra.clear();
    model.random_excitations.clear();
    model.frame = CoordinateFrame::Global;
    model.materials[0].mass_density_kg_per_m3 = Some(runmat_types::physical::MassDensity::new(1.0));
    model.materials[0].fluid = Some(MaterialFluidModel::IncompressibleNewtonian {
        dynamic_viscosity: DynamicViscosity::new(1.0),
    });
    model.material_assignments[0].target =
        PropertyRegionTarget::cell(runmat_meshing_core::fixtures::fixture_solid("slab"));
    model.cfd = Some(CfdDomain {
        density: MassDensity::new(1.0),
        dynamic_viscosity: DynamicViscosity::new(1.0),
        turbulence_intensity: TurbulenceIntensity::new(0.0),
        rheology: None,
        porous_zones: Vec::new(),
    });
    model.sequence = AnalysisSequence::from_steps(
        "retained-fluid",
        vec![ProcedureStep::fluid_static("steady")],
    );
    model
        .constraints
        .push(AnalysisConstraint::Fluid(FluidConstraint {
            constraint_id: "stationary".into(),
            kind: FluidConstraintKind::NoSlipWall {
                target: RegionTarget::boundary(runmat_meshing_core::fixtures::fixture_face(
                    "slab_insulated",
                )),
            },
        }));
    model
        .constraints
        .push(AnalysisConstraint::Fluid(FluidConstraint {
            constraint_id: "inlet".into(),
            kind: FluidConstraintKind::PrescribedVelocity {
                target: RegionTarget::boundary(runmat_meshing_core::fixtures::fixture_face(
                    "slab_left",
                )),
                value: PrescribedValue::Constant([Velocity::new(0.0); 3]),
            },
        }));
    model.boundaries.push(AnalysisBoundary {
        boundary_id: "outlet".into(),
        target: RegionTarget::boundary(runmat_meshing_core::fixtures::fixture_face("slab_right")),
        operator: BoundaryOperator::FluidOutletPressure {
            value: PrescribedValue::Constant(Pressure::new(0.0)),
        },
    });
    // Bind exactly the original boundary inventory, not a solver-created selector.
    for constraint in &mut model.constraints {
        let AnalysisConstraint::Fluid(c) = constraint else {
            unreachable!()
        };
        let target = match &mut c.kind {
            FluidConstraintKind::NoSlipWall { target }
            | FluidConstraintKind::PrescribedVelocity { target, .. } => target,
            _ => unreachable!(),
        };
        let entities = mesh
            .topology
            .boundary_faces
            .iter()
            .flat_map(|f| f.provenance.iter())
            .filter(|e| target.matches_persistent_entity(e))
            .cloned()
            .collect();
        target.bind_resolved_entities(entities).unwrap();
    }
    for b in &mut model.boundaries {
        let entities = mesh
            .topology
            .boundary_faces
            .iter()
            .flat_map(|f| f.provenance.iter())
            .filter(|e| b.target.matches_persistent_entity(e))
            .cloned()
            .collect();
        b.target.bind_resolved_entities(entities).unwrap();
    }
    let workload = FluidSteadyWorkload {
        model,
        step_id: ProcedureStepId("steady".into()),
        policy: CanonicalFluidPolicy {
            pressure_velocity_coupling: Default::default(),
            fixed_point_acceleration: None,
            maximum_outer_iterations: 20,
            maximum_nonorthogonal_iterations: 4,
            maximum_linear_iterations: 100,
            gmres_restart: 8,
            momentum_relaxation: 0.5,
            pressure_relaxation: 0.25,
            linear_relative_tolerance: 1e-10,
            physical_relative_tolerance: 1e-8,
            maximum_resident_bytes: Some(1_000_000_000),
            maximum_operator_work: Some(1_000_000_000_000),
            maximum_authority_items: 100,
        },
        geometry_policy: FiniteVolumeProjectionPolicy {
            max_cells: Some(100),
            max_face_incidences: Some(400),
            max_boundary_faces: Some(400),
            max_scratch_bytes: Some(10_000_000),
            max_traversal_operations: Some(1_000_000),
        },
    };
    (workload, mesh)
}

pub(crate) fn preparation() -> FluidStudyPreparation {
    let prior = crate::electrical_study::tests::preparation();
    let (workload, mesh) = authored();
    let compiled = CompiledAnalysisStudy {
        module_path: "tests/fluid.fea".into(),
        study_id: "fluid".into(),
        study_inputs: Vec::new(),
        outputs: Vec::new(),
        assessments: Vec::new(),
        properties: compile_property_map(&workload.model, Some(&mesh)).unwrap(),
        model: workload.model,
        solver_mesh_digest: mesh.canonical_digest,
    };
    let policy = FluidStudyPlanningPolicy {
        numerical: workload.policy,
        geometry: workload.geometry_policy,
    };
    let product = plan_fluid_study_product(compiled, &mesh, policy).unwrap();
    FluidStudyPreparation {
        local_execution: runmat_analysis_fea::planning::resource::LocalExecutionPlan::serial(),
        policy,
        product,
        mesh,
        frozen_project_digest: None,
        artifact_access: prior.artifact_access,
        program_revision: prior.program_revision,
        output_policy: prior.output_policy,
        required_capabilities: prior.required_capabilities,
    }
}
struct EmptyStore;
impl CacheImport for EmptyStore {
    fn read_verified(
        &self,
        _: Digest,
    ) -> runmat_execution_artifact::ArtifactResult<Option<Vec<u8>>> {
        Ok(None)
    }
}
fn empty_response() -> ProgramExecutionResponse {
    ProgramExecutionResponse::ExternalizedSuccess {
        outputs: Vec::new(),
        result_objects: Vec::new(),
    }
}
fn accounting(task: &PreparedFluidSteadyTask) -> AnalysisPhaseExecutionEvidence {
    crate::tests::phase_execution(
        &task.host,
        &task.program,
        &task.resource_plan.request,
        &empty_response(),
    )
}
#[test]
fn exact_authored_plan_and_owned_phases_fail_closed() {
    let mut study = FluidStudyCoordinator::new(preparation()).unwrap();
    assert_eq!(study.phase(), FluidStudyPhase::ReadyForSolve);
    assert!(study.prepare_field_recovery().is_err());
    assert!(study.prepare_next_observation(&EmptyStore).is_err());
    assert!(study.prepare_accuracy().is_err());
    assert!(study.prepare_next_state().is_err());
    assert!(study.prepare_publication().is_err());
    let task = study.prepare_solve().unwrap();
    assert_eq!(
        task.host.workload.kind,
        runmat_analysis_fea::execution::AnalysisWorkKind::SolveFluidSteady
    );
    assert_eq!(
        task.host.workload.outputs,
        vec![
            runmat_analysis_fea::execution::AnalysisNumericalObjectKind::SolverResult,
            runmat_analysis_fea::execution::AnalysisNumericalObjectKind::NumericalReceipt
        ]
    );
    assert!(study.prepare_solve().is_err());
    assert!(study.into_completed().is_err());
    // The product is sealed: a preparation can only pair it with another policy or mesh.
    let mut changed = preparation();
    changed.policy.numerical.momentum_relaxation *= 0.5;
    assert!(FluidStudyCoordinator::new(changed).is_err());
    let mut changed = preparation();
    changed.mesh.canonical_digest = runmat_meshing_core::StableDigest([7; 32]);
    assert!(FluidStudyCoordinator::new(changed).is_err());
}
#[test]
fn failed_import_never_installs_usage_or_terminal_authority() {
    let mut study = FluidStudyCoordinator::new(preparation()).unwrap();
    let task = study.prepare_solve().unwrap();
    let usage = accounting(task);
    let mut allocation = task.resource_plan.request.clone();
    allocation.memory_bytes += 1;
    let foreign =
        crate::tests::phase_execution(&task.host, &task.program, &allocation, &empty_response());
    assert!(study
        .complete_solve(&EmptyStore, empty_response(), foreign)
        .is_err());
    assert!(study
        .complete_solve(&EmptyStore, empty_response(), usage)
        .is_err());
    assert_eq!(study.phase(), FluidStudyPhase::AwaitingSolve);
    assert!(study.result().is_none() && study.fields().is_none() && study.tail.is_none());
    assert!(study.usage.is_none() && study.closures.is_empty());
}
#[test]
fn replay_requires_exact_phase_and_original_source() {
    assert!(import_fluid_study_completion(&EmptyStore, preparation(), Vec::new()).is_err());
    let mut study = FluidStudyCoordinator::new(preparation()).unwrap();
    let task = study.prepare_solve().unwrap();
    let usage = accounting(task);
    let wrong = FluidStudyResultClosure::new(
        FluidStudyResultPhase::Accuracy,
        &task.host,
        &empty_response(),
        &usage,
    );
    assert!(import_fluid_study_completion(&EmptyStore, preparation(), vec![wrong]).is_err());
    let mut changed = preparation();
    changed.frozen_project_digest = Some(Digest::sha256("foreign-fluid-project"));
    let mut foreign = FluidStudyCoordinator::new(changed).unwrap();
    let task = foreign.prepare_solve().unwrap();
    let closure = FluidStudyResultClosure::new(
        FluidStudyResultPhase::Solve,
        &task.host,
        &empty_response(),
        &usage,
    );
    assert!(import_fluid_study_completion(&EmptyStore, preparation(), vec![closure]).is_err());
}

#[test]
fn accuracy_owns_one_plan_and_preserves_original_state_source() {
    use crate::{prepare_analysis_objects, prepare_planned_resource_graph_usage};
    use runmat_analysis_fea::execution::{
        AnalysisNumericalObjectKind as Kind, FluidNumericalReceiptArtifact,
    };
    use runmat_execution_artifact::object::ObjectNamespace;

    let preparation = preparation();
    let workload = preparation.workload();
    let result = workload.execute(&preparation.mesh).unwrap();
    let receipt = FluidNumericalReceiptArtifact::new(workload, &result, &preparation.mesh).unwrap();
    let objects = |kind, identity, bytes: Vec<u8>| {
        prepare_analysis_objects(
            kind,
            identity,
            &bytes,
            preparation.output_policy.chunk_bytes(),
            ObjectNamespace::ResultValue,
            preparation.output_policy.inventory_limits(),
        )
        .unwrap()
    };
    let original = ImportedFluidSteadyResult {
        objects: objects(
            Kind::SolverResult,
            result.identity().unwrap().digest(),
            result.canonical_bytes().unwrap(),
        ),
        numerical_receipt_objects: objects(
            Kind::NumericalReceipt,
            receipt.identity().digest(),
            receipt.canonical_bytes().unwrap(),
        ),
        result,
        numerical_receipt: receipt,
    };
    let mut study = FluidStudyCoordinator::new(preparation).unwrap();
    let access = study.preparation.artifact_access.clone();
    let output_policy = study.preparation.output_policy;
    let task = study.prepare_solve().unwrap();
    let original_inputs = task.inputs.clone();
    let graph = accounting(task)
        .resource_graph(&task.resource_plan)
        .unwrap();
    let usage = prepare_planned_resource_graph_usage(
        &task.resource_plan,
        task.host.workload.identity.digest(),
        original.result.identity().unwrap().digest(),
        &graph,
        &access,
        output_policy.chunk_bytes(),
        output_policy.inventory_limits(),
    )
    .unwrap();
    study.result = Some(original);
    let mut tail = study.prepare_tail(&usage, None).unwrap();
    let accuracy = tail.prepare_accuracy().unwrap();
    assert_eq!(
        accuracy
            .inputs
            .iter()
            .filter(|i| i.manifest.kind == Kind::ResourcePlan)
            .count(),
        1
    );
    for kind in [Kind::FluidSteadyWorkload, Kind::Mesh, Kind::SolverResult] {
        assert_eq!(
            accuracy
                .inputs
                .iter()
                .filter(|i| i.manifest.kind == kind)
                .count(),
            1
        );
    }
    let retained = &study.solve_task.as_ref().unwrap().inputs;
    assert_eq!(retained, &original_inputs);
    assert_eq!(
        retained
            .iter()
            .filter(|i| i.manifest.kind == Kind::ResourcePlan)
            .count(),
        1
    );
}
