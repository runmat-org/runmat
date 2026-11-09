use super::*;
use crate::{FluidFieldRecoveryPreparation, FluidSteadyTaskPreparation};
use runmat_analysis_core::*;
use runmat_analysis_fea::fixtures::{fixture_model, FixtureId};
use runmat_analysis_fea::{
    physics::cfd::canonical::CanonicalFluidPolicy, topology::FiniteVolumeProjectionPolicy,
};
use runmat_execution::{ProgramEnvironment, ProgramRevision};
use runmat_execution_artifact::{object::ObjectInventoryLimits, ArtifactAccess};
use runmat_meshing_core::{fixtures, SolverMeshArtifact};
use std::collections::BTreeSet;

fn revision() -> ProgramRevision {
    ProgramRevision::new(
        Digest::sha256("fluid-graph"),
        Digest::sha256("source"),
        ProgramEnvironment::new(
            1,
            1,
            Digest::sha256("runtime"),
            Digest::sha256("abi"),
            "matlab",
        )
        .unwrap(),
    )
    .unwrap()
}
fn task(
    workload: FluidSteadyWorkload,
    mesh: SolverMeshArtifact,
    chunk: usize,
) -> crate::PreparedFluidSteadyTask {
    task_with_threads(
        workload,
        mesh,
        chunk,
        runmat_analysis_fea::planning::resource::LocalExecutionPlan::serial(),
    )
}
fn task_with_threads(
    workload: FluidSteadyWorkload,
    mesh: SolverMeshArtifact,
    chunk: usize,
    local_execution: runmat_analysis_fea::planning::resource::LocalExecutionPlan,
) -> crate::PreparedFluidSteadyTask {
    crate::prepare_fluid_steady_task(FluidSteadyTaskPreparation {
        local_execution,
        authored_study_digest: Digest::sha256("authored-fluid"),
        frozen_project_digest: None,
        program_revision: revision(),
        artifact_access: ArtifactAccess {
            authorization_scope: "fluid-bridge".into(),
            encryption_context: Digest::sha256("scope"),
        },
        workload,
        mesh,
        output_policy: AnalysisOutputPolicy::new(chunk, ObjectInventoryLimits::default()).unwrap(),
        required_capabilities: BTreeSet::new(),
    })
    .unwrap()
}

#[test]
fn task_closure_rederives_exact_cpu_plan_and_rejects_foreign_bytes_or_chunk_policy() {
    let (w, m) = authored();
    for chunk in [1024, 8192] {
        let task = task(w.clone(), m.clone(), chunk);
        assert_eq!(
            task.host.workload.kind,
            runmat_analysis_fea::execution::AnalysisWorkKind::SolveFluidSteady
        );
        assert_eq!(
            task.host.workload.outputs,
            [
                AnalysisNumericalObjectKind::SolverResult,
                AnalysisNumericalObjectKind::NumericalReceipt
            ]
        );
        assert_eq!(
            rederive_fluid_steady_resource_plan(&w, &m, &task.inputs, task.output_policy).unwrap(),
            task.resource_plan
        );
        crate::require_fluid_source_decode_budget(
            &task.inputs,
            &task.resource_plan,
            task.output_policy.inventory_limits(),
        )
        .unwrap();
        let mut foreign = task.inputs.clone();
        foreign[0] = prepare_analysis_objects(
            AnalysisNumericalObjectKind::FluidSteadyWorkload,
            w.identity().unwrap().digest(),
            b"foreign source",
            chunk,
            ObjectNamespace::InputValue,
            task.output_policy.inventory_limits(),
        )
        .unwrap();
        assert!(rederive_fluid_steady_resource_plan(&w, &m, &foreign, task.output_policy).is_err());
        let other = AnalysisOutputPolicy::new(
            if chunk == 1024 { 8192 } else { 1024 },
            task.output_policy.inventory_limits(),
        )
        .unwrap();
        assert!(rederive_fluid_steady_resource_plan(&w, &m, &task.inputs, other).is_err());
    }
}

#[test]
fn source_decode_authenticates_chunks_before_join_and_rejects_tiny_ceiling() {
    let (mut w, m) = authored();
    let task = task(w.clone(), m.clone(), 1024);
    let mut corrupted = task.inputs.clone();
    corrupted[0].objects[1].bytes[0] ^= 1;
    assert!(crate::require_fluid_source_decode_budget(
        &corrupted,
        &task.resource_plan,
        task.output_policy.inventory_limits()
    )
    .is_err());
    w.model.sequence.steps[0]
        .accuracy
        .resources
        .maximum_resident_bytes = Some(1);
    assert!(prepare_fluid_steady_inputs(
        &w,
        &m,
        task.output_policy,
        runmat_analysis_fea::planning::resource::LocalExecutionPlan::serial()
    )
    .is_err());
}

#[test]
fn self_hashed_lowered_decode_budget_cannot_replace_actual_source_profile() {
    let (w, m) = authored();
    let task = task(w, m, 1024);
    let mut forged_input = task.resource_plan.input.clone();
    let runmat_analysis_fea::planning::resource::FeaResourceDemand::FluidSteadySolve(demand) =
        &mut forged_input.demand
    else {
        unreachable!()
    };
    assert!(
        demand.inputs.decoded_resident_bytes_admission_budget.0
            > demand.inputs.encoded_payload_bytes.0
    );
    demand.inputs.decoded_resident_bytes_admission_budget = demand.inputs.encoded_payload_bytes;
    demand.projection.actual_decode_resident_bytes = demand.inputs.encoded_payload_bytes;
    let forged =
        runmat_analysis_fea::planning::resource::build_resource_plan(forged_input).unwrap();
    let mut inputs = task.inputs.clone();
    let plan = inputs
        .iter_mut()
        .find(|i| i.manifest.kind == AnalysisNumericalObjectKind::ResourcePlan)
        .unwrap();
    *plan = crate::portable_task_support::resource_plan_objects(
        &forged,
        1024,
        task.output_policy.inventory_limits(),
    )
    .unwrap();
    assert!(crate::require_fluid_source_decode_budget(
        &inputs,
        &forged,
        task.output_policy.inventory_limits()
    )
    .is_err());
}

#[test]
fn native_recovery_has_exact_original_layout_source_request_and_plan_without_state_profiles() {
    let (w, m) = authored();
    let task = task(w.clone(), m.clone(), 1024);
    let original = w
        .execute_with_resource_plan(&m, &task.resource_plan)
        .unwrap();
    let source = prepare_analysis_objects(
        AnalysisNumericalObjectKind::SolverResult,
        original.identity().unwrap().digest(),
        &original.canonical_bytes().unwrap(),
        1024,
        ObjectNamespace::ResultValue,
        task.output_policy.inventory_limits(),
    )
    .unwrap();
    let input = |kind| {
        task.inputs
            .iter()
            .find(|i| i.manifest.kind == kind)
            .unwrap()
            .clone()
    };
    let recovered = crate::prepare_fluid_field_recovery(FluidFieldRecoveryPreparation {
        source_host: task.host.clone(),
        workload: input(AnalysisNumericalObjectKind::FluidSteadyWorkload),
        mesh: input(AnalysisNumericalObjectKind::Mesh),
        source_resource_plan: input(AnalysisNumericalObjectKind::ResourcePlan),
        source_result: source,
        program_revision: revision(),
        output_policy: task.output_policy,
    })
    .unwrap();
    // Workload, mesh, source resource plan, source result and the recovery resource plan.
    assert_eq!(recovered.inputs.len(), 5);
    assert_eq!(
        recovered.host.workload.outputs,
        [AnalysisNumericalObjectKind::Field]
    );
    crate::require_fluid_recovery_source_decode_budget(
        &recovered.inputs,
        &recovered.resource_plan,
        recovered.output_policy.inventory_limits(),
    )
    .unwrap();
    assert_eq!(
        crate::build_prepared_fluid_field_recovery_resource_plan(
            &recovered.request,
            &recovered.inputs,
            recovered.output_policy
        )
        .unwrap(),
        recovered.resource_plan
    );
    let mut foreign = recovered.inputs.clone();
    let layout = foreign
        .iter_mut()
        .find(|i| i.manifest.kind == AnalysisNumericalObjectKind::SolverResult)
        .unwrap();
    *layout = prepare_analysis_objects(
        AnalysisNumericalObjectKind::SolverResult,
        layout.manifest.logical_identity,
        b"foreign result",
        1024,
        ObjectNamespace::InputValue,
        recovered.output_policy.inventory_limits(),
    )
    .unwrap();
    assert!(crate::build_prepared_fluid_field_recovery_resource_plan(
        &recovered.request,
        &foreign,
        recovered.output_policy
    )
    .is_err());
    let native = original
        .recover_native_fields(&w, &m, &recovered.request)
        .unwrap();
    // Velocity, pressure, face flux, vorticity, wall shear stress and face pressure.
    assert_eq!(native.fields().len(), 6);
}

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

#[test]
fn fluid_plan_carries_admitted_threads_and_worker_rederivation_converges() {
    let (w, m) = authored();
    let threads =
        runmat_analysis_fea::planning::resource::LocalExecutionPlan::from_cpu_millicores(2_000)
            .unwrap();
    let task = task_with_threads(w.clone(), m.clone(), 1024, threads);
    assert_eq!(task.resource_plan.request.cpu_millicores, 2_000);
    assert_eq!(
        rederive_fluid_steady_resource_plan(&w, &m, &task.inputs, task.output_policy).unwrap(),
        task.resource_plan
    );
    assert_eq!(
        w.execute_with_resource_plan(&m, &task.resource_plan)
            .unwrap(),
        w.execute(&m).unwrap()
    );
}

// witness: FEA-FLUID-ACCURACY-001/acceptance @
//   incompressible_laminar_steady/canonical_tetrahedra/public_study; coverage: partial;
//   boundary: The fluid accuracy source context accepts the original workload, mesh and result
//   under the authored accuracy plan and rejects the same receipt under a re-identified plan or a
//   loosened criterion; the context check only, on one steady fixture, not a full study run.
#[test]
fn fluid_accuracy_context_binds_the_authored_accuracy_plan() {
    let (w, m) = authored();
    let task = task(w.clone(), m.clone(), 1024);
    let original = w
        .execute_with_resource_plan(&m, &task.resource_plan)
        .unwrap();
    let receipt =
        runmat_analysis_fea::execution::FluidNumericalReceiptArtifact::new(&w, &original, &m)
            .unwrap();
    let object = |kind, identity, bytes: Vec<u8>| {
        prepare_analysis_objects(
            kind,
            identity,
            &bytes,
            1024,
            if kind == AnalysisNumericalObjectKind::SolverResult {
                ObjectNamespace::ResultValue
            } else {
                ObjectNamespace::InputValue
            },
            task.output_policy.inventory_limits(),
        )
        .unwrap()
    };
    let inputs = [
        object(
            AnalysisNumericalObjectKind::FluidSteadyWorkload,
            w.identity().unwrap().digest(),
            w.canonical_bytes().unwrap(),
        ),
        object(
            AnalysisNumericalObjectKind::Mesh,
            Digest::from_bytes(*m.canonical_digest.bytes()),
            runmat_meshing_core::CanonicalMeshingContract::canonical_encode(&m).unwrap(),
        ),
        object(
            AnalysisNumericalObjectKind::SolverResult,
            original.identity().unwrap().digest(),
            original.canonical_bytes().unwrap(),
        ),
    ];
    let refs = inputs.iter().collect::<Vec<_>>();
    let plan = w.model.sequence.steps[0].accuracy.clone();
    let request = |plan| {
        runmat_analysis_fea::execution::AccuracyEvaluationRequest::new(
            w.producer_identity().unwrap(),
            runmat_analysis_fea::execution::AccuracySourceKind::FluidSteady,
            receipt.source_result(),
            None,
            plan,
        )
        .unwrap()
    };
    let source =
        runmat_analysis_fea::execution::AccuracySourceReceipt::FluidSteady(receipt.clone());
    crate::validate_accuracy_source_context(&request(plan.clone()), &source, None, &refs).unwrap();
    let mut retagged = plan.clone();
    retagged.accuracy_plan_id = AccuracyPlanId("foreign-accuracy".into());
    assert!(
        crate::validate_accuracy_source_context(&request(retagged), &source, None, &refs).is_err()
    );
    let mut loosened = plan;
    if let AccuracyTolerance::Relative(value) = &mut loosened.criteria[0].tolerance {
        *value *= 10.0;
    }
    assert!(
        crate::validate_accuracy_source_context(&request(loosened), &source, None, &refs).is_err()
    );
}
