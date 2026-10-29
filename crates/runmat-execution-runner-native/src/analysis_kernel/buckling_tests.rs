use std::collections::BTreeSet;

use runmat_analysis_core::{
    ExcitationTarget, IndependentDomain, ProcedureStep, ProcedureStepId, ReferenceStatePolicy,
    RegionTarget, SequenceDependency, SequenceDependencyReason, StateTransition,
    StateTransitionMethod, TopologyTransitionPolicy, TransitionId,
};
use runmat_analysis_execution::{
    execute_serial_analysis, import_structural_buckling_result, prepare_analysis_objects,
    prepare_structural_buckling_task, AnalysisHostResponse, AnalysisKernel,
    AnalysisKernelInvocation, StructuralBucklingTaskPreparation,
};
use runmat_analysis_fea::{
    execution::{
        AnalysisNumericalObjectKind, AnalysisWorkKind, CanonicalSemanticState,
        CanonicalSemanticStateInput, CanonicalSemanticStatePayload, CanonicalStateTransition,
        CanonicalStateTransitionInput, SequenceIdentity, SolverResultIdentity, StepIdentity,
        StructuralBucklingSolverPolicy, StructuralBucklingTemplate,
    },
    fixtures::{fixture_model, FixtureId},
    planning::resource::{build_resource_plan, FeaResourcePhase},
    results::observation::ObservationExecutionPolicy,
    ComputeBackend, LinearStaticSolveOptions,
};
use runmat_execution::{Digest, ProgramRevision};
use runmat_execution_artifact::{object::ObjectInventoryLimits, ArtifactAccess, ObjectNamespace};
use runmat_meshing_core::{ElementOrder, PersistentEntityId, PersistentEntityKind, StableDigest};

use super::NativeAnalysisKernel;

fn digest(label: &str) -> Digest {
    Digest::sha256(label)
}

fn revision() -> ProgramRevision {
    ProgramRevision::new(
        digest("buckling-graph"),
        digest("buckling-source"),
        runmat_core::program_environment(runmat_core::CompatMode::Matlab),
    )
    .unwrap()
}

#[test]
fn native_buckling_task_round_trips_the_preload_bound_result() {
    let mut model = fixture_model(FixtureId::StructuralTransientTet4Reference);
    model.constraints.truncate(1);
    model.excitations[0].excitation_mut().target = ExcitationTarget::Region(
        RegionTarget::boundary(runmat_meshing_core::fixtures::fixture_face("tip")),
    );
    let runmat_analysis_core::PhysicalSignal::Force(signal) = &mut model.signals[0] else {
        panic!("structural fixture must carry a force signal");
    };
    signal.shape = runmat_analysis_core::SignalShape::Vector3;
    signal.representation = runmat_analysis_core::SignalRepresentation::Constant(
        runmat_analysis_core::SignalValue::Vector3([
            runmat_analysis_core::Force::new(0.0),
            runmat_analysis_core::Force::new(0.0),
            runmat_analysis_core::Force::new(-1.0),
        ]),
    );
    model.excitations[0].excitation_mut().direction = None;
    let preload = ProcedureStep::structural_preload("preload");
    let mut buckling =
        ProcedureStep::structural_buckling("buckling", "preload", "geometric_stiffness");
    buckling.independent_domain = IndependentDomain::Eigenvalue { requested_count: 1 };
    model.sequence.steps = vec![preload, buckling];
    model.sequence.dependencies = vec![SequenceDependency {
        predecessor: ProcedureStepId("preload".into()),
        successor: ProcedureStepId("buckling".into()),
        reason: SequenceDependencyReason::StateTransfer,
    }];
    model.sequence.transitions = vec![StateTransition {
        transition_id: TransitionId("preload-to-buckling".into()),
        producer_step: ProcedureStepId("preload".into()),
        consumer_step: ProcedureStepId("buckling".into()),
        state_id: runmat_analysis_core::StateId("geometric_stiffness".into()),
        method: StateTransitionMethod::Reuse,
        topology: TopologyTransitionPolicy::RequireIdentical,
        reference: ReferenceStatePolicy::Preserve,
    }];
    let mut mesh =
        runmat_meshing_core::fixtures::canonical_tetrahedron_solver_mesh(ElementOrder::Tet4);
    for (face_index, region) in [(0, "root"), (3, "tip")] {
        mesh.topology.boundary_faces[face_index]
            .provenance
            .push(PersistentEntityId::new(
                PersistentEntityKind::Face,
                region.into(),
                vec!["root".into()],
            ));
        mesh.topology.boundary_faces[face_index].provenance.sort();
    }
    let volume_region = runmat_meshing_core::fixtures::fixture_entity_for_selection(
        model.material_assignments[0]
            .selected_region_id()
            .expect("fixture uses selected material region"),
    )
    .expect("fixture material target is a fixture selector");
    mesh.topology.volume_elements[0].region_id = volume_region.clone();
    mesh.topology.regions[0].region_id = volume_region;
    mesh.canonical_digest = StableDigest::ZERO;
    mesh.seal_canonical_digest().unwrap();
    let topology_digest = Digest::from_bytes(*mesh.canonical_digest.bytes());
    let constraints =
        runmat_analysis_fea::constraints::CanonicalStructuralConstraintPlan::compile_for_model(
            &model,
            Some(&mesh),
        )
        .unwrap();
    let preload_result =
        runmat_analysis_fea::run_linear_static_with_prepared_constraint_plan_and_linearization(
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
    let consumer = StepIdentity::from_solver_contract(sequence, &model.sequence.steps[1]).unwrap();
    let source_result = SolverResultIdentity::from_step_result(
        producer,
        preload_result.linearization.identity().digest(),
    )
    .unwrap();
    let state = CanonicalSemanticState::new(CanonicalSemanticStateInput {
        producer,
        declaration: model.sequence.steps[0].state_outputs[1].clone(),
        topology_digest,
        source_result,
        payload: CanonicalSemanticStatePayload::StructuralLinearization {
            linearization: Box::new(preload_result.linearization),
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
        step_id: ProcedureStepId("buckling".into()),
        backend: ComputeBackend::Cpu,
        observation_execution: ObservationExecutionPolicy::scalable(),
        solver: StructuralBucklingSolverPolicy::new(1, 1.0e-10, 8).unwrap(),
    }
    .bind(transition)
    .unwrap();
    let workload_identity = workload.identity();
    let state_identity = state.identity;
    let recovery_workload = workload.clone();
    let recovery_state = state.clone();
    let recovery_mesh = mesh.clone();
    let limits = ObjectInventoryLimits::default();
    let access = ArtifactAccess {
        authorization_scope: "study:buckling".into(),
        encryption_context: digest("buckling-encryption"),
    };
    let prepared = prepare_structural_buckling_task(StructuralBucklingTaskPreparation {
        local_execution: runmat_analysis_fea::planning::resource::LocalExecutionPlan::serial(),
        authored_study_digest: digest("buckling-study"),
        frozen_project_digest: None,
        program_revision: revision(),
        artifact_access: access,
        workload,
        mesh,
        prestress_state: state,
        chunk_bytes: 256,
        inventory_limits: limits,
        required_capabilities: BTreeSet::new(),
        accelerator_requests: Vec::new(),
    })
    .unwrap();
    assert_eq!(
        prepared.resource_plan.input.phase,
        FeaResourcePhase::StructuralBuckling(workload_identity)
    );
    assert!(prepared.inputs.iter().any(|input| {
        input.manifest.kind == AnalysisNumericalObjectKind::SemanticState
            && input.manifest.logical_identity == state_identity.digest()
    }));
    assert_substituted_inputs_are_rejected(&prepared, limits);
    let directory = tempfile::tempdir().unwrap();
    let mut store = runmat_execution_artifact::cache::FilesystemObjectStore::open(
        directory.path().join("objects"),
        limits.max_object_bytes,
    )
    .unwrap();
    prepared.program.write_to(&mut store).unwrap();
    let completed = execute_serial_analysis(
        &prepared.program.request,
        &mut store,
        &NativeAnalysisKernel,
        256,
        limits,
    )
    .unwrap();
    let response = AnalysisHostResponse::completed(&prepared.host, &completed)
        .unwrap()
        .program_response();
    let imported = import_structural_buckling_result(&store, response, &prepared, limits).unwrap();
    assert_eq!(imported.result.modes().len(), 1);
    assert!(imported.result.modes()[0].load_factor > 0.0);
    assert!(imported.result.modes()[0].residual_norm <= 1.0e-12);
    imported
        .numerical_receipt
        .validate_against(imported.identity, &imported.result)
        .unwrap();
    let recovered = runmat_analysis_fea::recover_completed_structural_buckling_run(
        &recovery_workload,
        &recovery_mesh,
        &recovery_state,
        &imported.result,
        &imported.numerical_receipt,
    )
    .unwrap();
    assert_eq!(recovered.load_factors.len(), 1);
    assert_eq!(recovered.mode_shapes.len(), 1);
    assert!(recovered
        .run
        .field(runmat_analysis_fea::FEA_FIELD_BUCKLING_LOAD_FACTOR)
        .is_some());
    let translation_field_id = runmat_analysis_fea::fea_buckling_mode_shape_field_id(1);
    assert_eq!(recovered.mode_shapes[0].field_id, translation_field_id);
    assert_eq!(recovered.mode_shapes[0].shape, vec![4, 3]);
    assert!(recovered.run.field(&translation_field_id).is_none());
    let rotation_field_id = runmat_analysis_fea::fea_buckling_rotation_shape_field_id(1);
    assert_eq!(
        recovered.rotation_mode_shapes[0].field_id,
        rotation_field_id
    );
    assert_eq!(recovered.rotation_mode_shapes[0].shape, vec![4, 3]);
    assert!(recovered.run.field(&rotation_field_id).is_none());
}

fn assert_substituted_inputs_are_rejected(
    prepared: &runmat_analysis_execution::PreparedStructuralBucklingTask,
    limits: ObjectInventoryLimits,
) {
    let mut substituted_resource_inputs = prepared.inputs.clone();
    let mut substituted_resource_input = prepared.resource_plan.input.clone();
    substituted_resource_input.worker.cpu_millicores += 1;
    let substituted_resource = build_resource_plan(substituted_resource_input).unwrap();
    let resource_objects = prepare_analysis_objects(
        AnalysisNumericalObjectKind::ResourcePlan,
        substituted_resource.identity.digest(),
        &substituted_resource.canonical_bytes().unwrap(),
        256,
        ObjectNamespace::InputValue,
        limits,
    )
    .unwrap();
    let resource_index = substituted_resource_inputs
        .iter()
        .position(|input| input.manifest.kind == AnalysisNumericalObjectKind::ResourcePlan)
        .unwrap();
    substituted_resource_inputs[resource_index] = resource_objects;
    assert!(NativeAnalysisKernel
        .execute(AnalysisKernelInvocation {
            output_policy: runmat_analysis_execution::AnalysisOutputPolicy::new(
                4096,
                ObjectInventoryLimits::default()
            )
            .unwrap(),
            kind: AnalysisWorkKind::SolveStructuralBucklingModes,
            host: &prepared.host,
            inputs: &substituted_resource_inputs,
        })
        .is_err());

    let mut substituted_state_inputs = prepared.inputs.clone();
    let state_index = substituted_state_inputs
        .iter()
        .position(|input| input.manifest.kind == AnalysisNumericalObjectKind::SemanticState)
        .unwrap();
    substituted_state_inputs[state_index]
        .manifest
        .logical_identity = digest("other-prestress");
    assert!(NativeAnalysisKernel
        .execute(AnalysisKernelInvocation {
            output_policy: runmat_analysis_execution::AnalysisOutputPolicy::new(
                4096,
                ObjectInventoryLimits::default()
            )
            .unwrap(),
            kind: AnalysisWorkKind::SolveStructuralBucklingModes,
            host: &prepared.host,
            inputs: &substituted_state_inputs,
        })
        .is_err());
}
