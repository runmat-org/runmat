use runmat_analysis_core::{
    AnalysisConstraint, ExcitationTarget, FormulationRequirement, IndependentDomain,
    PrescribedStructuralDofs, ProcedureStep, ProcedureStepId, ReferenceStatePolicy, RegionTarget,
    SequenceDependency, SequenceDependencyReason, StateId, StateTransition, StateTransitionMethod,
    StructuralConstraint, StructuralConstraintKind, TopologyTransitionPolicy, TransitionId,
};
use runmat_analysis_execution::{
    execute_serial_analysis, import_structural_frequency_entry,
    import_structural_frequency_recovered_fields, import_structural_frequency_result,
    prepare_analysis_objects, prepare_structural_frequency_fanout,
    prepare_structural_frequency_field_recovery,
    prepare_structural_frequency_result_assembly_request_objects,
    prepare_structural_frequency_result_task, prepare_structural_frequency_workload_objects,
    AnalysisHostResponse, ImportedRecoveredFields, ImportedStructuralFrequencyResult,
    PreparedAnalysisObjects, StructuralFrequencyFanoutPreparation,
    StructuralFrequencyFieldRecoveryPreparation, StructuralFrequencyResultPreparation,
};
use runmat_analysis_fea::{
    execution::{
        AnalysisNumericalObjectKind, CanonicalSemanticState, CanonicalSemanticStateInput,
        CanonicalSemanticStatePayload, SequenceIdentity, SolverResultIdentity, StepIdentity,
        StructuralFrequencyResultAssemblyRequest, StructuralFrequencySolverPlan,
        StructuralFrequencyWorkload, StructuralFrequencyWorkloadInput,
    },
    fixtures::{fixture_model, FixtureId},
    frequency::FrequencyResponsePlan,
    solve::dynamics::DirectHarmonicPolicy,
    ComputeBackend, ModalSolveOptions, ModalTruncationPolicy,
};
use runmat_execution::{Digest, ProgramEnvironment, ProgramRevision};
use runmat_execution_artifact::object::ObjectInventoryLimits;
use runmat_execution_artifact::{ArtifactAccess, ObjectNamespace};
use runmat_meshing_core::{CanonicalMeshingContract, ElementOrder};

use super::NativeAnalysisKernel;

fn digest(label: &str) -> Digest {
    Digest::sha256(label)
}

fn access() -> ArtifactAccess {
    ArtifactAccess {
        authorization_scope: "study:structural-frequency".into(),
        encryption_context: digest("structural-frequency-encryption"),
    }
}

fn revision() -> ProgramRevision {
    ProgramRevision::new(
        digest("structural-frequency-graph"),
        digest("structural-frequency-source"),
        ProgramEnvironment::new(
            1,
            1,
            digest("structural-frequency-runtime"),
            digest("structural-frequency-catalog"),
            "matlab",
        )
        .unwrap(),
    )
    .unwrap()
}

fn fixture_model_and_mesh() -> (
    runmat_analysis_core::AnalysisModel,
    runmat_meshing_core::SolverMeshArtifact,
) {
    let mut model = fixture_model(FixtureId::CantileverLinearStatic);
    model.material_assignments[0].region_id = "region".into();
    model.constraints = vec![AnalysisConstraint::Structural(StructuralConstraint {
        constraint_id: "fixed-face".into(),
        frame: runmat_analysis_core::CoordinateFrame::Global,
        kind: StructuralConstraintKind::Prescribed(PrescribedStructuralDofs {
            target: RegionTarget::boundary("face:0"),
            translations: [
                runmat_analysis_core::StructuralTranslationDof::Ux,
                runmat_analysis_core::StructuralTranslationDof::Uy,
                runmat_analysis_core::StructuralTranslationDof::Uz,
            ]
            .into_iter()
            .map(|dof| runmat_analysis_core::TranslationDofValue {
                dof,
                value: runmat_analysis_core::PrescribedValue::Constant(
                    runmat_analysis_core::Displacement::new(0.0),
                ),
            })
            .collect(),
            rotations: Vec::new(),
        }),
    })];
    model.excitations[0].excitation_mut().target =
        ExcitationTarget::Region(RegionTarget::boundary("face:3"));
    (
        model,
        runmat_meshing_core::fixtures::canonical_tetrahedron_solver_mesh(ElementOrder::Tet4),
    )
}

fn direct_fixture() -> (
    StructuralFrequencyWorkload,
    runmat_meshing_core::SolverMeshArtifact,
) {
    let (mut model, mesh) = fixture_model_and_mesh();
    let mut step = ProcedureStep::structural_harmonic("frequency");
    step.independent_domain = IndependentDomain::FrequencyHertz(vec![20.0, 40.0]);
    model.sequence.steps = vec![step];
    let workload = StructuralFrequencyWorkload::new(StructuralFrequencyWorkloadInput {
        model,
        step_id: runmat_analysis_core::ProcedureStepId("frequency".into()),
        backend: ComputeBackend::Cpu,
        formulation_requirement: FormulationRequirement::Experimental,
        observation_execution:
            runmat_analysis_fea::results::observation::ObservationExecutionPolicy::scalable(),
        plan: FrequencyResponsePlan::new(20.0, &[40.0], &[]).unwrap(),
        solver: StructuralFrequencySolverPlan::Direct {
            policy: DirectHarmonicPolicy::default(),
        },
    })
    .unwrap();
    (workload, mesh)
}

fn execute_frequency(
    workload: &StructuralFrequencyWorkload,
    mesh: &runmat_meshing_core::SolverMeshArtifact,
    modal_state: Option<PreparedAnalysisObjects>,
    recover_fields: bool,
) -> (
    ImportedStructuralFrequencyResult,
    Vec<ImportedRecoveredFields>,
) {
    let limits = ObjectInventoryLimits::default();
    let namespace = ObjectNamespace::InputValue;
    let workload_objects =
        prepare_structural_frequency_workload_objects(workload, 256, namespace, limits).unwrap();
    let mesh_objects = prepare_analysis_objects(
        AnalysisNumericalObjectKind::Mesh,
        Digest::from_bytes(*mesh.canonical_digest.bytes()),
        &mesh.canonical_encode().unwrap(),
        256,
        namespace,
        limits,
    )
    .unwrap();
    let directory = tempfile::tempdir().unwrap();
    let mut store = runmat_execution_artifact::cache::FilesystemObjectStore::open(
        directory.path().join("objects"),
        limits.max_object_bytes,
    )
    .unwrap();
    let fanout = prepare_structural_frequency_fanout(StructuralFrequencyFanoutPreparation {
        workload: workload_objects.clone(),
        mesh: mesh_objects.clone(),
        modal_state,
        artifact_access: access(),
        program_revision: revision(),
        chunk_bytes: 256,
        inventory_limits: limits,
    })
    .unwrap();
    let mut imported = Vec::new();
    let mut recovered = Vec::new();
    let mut recovery_sources = Vec::new();
    for task in fanout {
        task.program.write_to(&mut store).unwrap();
        let completed = execute_serial_analysis(
            &task.program.request,
            &mut store,
            &NativeAnalysisKernel,
            256,
            limits,
        )
        .unwrap();
        let response = AnalysisHostResponse::completed(&task.host, &completed)
            .unwrap()
            .program_response();
        let entry = import_structural_frequency_entry(&store, response, &task, limits).unwrap();
        recovery_sources.push((task.host.clone(), entry.objects.clone()));
        if recover_fields {
            let recovery = prepare_structural_frequency_field_recovery(
                StructuralFrequencyFieldRecoveryPreparation {
                    source_host: task.host.clone(),
                    workload: workload_objects.clone(),
                    mesh: mesh_objects.clone(),
                    source_entry: entry.objects.clone(),
                    artifact_access: access(),
                    program_revision: revision(),
                    chunk_bytes: 256,
                    inventory_limits: limits,
                },
            )
            .unwrap();
            assert!(recovery
                .host
                .workload
                .inputs
                .iter()
                .all(|input| input.kind != AnalysisNumericalObjectKind::SolverResult));
            assert_eq!(
                recovery
                    .host
                    .workload
                    .inputs
                    .iter()
                    .filter(|input| {
                        input.kind == AnalysisNumericalObjectKind::StructuralFrequencyEntryResult
                    })
                    .count(),
                1
            );
            recovery.program.write_to(&mut store).unwrap();
            let completed = execute_serial_analysis(
                &recovery.program.request,
                &mut store,
                &NativeAnalysisKernel,
                256,
                limits,
            )
            .unwrap();
            let response = AnalysisHostResponse::completed(&recovery.host, &completed)
                .unwrap()
                .program_response();
            recovered.push(
                import_structural_frequency_recovered_fields(&store, response, &recovery, limits)
                    .unwrap(),
            );
        }
        imported.push(entry);
    }
    if recover_fields && recovery_sources.len() > 1 {
        assert!(prepare_structural_frequency_field_recovery(
            StructuralFrequencyFieldRecoveryPreparation {
                source_host: recovery_sources[0].0.clone(),
                workload: workload_objects.clone(),
                mesh: mesh_objects.clone(),
                source_entry: recovery_sources[1].1.clone(),
                artifact_access: access(),
                program_revision: revision(),
                chunk_bytes: 256,
                inventory_limits: limits,
            },
        )
        .is_err());
    }
    imported.reverse();
    let entries = imported
        .iter()
        .map(|entry| entry.result.clone())
        .collect::<Vec<_>>();
    let assembly = StructuralFrequencyResultAssemblyRequest::new(&entries).unwrap();
    let assembly_objects = prepare_structural_frequency_result_assembly_request_objects(
        &assembly, 256, namespace, limits,
    )
    .unwrap();
    let task = prepare_structural_frequency_result_task(StructuralFrequencyResultPreparation {
        request: assembly_objects,
        entries: imported.into_iter().map(|entry| entry.objects).collect(),
        artifact_access: access(),
        program_revision: revision(),
    })
    .unwrap();
    task.program.write_to(&mut store).unwrap();
    let completed = execute_serial_analysis(
        &task.program.request,
        &mut store,
        &NativeAnalysisKernel,
        256,
        limits,
    )
    .unwrap();
    let response = AnalysisHostResponse::completed(&task.host, &completed)
        .unwrap()
        .program_response();
    let result = import_structural_frequency_result(&store, response, &task, limits).unwrap();
    (result, recovered)
}

#[test]
fn native_frequency_fanout_and_fanin_are_portable_and_order_independent() {
    let (workload, mesh) = direct_fixture();
    let (result, recovered) = execute_frequency(&workload, &mesh, None, true);
    assert_eq!(
        result
            .result
            .samples()
            .iter()
            .map(|sample| sample.entry().frequency_hz())
            .collect::<Vec<_>>(),
        vec![20.0, 40.0]
    );
    assert_eq!(result.numerical_receipt.samples().len(), 2);
    assert_eq!(recovered.len(), 2);
    assert!(recovered
        .iter()
        .any(|artifact| artifact.manifest.fields.iter().any(|field| {
            field.sample
                == runmat_analysis_fea::results::field::RecoveredFieldSample::FrequencyHertz(20.0)
                && field.definition.quantity()
                    == runmat_analysis_core::ObservationQuantity::ReactionForce
        })));
    assert!(recovered
        .iter()
        .any(|artifact| artifact.manifest.fields.iter().any(|field| {
            field.sample
                == runmat_analysis_fea::results::field::RecoveredFieldSample::FrequencyHertz(40.0)
                && field.definition.quantity() == runmat_analysis_core::ObservationQuantity::Stress
        })));
}

#[test]
fn native_modal_frequency_uses_the_exact_portable_basis_state() {
    let (mut model, mesh) = fixture_model_and_mesh();
    let modal_step = ProcedureStep::structural_modal("modes");
    let mut frequency_step =
        ProcedureStep::structural_modal_harmonic("frequency", "modes", "modal_basis");
    frequency_step.independent_domain = IndependentDomain::FrequencyHertz(vec![20.0, 40.0]);
    model.sequence.steps = vec![modal_step, frequency_step];
    model.sequence.dependencies = vec![SequenceDependency {
        predecessor: ProcedureStepId("modes".into()),
        successor: ProcedureStepId("frequency".into()),
        reason: SequenceDependencyReason::StateTransfer,
    }];
    model.sequence.transitions = vec![StateTransition {
        transition_id: TransitionId("modal-basis-to-frequency".into()),
        producer_step: ProcedureStepId("modes".into()),
        consumer_step: ProcedureStepId("frequency".into()),
        state_id: StateId("modal_basis".into()),
        method: StateTransitionMethod::Reuse,
        topology: TopologyTransitionPolicy::RequireIdentical,
        reference: ReferenceStatePolicy::Preserve,
    }];

    let modal = runmat_analysis_fea::run_modal_with_options(
        &model,
        ComputeBackend::Cpu,
        ModalSolveOptions {
            truncation: ModalTruncationPolicy::requested(6),
            solver_mesh: Some(mesh.clone()),
            formulation_requirement: FormulationRequirement::Experimental,
            ..ModalSolveOptions::default()
        },
    )
    .unwrap();
    let sequence = SequenceIdentity::from_solver_contract(&model.sequence).unwrap();
    let producer = StepIdentity::from_solver_contract(sequence, &model.sequence.steps[0]).unwrap();
    let source_result =
        SolverResultIdentity::from_step_result(producer, modal.basis.identity().digest()).unwrap();
    let state = CanonicalSemanticState::new(CanonicalSemanticStateInput {
        producer,
        declaration: model.sequence.steps[0].state_outputs[0].clone(),
        topology_digest: Digest::from_bytes(*mesh.canonical_digest.bytes()),
        source_result,
        payload: CanonicalSemanticStatePayload::ModalBasis {
            basis: Box::new(modal.basis),
        },
    })
    .unwrap();
    let state_objects = prepare_analysis_objects(
        AnalysisNumericalObjectKind::SemanticState,
        state.identity.digest(),
        &state.canonical_bytes().unwrap(),
        256,
        ObjectNamespace::ResultValue,
        ObjectInventoryLimits::default(),
    )
    .unwrap();
    let workload = StructuralFrequencyWorkload::new(StructuralFrequencyWorkloadInput {
        model,
        step_id: runmat_analysis_core::ProcedureStepId("frequency".into()),
        backend: ComputeBackend::Cpu,
        formulation_requirement: FormulationRequirement::Experimental,
        observation_execution:
            runmat_analysis_fea::results::observation::ObservationExecutionPolicy::scalable(),
        plan: FrequencyResponsePlan::new(20.0, &[40.0], &[]).unwrap(),
        solver: StructuralFrequencySolverPlan::ModalReduced {
            basis_state: state.identity,
        },
    })
    .unwrap();

    let (result, _) = execute_frequency(&workload, &mesh, Some(state_objects), false);
    assert_eq!(result.result.samples().len(), 2);
    match result.numerical_receipt.solution_space() {
        runmat_analysis_fea::execution::StructuralFrequencySolutionSpace::ModalReduced {
            basis_state,
            ..
        } => assert_eq!(basis_state, &state.identity),
        runmat_analysis_fea::execution::StructuralFrequencySolutionSpace::Direct => {
            panic!("modal frequency execution published a direct numerical receipt")
        }
    }
}
