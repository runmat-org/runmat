use super::*;

#[test]
fn structural_study_intent_is_path_free_and_does_not_require_a_mesh() {
    let _guard = analysis_test_guard();
    let mut spec = sample_linear_static_study_spec();
    let mut model = analysis_create_model_op(
        &spec.geometry,
        spec.create_model_intent.clone(),
        OperationContext::new(None, None),
    )
    .expect("sample structural model should resolve")
    .data;
    set_model_contact(&mut model, 0.0);
    spec.model = Some(model);
    spec.geometry.source.path = "/author/machine/private/part.step".to_string();
    spec.geometry.source.sha256 = "11".repeat(32);
    spec.solver_mesh_artifact_path = Some("/missing/local/mesh.cbor".to_string());
    let AnalysisSolverOptions::StructuralLinear(options) =
        &mut spec.execution.steps[0].solver_options
    else {
        panic!("sample study must be structural-linear");
    };
    options.solver_mesh_artifact_path = Some("/another/missing/mesh.cbor".to_string());

    let intent = analysis_prepare_structural_study_intent_op(
        &spec,
        OperationContext::new(Some("trace-intent".into()), Some("request-intent".into())),
    )
    .expect("pre-mesh intent preparation must not load a solver mesh");
    let workload = only_workload(&intent);
    assert!(workload
        .model
        .sequence
        .steps
        .iter()
        .any(|step| step.step_id == workload.step_id));
    assert_eq!(spec.execution.steps[0].backend, workload.backend);
    assert_eq!(intent.meshing_contacts().len(), 1);
    assert_eq!(
        intent.meshing_contacts()[0].side_a_face_selectors,
        vec![intent.model().interfaces[0].primary.region_id.clone()]
    );
    assert_eq!(
        intent.meshing_contacts()[0].side_b_face_selectors,
        vec![intent.model().interfaces[0].secondary.region_id.clone()]
    );

    let bytes = intent
        .continuation_bytes()
        .expect("path-free intent continuation should encode");
    let encoded = String::from_utf8(bytes.clone()).expect("continuation is JSON");
    assert!(!encoded.contains("/author/machine"));
    assert!(!encoded.contains("/missing/local"));
    assert!(!encoded.contains("/another/missing"));

    let restored = PreparedStructuralStudyIntent::from_continuation_bytes(&bytes)
        .expect("path-free intent continuation should restore");
    assert_eq!(restored.study_id(), intent.study_id());
    assert_eq!(
        restored.authored_study_digest(),
        intent.authored_study_digest()
    );
    assert_eq!(restored.model(), intent.model());
    assert_eq!(restored.workloads(), intent.workloads());
}

#[test]
fn structural_study_intent_identity_is_independent_of_local_paths() {
    let _guard = analysis_test_guard();
    let mut first = sample_linear_static_study_spec();
    first.geometry.source.sha256 = "22".repeat(32);
    let mut second = first.clone();
    first.geometry.source.path = "/first/part.step".to_string();
    second.geometry.source.path = "/second/part.step".to_string();
    first.solver_mesh_artifact_path = Some("/first/mesh.cbor".to_string());
    second.solver_mesh_artifact_path = Some("/second/mesh.cbor".to_string());
    if let AnalysisSolverOptions::StructuralLinear(options) =
        &mut second.execution.steps[0].solver_options
    {
        options.solver_mesh_artifact_path = Some("/second/options-mesh.cbor".to_string());
    }
    let authored_study_digest = analysis_study_content_digest(&second);

    let first =
        analysis_prepare_structural_study_intent_op(&first, OperationContext::new(None, None))
            .expect("first intent should prepare");
    let second =
        analysis_prepare_structural_study_intent_op(&second, OperationContext::new(None, None))
            .expect("second intent should prepare");
    assert_eq!(
        first.authored_study_digest(),
        second.authored_study_digest()
    );
    assert_eq!(first.authored_study_digest(), authored_study_digest);
    assert_eq!(first.workloads(), second.workloads());
}

#[test]
fn structural_study_intent_admits_one_typed_direct_transient_workload() {
    let _guard = analysis_test_guard();
    let spec = sample_direct_transient_study_spec("44".repeat(32), 9, 4);
    let step = spec.model.as_ref().unwrap().sequence.steps[0].clone();

    let intent = analysis_prepare_structural_study_intent_op(
        &spec,
        OperationContext::new(Some("trace-direct-transient-intent".into()), None),
    )
    .expect("direct transient intent should prepare without a local mesh");
    let runmat_analysis_fea::execution::StructuralStudyWorkloads::DirectTransient {
        workload,
        block_plan,
    } = intent.workloads()
    else {
        panic!("direct transient study produced the wrong workload kind")
    };
    assert_eq!(workload.step_id, step.step_id);
    assert_eq!(workload.time_step_policy.interval_count(), 9);
    assert_eq!(block_plan.maximum_intervals_per_block(), 4);
    assert_eq!(
        block_plan.terminal_schedule_ordinals(9).unwrap(),
        vec![4, 8, 9]
    );

    let restored = PreparedStructuralStudyIntent::from_continuation_bytes(
        &intent.continuation_bytes().unwrap(),
    )
    .expect("direct transient intent continuation should restore");
    assert_eq!(restored.workloads(), intent.workloads());

    let mut substituted = intent.clone();
    let runmat_analysis_fea::execution::StructuralStudyWorkloads::DirectTransient {
        workload, ..
    } = &mut substituted.workloads
    else {
        unreachable!("test prepared a direct transient")
    };
    workload.residual_target *= 2.0;
    substituted.intent_digest = substituted.semantic_digest().unwrap();
    assert!(substituted.continuation_bytes().is_err());
}

#[test]
fn structural_study_intent_completes_a_partitioned_direct_transient_with_state() {
    let _guard = analysis_test_guard();
    storage::reset_artifact_store_for_tests();
    let root = temp_artifact_root("remote-direct-transient-completion");
    let _ = fs::remove_dir_all(&root);
    let _runtime_guard = scoped_fea_artifact_roots(&root);
    let meshing_output = ready_minimal_meshing_output();
    let spec = sample_direct_transient_study_spec(
        digest_hex(meshing_output.artifact.geometry.source_digest),
        4,
        2,
    );
    let intent = analysis_prepare_structural_study_intent_op(
        &spec,
        OperationContext::new(Some("trace-direct-transient-completion".into()), None),
    )
    .expect("direct transient intent should prepare");
    let runmat_analysis_fea::execution::StructuralStudyWorkloads::DirectTransient {
        mut workload,
        block_plan,
    } = intent.workloads()
    else {
        panic!("test requires a direct transient workload")
    };
    let aliases = runmat_analysis_core::BoundaryRegionAliases::new(
        meshing_output.boundary_region_ids.clone(),
    )
    .unwrap();
    workload.model =
        runmat_analysis_core::resolve_boundary_region_aliases(&workload.model, &aliases).unwrap();
    let options = runmat_analysis_fea::solve::transient::TransientSolveOptions {
        formulation_requirement: workload.formulation_requirement,
        time_step_policy: workload.time_step_policy,
        residual_target: workload.residual_target,
        progress_operation: "test.remote_direct_transient".into(),
        solver_mesh: Some(meshing_output.artifact.clone()),
    };
    let scope = runmat_analysis_fea::pipeline::transient::prepare_transient_checkpoint_scope(
        &workload,
        &meshing_output.artifact,
    )
    .unwrap();
    let terminal_schedule_ordinal = u64::try_from(scope.schedule.points.len() - 1).unwrap();
    let mut predecessor = None;
    let mut segments = Vec::new();
    for terminal in block_plan
        .terminal_schedule_ordinals(terminal_schedule_ordinal)
        .unwrap()
    {
        let request = runmat_analysis_fea::execution::StructuralTransientBlockRequest::new(
            scope.clone(),
            predecessor.as_ref(),
            terminal,
        )
        .unwrap();
        let segment = runmat_analysis_fea::pipeline::transient::run_transient_block(
            runmat_analysis_fea::pipeline::transient::StructuralTransientRunBlockInput {
                model: &workload.model,
                backend: workload.backend,
                options: &options,
                request: &request,
                predecessor: predecessor.as_ref(),
                observation_cursors: Vec::new(),
            },
        )
        .unwrap();
        predecessor = Some(segment.manifest().terminal_checkpoint.clone());
        segments.push(segment);
    }
    let assembly =
        runmat_analysis_fea::execution::StructuralTransientResultAssemblyRequest::new(&segments)
            .unwrap();
    let result = runmat_analysis_fea::execution::StructuralTransientSolverResult::assemble(
        &assembly, &segments,
    )
    .unwrap();
    let numerical_receipt =
        runmat_analysis_fea::execution::StructuralTransientNumericalReceipt::new(
            &result, &assembly, &segments,
        )
        .unwrap();
    let result_identity = result.manifest().identity;
    let envelope = analysis_complete_structural_study_intent_op(
        intent,
        meshing_output,
        StructuralStudyNumericalResult::DirectTransient(Box::new(
            StructuralStudyTransientNumericalResult {
                frozen_project_digest: None,
                step_id: workload.step_id.clone(),
                result,
                numerical_receipt,
            },
        )),
        StructuralStudyCompletionIdentity::new("server-run-direct-transient").unwrap(),
    )
    .expect("partitioned direct transient should enter normal study publication");

    assert_eq!(envelope.data.steps.len(), 1);
    assert_eq!(
        envelope.data.states.len(),
        spec.model.as_ref().unwrap().sequence.steps[0]
            .state_outputs
            .len()
    );
    assert!(envelope
        .data
        .states
        .iter()
        .all(|state| state.source_result == result_identity
            && Path::new(&state.artifact_path).is_file()));
    assert_eq!(storage::list_run_results().unwrap().len(), 1);
    drop(_runtime_guard);
    let _ = fs::remove_dir_all(&root);
}

#[test]
fn structural_study_intent_continuation_rejects_semantic_tampering() {
    let _guard = analysis_test_guard();
    let mut spec = sample_linear_static_study_spec();
    spec.geometry.source.sha256 = "33".repeat(32);
    let intent =
        analysis_prepare_structural_study_intent_op(&spec, OperationContext::new(None, None))
            .expect("intent should prepare");
    let bytes = intent.continuation_bytes().expect("intent should encode");
    let mut value: serde_json::Value = serde_json::from_slice(&bytes).unwrap();
    value["study_id"] = serde_json::Value::String("substituted-study".into());
    let tampered = serde_json::to_vec(&value).unwrap();

    assert!(PreparedStructuralStudyIntent::from_continuation_bytes(&tampered).is_err());
    assert!(PreparedStructuralStudyIntent::from_continuation_bytes(&[]).is_err());

    let mut wrong_schema: serde_json::Value = serde_json::from_slice(&bytes).unwrap();
    wrong_schema["schema_version"] = serde_json::Value::from(2);
    assert!(PreparedStructuralStudyIntent::from_continuation_bytes(
        &serde_json::to_vec(&wrong_schema).unwrap()
    )
    .is_err());

    let mut unknown_field: serde_json::Value = serde_json::from_slice(&bytes).unwrap();
    unknown_field["unexpected"] = serde_json::Value::Bool(true);
    assert!(PreparedStructuralStudyIntent::from_continuation_bytes(
        &serde_json::to_vec(&unknown_field).unwrap()
    )
    .is_err());

    let mut invalid_settings: serde_json::Value = serde_json::from_slice(&bytes).unwrap();
    invalid_settings["meshing_settings"]["target_edge_length_m"] = serde_json::Value::from(0.0);
    assert!(PreparedStructuralStudyIntent::from_continuation_bytes(
        &serde_json::to_vec(&invalid_settings).unwrap()
    )
    .is_err());

    let mut substituted_workload = intent.clone();
    let runmat_analysis_fea::execution::StructuralStudyWorkloads::LinearSequence { workloads } =
        &mut substituted_workload.workloads
    else {
        unreachable!("test prepared a linear sequence")
    };
    workloads.values_mut().next().unwrap().preconditioner =
        runmat_analysis_fea::solve::preconditioner::SpdPreconditionerKind::Ilu0;
    substituted_workload.intent_digest = substituted_workload.semantic_digest().unwrap();
    assert!(substituted_workload.continuation_bytes().is_err());
}

#[test]
fn structural_study_intent_completion_reenters_shared_quality_and_persistence() {
    let _guard = analysis_test_guard();
    storage::reset_artifact_store_for_tests();
    let root = temp_artifact_root("remote-study-intent-completion");
    let _ = fs::remove_dir_all(&root);
    let _runtime_guard = scoped_fea_artifact_roots(&root);
    let meshing_output = ready_minimal_meshing_output();
    let mut spec = sample_linear_static_study_spec();
    spec.geometry.source.sha256 = digest_hex(meshing_output.artifact.geometry.source_digest);
    let intent = analysis_prepare_structural_study_intent_op(
        &spec,
        OperationContext::new(
            Some("trace-remote-completion".into()),
            Some("request-remote-completion".into()),
        ),
    )
    .expect("remote study intent should prepare");
    let executor = CanonicalStructuralExecutor {
        calls: AtomicU64::new(0),
    };
    let workload = only_workload(&intent);
    let numerical_result = executor
        .execute(&workload, &meshing_output.artifact)
        .expect("portable structural solve should succeed");
    let numerical_result = sequence_result(&intent, &meshing_output.artifact, numerical_result);
    let retry_intent = intent.clone();
    let retry_mesh = meshing_output.clone();
    let retry_result = numerical_result.clone();

    let envelope = analysis_complete_structural_study_intent_op(
        intent,
        meshing_output,
        numerical_result,
        StructuralStudyCompletionIdentity::new("server-run-completion").unwrap(),
    )
    .expect("remote study result should reenter runtime completion");
    let retried = analysis_complete_structural_study_intent_op(
        retry_intent,
        retry_mesh,
        retry_result,
        StructuralStudyCompletionIdentity::new("server-run-completion").unwrap(),
    )
    .expect("retrying one server allocation should converge on its completed run");

    assert_eq!(
        envelope.trace_id.as_deref(),
        Some("trace-remote-completion")
    );
    assert_eq!(envelope.data.study_id, spec.study_id);
    assert_eq!(retried.data, envelope.data);
    assert_eq!(storage::list_run_results().unwrap().len(), 1);
    let stored = storage::load_run_result(&envelope.data.steps[0].run_id)
        .expect("completed remote run lookup should succeed")
        .expect("completed remote run must be stored");
    let mut conflicting = stored.clone();
    let runmat_analysis_core::AnalysisFieldValues::HostF64(values) =
        &mut conflicting.run.fields[0].values
    else {
        panic!("portable CPU result must retain host field values");
    };
    values[0] += 1.0;
    assert!(storage::persist_run_result(&conflicting).is_err());

    let round_trip: AnalysisRunResult =
        serde_json::from_slice(&serde_json::to_vec(&stored).unwrap()).unwrap();
    assert_eq!(round_trip, stored, "analysis runs must round-trip exactly");

    let filesystem_store = storage::FilesystemAnalysisArtifactStore::new(root.join("run-store"));
    storage::AnalysisArtifactStore::persist_run(&filesystem_store, &stored)
        .expect("first filesystem publication should succeed");
    storage::AnalysisArtifactStore::persist_run(&filesystem_store, &stored)
        .expect("identical filesystem publication should be idempotent");
    assert!(storage::AnalysisArtifactStore::persist_run(&filesystem_store, &conflicting).is_err());
    assert!(envelope.data.solver_mesh_artifact_path.is_some());
    assert!(envelope.data.meshing_evidence_artifact_path.is_some());
    assert!(Path::new(&envelope.data.evidence_artifact_path).is_file());
    assert!(Path::new(
        envelope
            .data
            .solver_mesh_artifact_path
            .as_deref()
            .expect("mesh artifact path")
    )
    .is_file());
    assert!(Path::new(
        envelope
            .data
            .meshing_evidence_artifact_path
            .as_deref()
            .expect("meshing evidence path")
    )
    .is_file());
    drop(_runtime_guard);
    let _ = fs::remove_dir_all(&root);
}

#[test]
fn structural_study_intent_completion_rejects_a_different_geometry_source() {
    let _guard = analysis_test_guard();
    let meshing_output = ready_minimal_meshing_output();
    let mut spec = sample_linear_static_study_spec();
    spec.geometry.source.sha256 = "77".repeat(32);
    let intent =
        analysis_prepare_structural_study_intent_op(&spec, OperationContext::new(None, None))
            .expect("remote study intent should prepare");
    let executor = CanonicalStructuralExecutor {
        calls: AtomicU64::new(0),
    };
    let workload = only_workload(&intent);
    let numerical_result = executor
        .execute(&workload, &meshing_output.artifact)
        .expect("portable structural solve should succeed");
    let numerical_result = sequence_result(&intent, &meshing_output.artifact, numerical_result);

    let error = analysis_complete_structural_study_intent_op(
        intent,
        meshing_output,
        numerical_result,
        StructuralStudyCompletionIdentity::new("server-run-source-mismatch").unwrap(),
    )
    .expect_err("a mesh from another geometry source must fail closed");
    assert_eq!(
        error.error_code,
        "RM.FEA.RUN_STUDY.REMOTE_MESH_IDENTITY_MISMATCH"
    );
}

#[test]
fn structural_study_intent_completion_rejects_a_different_meshing_request() {
    let _guard = analysis_test_guard();
    let mut meshing_output = ready_minimal_meshing_output();
    let mut spec = sample_linear_static_study_spec();
    spec.geometry.source.sha256 = digest_hex(meshing_output.artifact.geometry.source_digest);
    let intent =
        analysis_prepare_structural_study_intent_op(&spec, OperationContext::new(None, None))
            .expect("remote study intent should prepare");
    meshing_output.artifact.resolved_request.deterministic_seed ^= 1;
    meshing_output.artifact.canonical_digest = StableDigest::ZERO;
    meshing_output.artifact.seal_canonical_digest().unwrap();
    meshing_output.evidence = runmat_meshing_core::fixtures::canonical_tetrahedron_meshing_evidence(
        &meshing_output.artifact,
    );
    let executor = CanonicalStructuralExecutor {
        calls: AtomicU64::new(0),
    };
    let workload = only_workload(&intent);
    let numerical_result = executor
        .execute(&workload, &meshing_output.artifact)
        .expect("portable structural solve should succeed");
    let numerical_result = sequence_result(&intent, &meshing_output.artifact, numerical_result);

    let error = analysis_complete_structural_study_intent_op(
        intent,
        meshing_output,
        numerical_result,
        StructuralStudyCompletionIdentity::new("server-run-request-mismatch").unwrap(),
    )
    .expect_err("a mesh from another canonical request must fail closed");
    assert_eq!(
        error.error_code,
        "RM.FEA.RUN_STUDY.REMOTE_MESH_IDENTITY_MISMATCH"
    );
}

fn sample_direct_transient_study_spec(
    geometry_sha256: String,
    interval_count: usize,
    maximum_intervals_per_block: u64,
) -> AnalysisStudySpec {
    let mut spec = sample_linear_static_study_spec();
    spec.geometry.source.sha256 = geometry_sha256;
    let mut model = analysis_create_model_op(
        &spec.geometry,
        spec.create_model_intent.clone(),
        OperationContext::new(None, None),
    )
    .expect("sample structural model should resolve")
    .data;
    let step = ProcedureStep::structural_transient("step_direct_transient");
    model.sequence = AnalysisSequence::from_steps("sequence_direct_transient", vec![step.clone()]);
    spec.model = Some(model);
    spec.execution = AnalysisExecutionPlan {
        steps: vec![AnalysisExecutionStep {
            step_id: step.step_id,
            backend: ComputeBackend::Cpu,
            requirement: FormulationRequirement::Experimental,
            solver_options: AnalysisSolverOptions::StructuralTransient(
                AnalysisTransientRunOptions {
                    time_step_policy: AnalysisStructuralTimeStepPolicy::Fixed { interval_count },
                    maximum_intervals_per_block,
                    ..AnalysisTransientRunOptions::solid_recommended()
                },
            ),
        }],
    };
    spec
}

fn ready_minimal_meshing_output() -> crate::meshing::MeshingExecutionOutput {
    let mut artifact =
        runmat_meshing_core::fixtures::canonical_tetrahedron_solver_mesh(ElementOrder::Tet4);
    artifact.resolved_request = structural_study::resolve_intent_meshing_request(
        UnitSystem::Meter,
        Some(runmat_meshing_core::MeshingRequestSettings::default()),
    )
    .unwrap()
    .unwrap();
    artifact.topology.boundary_faces[0]
        .provenance
        .push(persistent_face("root"));
    artifact.topology.boundary_faces[1]
        .provenance
        .push(persistent_face("tip"));
    for face in &mut artifact.topology.boundary_faces {
        face.provenance.sort();
    }
    artifact.canonical_digest = StableDigest::ZERO;
    artifact
        .seal_canonical_digest()
        .expect("seal remote solver mesh fixture");
    let evidence = runmat_meshing_core::fixtures::canonical_tetrahedron_meshing_evidence(&artifact);
    crate::meshing::MeshingExecutionOutput {
        artifact,
        evidence,
        boundary_region_ids: std::collections::BTreeMap::new(),
    }
}

fn digest_hex(digest: StableDigest) -> String {
    digest
        .bytes()
        .iter()
        .map(|byte| format!("{byte:02x}"))
        .collect()
}

fn only_workload(
    intent: &PreparedStructuralStudyIntent,
) -> runmat_analysis_fea::execution::StructuralLinearWorkload {
    let runmat_analysis_fea::execution::StructuralStudyWorkloads::LinearSequence { workloads } =
        intent.workloads()
    else {
        panic!("test requires a structural-linear sequence intent")
    };
    assert_eq!(workloads.len(), 1);
    workloads.into_values().next().unwrap()
}

fn sequence_result(
    intent: &PreparedStructuralStudyIntent,
    mesh: &runmat_meshing_core::SolverMeshArtifact,
    result: runmat_analysis_fea::FeaRunResult,
) -> StructuralStudyNumericalResult {
    let workload = only_workload(intent);
    let _study = runmat_analysis_fea::execution::StudyIdentity::from_authored_digest(
        intent.authored_study_digest(),
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
    let solver_result_identity = test_solver_result_identity("intent-result");
    let fields = result
        .fields
        .iter()
        .filter(|field| !field.is_empty() && field.as_host_f64().is_some())
        .cloned()
        .collect::<Vec<_>>();
    let states = step
        .state_outputs
        .iter()
        .map(|declaration| {
            runmat_analysis_fea::execution::CanonicalSemanticState::new(
                runmat_analysis_fea::execution::CanonicalSemanticStateInput {
                    producer,
                    declaration: declaration.clone(),
                    topology_digest: runmat_execution::Digest::from_bytes(
                        *mesh.canonical_digest.bytes(),
                    ),
                    source_result: solver_result_identity,
                    payload:
                        runmat_analysis_fea::execution::CanonicalSemanticStatePayload::Fields {
                            fields: fields.clone(),
                        },
                },
            )
            .unwrap()
        })
        .collect();
    StructuralStudyNumericalResult::LinearSequence(StructuralStudySequenceNumericalResult {
        frozen_project_digest: None,
        steps: vec![StructuralStudyStepNumericalResult {
            step_id: workload.step_id.clone(),
            solver_result_identity,
            result,
        }],
        states,
    })
}
