use std::collections::BTreeSet;
use std::net::{Ipv4Addr, SocketAddr};
use std::sync::{
    atomic::{AtomicBool, Ordering},
    Arc,
};
use std::time::Duration;

use rcgen::{generate_simple_self_signed, CertifiedKey};
use runmat_analysis_execution::{
    build_analysis_task_submission, build_structural_task_submission, import_analysis_objects,
    import_analysis_output_set, prepare_structural_linear_program, AnalysisExecutionContext,
    AnalysisHostWorkload, PreparedAnalysisProgram, PreparedStructuralLinearProgram,
    StructuralLinearProgramPreparation, StructuralSequencePreparation,
};
use runmat_analysis_fea::execution::{
    AnalysisNumericalObjectKind, StructuralLinearWorkload, StructuralLinearWorkloadResult,
    DEFAULT_ANALYSIS_CHUNK_BYTES,
};
use runmat_analysis_fea::fixtures::{fixture_model, FixtureId};
use runmat_analysis_fea::solve::backend::kind::LinearAlgebraBackendKind;
use runmat_analysis_fea::solve::preconditioner::SpdPreconditionerKind;
use runmat_analysis_fea::ComputeBackend;
use runmat_execution::identity::ArtifactId;
use runmat_execution::resource::{Capability, ResourceInventory, ResourceRequest};
use runmat_execution::task::RetryPolicy;
use runmat_execution::value::ValuePayload;
use runmat_execution::{Digest, ProgramEnvironment, ProgramRevision};
use runmat_execution_artifact::cache::CacheImport;
use runmat_execution_artifact::object::ObjectInventoryLimits;
use runmat_execution_artifact::{
    archive::{write_bundle, ArchiveLimits},
    ArtifactAccess, ExecutableForm, ExecutionBundleBuilder, ProgramExecutionResponse,
};
use runmat_execution_runner::{PoolSpec, WorkerSpec};
use runmat_execution_runner_native::{
    execute_analysis_program_request, execute_structural_sequence_locally,
    execute_structural_sequence_remotely, run_remote_worker_quic, NativeAnalysisHostLimits,
    NativeAnalysisKernel, NativeExecutionConfig, NativeExecutionError, NativeProgramSession,
    NativeProgramTask, QuicRemoteWorkerChannel, RemotePoolDriver, RemoteWorkerChannel,
    RemoteWorkerChannelConfig, RemoteWorkerQuicRequest, StructuralSequenceExecutionPolicy,
};
use runmat_execution_transport_native::frame::FrameLimits;
use runmat_execution_transport_native::overlay::{PinnedQuicEndpoint, QuicOverlayListener};
use runmat_meshing_core::{ElementOrder, PersistentEntityId, PersistentEntityKind, StableDigest};

#[path = "analysis_process_conformance/publication_chain.rs"]
mod publication_chain;
#[path = "analysis_process_conformance/signal.rs"]
mod signal;
use publication_chain::{success_response, PublicationChainDriver};

fn main() {
    let mode = std::env::args_os().nth(1);
    if mode.as_deref() == Some(std::ffi::OsStr::new("--child")) {
        tokio::runtime::Builder::new_current_thread()
            .enable_all()
            .build()
            .unwrap()
            .block_on(runmat_execution_runner_native::run_worker_stdio())
            .unwrap();
        return;
    }
    if mode.as_deref()
        == Some(std::ffi::OsStr::new(
            runmat_process_host::HiddenMode::ExecutionDriver.marker(),
        ))
    {
        tokio::runtime::Builder::new_current_thread()
            .enable_all()
            .build()
            .unwrap()
            .block_on(run_durable_driver());
        return;
    }
    tokio::runtime::Builder::new_current_thread()
        .enable_all()
        .build()
        .unwrap()
        .block_on(parent());
}

async fn parent() {
    let directory = tempfile::tempdir().unwrap();
    let (host, program, input_objects, resource_plan) =
        fixture(revision(), "analysis-process-conformance");
    let request = program.request.clone();
    let (second_host, second_program, _second_inputs, second_resource_plan) = fixture_for_study(
        revision(),
        "analysis-process-conformance",
        digest("authored-study-second"),
    );
    let second_request = second_program.request.clone();
    let mut config = NativeExecutionConfig::for_current_executable().unwrap();
    config.executable = std::env::current_exe().unwrap();
    config.worker_arguments = vec!["--child".into()];
    config.max_workers = 2;
    config.store_root = directory.path().join("native-analysis-session");
    config.worker_capabilities = capabilities();
    let session = NativeProgramSession::new(config).unwrap();
    let mut store = session.object_store();
    program.write_to(&mut store).unwrap();
    second_program.write_to(&mut store).unwrap();
    let submissions = vec![
        (
            request.clone(),
            analysis_submission(
                session.scope_id(),
                session.pool_id(),
                &host,
                &request,
                Some(&resource_plan),
                resource_plan.request.clone(),
            ),
        ),
        (
            second_request.clone(),
            analysis_submission(
                session.scope_id(),
                session.pool_id(),
                &second_host,
                &second_request,
                Some(&second_resource_plan),
                second_resource_plan.request.clone(),
            ),
        ),
    ];
    let mut tasks = session.submit_graph(submissions).unwrap();
    let second_task = tasks.pop().unwrap();
    let task = tasks.pop().unwrap();
    let process_result = wait_for_native_task(task).await;
    let second_result = wait_for_native_task(second_task).await;
    let process_payload = import_result(&store, success_response(&process_result), &host);
    assert_eq!(
        import_result(&store, success_response(&second_result), &second_host),
        process_payload
    );

    let mut inline_store = store.clone();
    let inline = execute_analysis_program_request(
        &request,
        &mut inline_store,
        &NativeAnalysisKernel,
        NativeAnalysisHostLimits::default(),
    );
    if !matches!(inline, ProgramExecutionResponse::ExternalizedSuccess { .. }) {
        panic!("inline analysis execution failed: {inline:?}");
    }
    let inline_payload = import_result(&inline_store, inline, &host);
    assert_eq!(process_payload, inline_payload);

    let decoded = StructuralLinearWorkloadResult::from_canonical_bytes(&process_payload).unwrap();
    assert!(!decoded.run.fields_are_empty());
    assert!(decoded
        .run
        .diagnostics
        .iter()
        .all(|diagnostic| !diagnostic.message.contains("artifact_path")));

    let prepared_solve = PreparedStructuralLinearProgram {
        host: host.clone(),
        program: program.clone(),
        inputs: input_objects,
        resource_plan,
    };
    let mut publication_driver = publication_chain::ProcessDriver {
        session: &session,
        store: &mut store,
    };
    let published = publication_chain::execute_publication_chain(
        &mut publication_driver,
        &prepared_solve,
        &process_result,
        revision(),
    )
    .await;
    assert_eq!(
        published.source_result,
        decoded_result_identity(&host, &process_payload)
    );

    let mut sequence_config = NativeExecutionConfig::for_current_executable().unwrap();
    sequence_config.executable = std::env::current_exe().unwrap();
    sequence_config.worker_arguments = vec!["--child".into()];
    sequence_config.max_workers = 2;
    sequence_config.store_root = directory.path().join("native-analysis-sequence");
    let sequence = execute_structural_sequence_locally(
        sequence_config,
        sequence_preparation(revision(), "local-analysis-sequence"),
        sequence_policy(),
    )
    .await
    .unwrap();
    assert_sequence(&sequence);

    let mut cancelled_sequence_config = NativeExecutionConfig::for_current_executable().unwrap();
    cancelled_sequence_config.executable = std::env::current_exe().unwrap();
    cancelled_sequence_config.worker_arguments = vec!["--child".into()];
    cancelled_sequence_config.max_workers = 2;
    cancelled_sequence_config.store_root = directory.path().join("cancelled-analysis-sequence");
    let mut cancellation_policy = sequence_policy();
    cancellation_policy.resources.max_wall_millis = 1;
    let cancellation = execute_structural_sequence_locally(
        cancelled_sequence_config,
        sequence_preparation(revision(), "cancelled-analysis-sequence"),
        cancellation_policy,
    )
    .await;
    let Err(cancellation) = cancellation else {
        panic!("one-millisecond structural sequence unexpectedly completed");
    };
    assert!(
        cancellation
            .to_string()
            .contains("exceeded its wall-time limit after reaching quiescence"),
        "unexpected structural sequence cancellation error: {cancellation}"
    );

    let mut user_cancelled_config = NativeExecutionConfig::for_current_executable().unwrap();
    user_cancelled_config.executable = std::env::current_exe().unwrap();
    user_cancelled_config.worker_arguments = vec!["--child".into()];
    user_cancelled_config.max_workers = 2;
    user_cancelled_config.store_root = directory.path().join("user-cancelled-analysis-sequence");
    let interrupt = Arc::new(AtomicBool::new(true));
    let interrupt_guard =
        runmat_runtime::interrupt::replace_interrupt(Some(Arc::clone(&interrupt)));
    let cancellation = execute_structural_sequence_locally(
        user_cancelled_config,
        sequence_preparation(revision(), "user-cancelled-analysis-sequence"),
        sequence_policy(),
    )
    .await;
    assert!(matches!(cancellation, Err(NativeExecutionError::Cancelled)));
    interrupt.store(false, Ordering::Relaxed);
    drop(interrupt_guard);

    remote_conformance(process_payload, published.clone()).await;
    durable_conformance(inline_payload, published).await;
    signal::run().await;
}

async fn wait_for_native_task(task: NativeProgramTask) -> runmat_execution_runner::AttemptSuccess {
    tokio::time::timeout(Duration::from_secs(15), async {
        loop {
            if let Some(result) = task.try_result() {
                break result.unwrap();
            }
            tokio::time::sleep(Duration::from_millis(10)).await;
        }
    })
    .await
    .unwrap()
}

fn decoded_result_identity(
    host: &AnalysisHostWorkload,
    payload: &[u8],
) -> runmat_analysis_fea::execution::SolverResultIdentity {
    runmat_analysis_fea::execution::SolverResultIdentity::new(
        host.workload.identity,
        Digest::sha256(payload),
    )
    .unwrap()
}

async fn run_durable_driver() {
    let invocation = runmat_execution_runner_native::supervisor::prepare_batch_driver().unwrap();
    let job_directory = invocation.job_directory().to_path_buf();
    let runmat_execution_runner_native::supervisor::BatchDriverInvocation::Program {
        submission,
        ..
    } = invocation
    else {
        panic!("analysis conformance driver received a script job")
    };
    let response =
        runmat_execution_runner_native::supervisor::execute_program_batch(*submission).await;
    let succeeded = matches!(
        response,
        ProgramExecutionResponse::Success { .. }
            | ProgramExecutionResponse::ExternalizedSuccess { .. }
    );
    runmat_execution_runner_native::supervisor::complete_batch_driver_with_response(
        &job_directory,
        response,
    )
    .unwrap();
    if !succeeded {
        std::process::exit(1);
    }
}

async fn durable_conformance(
    expected_payload: Vec<u8>,
    expected_publication: runmat_analysis_fea::results::publication::PublishedAnalysisResult,
) {
    let directory = tempfile::tempdir().unwrap();
    let mut driver = publication_chain::DurableDriver::open(
        runmat_execution_runner_native::supervisor::LocalSupervisorConfig {
            executable: std::env::current_exe().unwrap(),
            paths: runmat_execution_runner_native::supervisor::SupervisorPaths::new(
                directory.path().join("durable-analysis"),
            )
            .unwrap(),
            max_stderr_bytes: 1024 * 1024,
            max_object_bytes: 64 * 1024 * 1024,
        },
    );
    let (host, program, input_objects, resource_plan) = fixture(revision(), "durable-analysis");
    let solve = PreparedStructuralLinearProgram {
        host,
        program,
        inputs: input_objects,
        resource_plan,
    };
    let solve_host = solve.host.clone();
    let solve_program = solve.program.clone();
    let solve_success = driver.execute(&solve_host, &solve_program).await;
    assert_eq!(
        import_result(
            driver.store(),
            success_response(&solve_success),
            &solve.host,
        ),
        expected_payload
    );
    let publication = publication_chain::execute_publication_chain(
        &mut driver,
        &solve,
        &solve_success,
        revision(),
    )
    .await;
    assert_eq!(publication, expected_publication);
}

async fn remote_conformance(
    expected_payload: Vec<u8>,
    expected_publication: runmat_analysis_fea::results::publication::PublishedAnalysisResult,
) {
    let project_root = tempfile::tempdir().unwrap();
    std::fs::create_dir_all(project_root.path().join("src")).unwrap();
    std::fs::write(
        project_root.path().join("runmat.toml"),
        "[package]\nname = \"remote-analysis\"\n[sources]\nroots = [\"src\"]\n",
    )
    .unwrap();
    let project = runmat_package::build_frozen_project(
        &project_root.path().join("runmat.toml"),
        BTreeSet::new(),
    )
    .unwrap();
    let project_identity = project.revision();
    let revision = ProgramRevision::new(
        Digest::from_bytes(*project_identity.graph_digest.bytes()),
        Digest::from_bytes(*project_identity.source_revision.bytes()),
        runmat_core::program_environment(runmat_core::CompatMode::Matlab),
    )
    .unwrap();
    let (host, program, input_objects, resource_plan) =
        fixture(revision.clone(), "remote-analysis");
    let request = program.request.clone();
    let (second_host, second_program, _second_inputs, second_resource_plan) = fixture_for_study(
        revision.clone(),
        "remote-analysis",
        digest("remote-authored-study-second"),
    );
    let second_request = second_program.request.clone();
    let bundle = ExecutionBundleBuilder::native(&project, revision.clone())
        .unwrap()
        .with_compiled_package_closure()
        .with_materialized_program(
            request.recipe.clone(),
            ExecutableForm::AnalysisWorkload,
            request.artifact.executable_bytes.clone(),
        )
        .build()
        .unwrap();
    let mut bundle_bytes = Vec::new();
    write_bundle(&bundle, &mut bundle_bytes, ArchiveLimits::default()).unwrap();

    let CertifiedKey { cert, signing_key } =
        generate_simple_self_signed(vec!["runmat.execution".into()]).unwrap();
    let certificate_der = cert.der().to_vec();
    let signing_key_der = signing_key.serialize_der();
    let first_listener = QuicOverlayListener::bind(
        SocketAddr::from((Ipv4Addr::LOCALHOST, 0)),
        vec![certificate_der.clone()],
        signing_key_der.clone(),
        FrameLimits::default(),
    )
    .unwrap();
    let second_listener = QuicOverlayListener::bind(
        SocketAddr::from((Ipv4Addr::LOCALHOST, 0)),
        vec![certificate_der.clone()],
        signing_key_der,
        FrameLimits::default(),
    )
    .unwrap();
    let first_authority = first_listener.local_addr().unwrap();
    let second_authority = second_listener.local_addr().unwrap();
    let scope_id = runmat_execution::ExecutionScopeId::derive(&[b"remote-analysis-scope"]);
    let pool_id = runmat_execution::PoolId::derive(&[b"remote-analysis-pool"]);
    let first_worker = WorkerSpec {
        id: runmat_execution::identity::WorkerId::derive(&[b"remote-analysis-worker-a"]),
        pool_id,
        resources: inventory(),
    };
    let second_worker = WorkerSpec {
        id: runmat_execution::identity::WorkerId::derive(&[b"remote-analysis-worker-b"]),
        pool_id,
        resources: inventory(),
    };
    let run_key =
        runmat_execution_artifact::encryption::RunKeyMaterial::from_entropy([29; 32]).unwrap();
    let first_server = run_remote_worker_quic(RemoteWorkerQuicRequest {
        listener: first_listener,
        run_identity: "remote-analysis".into(),
        worker: first_worker.clone(),
        driver_fence: 41,
        session_id: [31; 16],
        run_key: run_key.clone(),
        limits: FrameLimits::default(),
    });
    let second_server = run_remote_worker_quic(RemoteWorkerQuicRequest {
        listener: second_listener,
        run_identity: "remote-analysis".into(),
        worker: second_worker.clone(),
        driver_fence: 41,
        session_id: [32; 16],
        run_key: run_key.clone(),
        limits: FrameLimits::default(),
    });
    let client = async {
        let first_channel = QuicRemoteWorkerChannel::connect(
            RemoteWorkerChannelConfig {
                run_identity: "remote-analysis".into(),
                node_identity: "remote-analysis-node-a".into(),
                worker: first_worker,
                driver_fence: 41,
                session_id: [31; 16],
                run_key: run_key.clone(),
                limits: FrameLimits::default(),
            },
            &PinnedQuicEndpoint {
                authority: first_authority,
                server_name: "runmat.execution".into(),
                certificate_der: certificate_der.clone(),
            },
        )
        .await
        .unwrap();
        let second_channel = QuicRemoteWorkerChannel::connect(
            RemoteWorkerChannelConfig {
                run_identity: "remote-analysis".into(),
                node_identity: "remote-analysis-node-b".into(),
                worker: second_worker,
                driver_fence: 41,
                session_id: [32; 16],
                run_key,
                limits: FrameLimits::default(),
            },
            &PinnedQuicEndpoint {
                authority: second_authority,
                server_name: "runmat.execution".into(),
                certificate_der,
            },
        )
        .await
        .unwrap();
        let pool = RemotePoolDriver::new_with_value_scope(
            scope_id,
            PoolSpec {
                id: pool_id,
                min_workers: 2,
                max_workers: 2,
                max_in_flight: 2,
                resource_limit: scaled_inventory(2),
            },
            41,
            bundle_bytes,
            "remote-analysis",
        )
        .unwrap();
        pool.add_worker(first_channel.clone()).await.unwrap();
        pool.add_worker(second_channel.clone()).await.unwrap();
        for object in &program.objects {
            pool.register_execution_object(object.reference.clone(), object.bytes.clone())
                .unwrap();
        }
        for object in &second_program.objects {
            pool.register_execution_object(object.reference.clone(), object.bytes.clone())
                .unwrap();
        }
        let mut completions = pool
            .submit_graph(vec![
                (
                    request.clone(),
                    analysis_submission(
                        scope_id,
                        pool_id,
                        &host,
                        &request,
                        Some(&resource_plan),
                        resource_plan.request.clone(),
                    ),
                ),
                (
                    second_request.clone(),
                    analysis_submission(
                        scope_id,
                        pool_id,
                        &second_host,
                        &second_request,
                        Some(&second_resource_plan),
                        second_resource_plan.request.clone(),
                    ),
                ),
            ])
            .unwrap();
        let second_completion = completions.pop().unwrap();
        let completion = completions.pop().unwrap();
        let success = tokio::time::timeout(Duration::from_secs(15), completion.wait())
            .await
            .expect("remote analysis completion timeout")
            .unwrap();
        let second_success =
            tokio::time::timeout(Duration::from_secs(15), second_completion.wait())
                .await
                .expect("second remote analysis completion timeout")
                .unwrap();
        let mut result_store = publication_chain::MemoryStore::default();
        for reference in success
            .result_objects
            .iter()
            .chain(&second_success.result_objects)
        {
            result_store.0.insert(
                reference.logical_digest,
                pool.execution_object(reference)
                    .unwrap()
                    .expect("verified remote analysis result")
                    .to_vec(),
            );
        }
        let payload = import_result(&result_store, success_response(&success), &host);
        assert_eq!(payload, expected_payload);
        assert_eq!(
            import_result(
                &result_store,
                success_response(&second_success),
                &second_host,
            ),
            expected_payload
        );
        let solve = PreparedStructuralLinearProgram {
            host: host.clone(),
            program: program.clone(),
            inputs: input_objects,
            resource_plan,
        };
        let mut publication_driver = publication_chain::RemoteDriver {
            pool: Arc::clone(&pool),
            store: result_store,
            scope_id,
            pool_id,
        };
        let publication = publication_chain::execute_publication_chain(
            &mut publication_driver,
            &solve,
            &success,
            revision.clone(),
        )
        .await;
        assert_eq!(publication, expected_publication);
        let sequence = execute_structural_sequence_remotely(
            Arc::clone(&pool),
            sequence_preparation(revision.clone(), "remote-analysis"),
            sequence_policy(),
        )
        .await
        .unwrap();
        assert_sequence(&sequence);
        first_channel.drain().await.unwrap();
        second_channel.drain().await.unwrap();
    };
    let (first_server, second_server, ()) = tokio::join!(first_server, second_server, client);
    first_server.unwrap();
    second_server.unwrap();
}

fn import_result(
    store: &impl CacheImport,
    response: ProgramExecutionResponse,
    host: &AnalysisHostWorkload,
) -> Vec<u8> {
    let output_set = import_analysis_output_set(store, response, host).unwrap();
    let [result, _receipt] = output_set.manifest.outputs.as_slice() else {
        panic!("analysis result set must contain result and numerical receipt")
    };
    let result_identity = runmat_analysis_fea::execution::SolverResultIdentity::new(
        host.workload.identity,
        result.payload_digest,
    )
    .unwrap();
    import_analysis_objects(
        store,
        &result.root,
        AnalysisNumericalObjectKind::SolverResult,
        result_identity.digest(),
        result.payload_digest,
        &host.artifact_access,
        ObjectInventoryLimits::default(),
    )
    .unwrap()
    .payload()
}

fn fixture(
    revision: ProgramRevision,
    authorization_scope: &str,
) -> (
    AnalysisHostWorkload,
    PreparedAnalysisProgram,
    Vec<runmat_analysis_execution::PreparedAnalysisObjects>,
    runmat_analysis_fea::planning::resource::FeaResourcePlan,
) {
    fixture_for_study(revision, authorization_scope, digest("authored-study"))
}

fn fixture_for_study(
    revision: ProgramRevision,
    authorization_scope: &str,
    authored_study_digest: Digest,
) -> (
    AnalysisHostWorkload,
    PreparedAnalysisProgram,
    Vec<runmat_analysis_execution::PreparedAnalysisObjects>,
    runmat_analysis_fea::planning::resource::FeaResourcePlan,
) {
    let mut model = fixture_model(FixtureId::CantileverLinearStatic);
    model.sequence.steps[0].observations = vec![structural_displacement_observation()];
    let step_id = model.sequence.steps[0].step_id.clone();
    let solve = StructuralLinearWorkload {
        schema_version: 1,
        model,
        step_id,
        backend: ComputeBackend::Cpu,
        formulation_requirement: runmat_runtime::analysis::FormulationRequirement::Experimental,
        preconditioner: SpdPreconditionerKind::Jacobi,
        algebra_backend: LinearAlgebraBackendKind::CpuReference,
        require_solver_mesh_for_solid: true,
        thermo_mechanical_context: None,
        electro_thermal_context: None,
        initial_guess_transition: None,
    };
    let mesh = solver_mesh(&solve);
    let access = ArtifactAccess {
        authorization_scope: authorization_scope.into(),
        encryption_context: digest("encryption"),
    };
    let prepared = prepare_structural_linear_program(StructuralLinearProgramPreparation {
        authored_study_digest,
        frozen_project_digest: None,
        program_revision: revision,
        artifact_access: access,
        workload: solve,
        mesh,
        initial_guess: None,
        chunk_bytes: DEFAULT_ANALYSIS_CHUNK_BYTES,
        inventory_limits: ObjectInventoryLimits::default(),
        required_capabilities: BTreeSet::new(),
    })
    .unwrap();
    (
        prepared.host,
        prepared.program,
        prepared.inputs,
        prepared.resource_plan,
    )
}

fn structural_displacement_observation() -> runmat_analysis_core::ObservationRequest {
    runmat_analysis_core::ObservationRequest::Field(runmat_analysis_core::ObservationSpec {
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
        cost: runmat_analysis_core::ObservationCostEstimate {
            operation_count: 1_000,
            peak_resident_bytes: 1_000_000,
            stored_bytes: 1_000_000,
            transfer_bytes: 1_000_000,
            collective_count: 0,
        },
    })
}

fn sequence_preparation(
    revision: ProgramRevision,
    authorization_scope: &str,
) -> StructuralSequencePreparation {
    let mut model = fixture_model(FixtureId::CantileverLinearStatic);
    let mut root = model.sequence.steps[0].clone();
    root.step_id = runmat_analysis_core::ProcedureStepId("preload".into());
    root.state_outputs[0].persistence = runmat_analysis_core::StatePersistence::Canonical;
    let state_id = root.state_outputs[0].state_id.clone();
    let branch = |step_id: &str| {
        let mut step = root.clone();
        step.step_id = runmat_analysis_core::ProcedureStepId(step_id.into());
        step.state_inputs = vec![runmat_analysis_core::StateInput {
            input_id: format!("{step_id}-initial-guess"),
            required_kind: runmat_analysis_core::StateKind::StructuralPrimary,
            source: runmat_analysis_core::StateSource::Upstream {
                producer_step: root.step_id.clone(),
                state_id: state_id.clone(),
            },
            use_as: runmat_analysis_core::StateUse::InitialGuess,
            conflict: runmat_analysis_core::InitialStateConflictPolicy::Reject,
        }];
        step
    };
    let right = branch("right");
    let left = branch("left");
    let transition =
        |consumer: &runmat_analysis_core::ProcedureStepId| runmat_analysis_core::StateTransition {
            transition_id: runmat_analysis_core::TransitionId(format!(
                "preload-to-{}",
                consumer.as_str()
            )),
            producer_step: root.step_id.clone(),
            consumer_step: consumer.clone(),
            state_id: state_id.clone(),
            method: runmat_analysis_core::StateTransitionMethod::InitialGuessOnly,
            topology: runmat_analysis_core::TopologyTransitionPolicy::RequireIdentical,
            reference: runmat_analysis_core::ReferenceStatePolicy::Preserve,
        };
    model.sequence = runmat_analysis_core::AnalysisSequence {
        sequence_id: "process-preload-branches".into(),
        steps: vec![root.clone(), right.clone(), left.clone()],
        dependencies: vec![
            runmat_analysis_core::SequenceDependency {
                predecessor: root.step_id.clone(),
                successor: right.step_id.clone(),
                reason: runmat_analysis_core::SequenceDependencyReason::StateTransfer,
            },
            runmat_analysis_core::SequenceDependency {
                predecessor: root.step_id.clone(),
                successor: left.step_id.clone(),
                reason: runmat_analysis_core::SequenceDependencyReason::StateTransfer,
            },
        ],
        transitions: vec![transition(&right.step_id), transition(&left.step_id)],
        branches: vec![runmat_analysis_core::SequenceBranch {
            branch_id: runmat_analysis_core::SequenceBranchId("post-preload".into()),
            producer_step: root.step_id.clone(),
            state_ids: vec![state_id],
            consumer_steps: vec![right.step_id.clone(), left.step_id.clone()],
        }],
    };
    let workload = StructuralLinearWorkload {
        schema_version: 1,
        model,
        step_id: root.step_id,
        backend: ComputeBackend::Cpu,
        formulation_requirement: runmat_runtime::analysis::FormulationRequirement::Experimental,
        preconditioner: SpdPreconditionerKind::Jacobi,
        algebra_backend: LinearAlgebraBackendKind::CpuReference,
        require_solver_mesh_for_solid: true,
        thermo_mechanical_context: None,
        electro_thermal_context: None,
        initial_guess_transition: None,
    };
    let mesh = solver_mesh(&workload);
    let workloads = workload
        .model
        .sequence
        .steps
        .iter()
        .map(|step| {
            let mut step_workload = workload.clone();
            step_workload.step_id = step.step_id.clone();
            (step.step_id.clone(), step_workload)
        })
        .collect();
    StructuralSequencePreparation {
        authored_study_digest: digest("process-sequence-study"),
        frozen_project_digest: None,
        program_revision: revision,
        artifact_access: ArtifactAccess {
            authorization_scope: authorization_scope.into(),
            encryption_context: digest("process-sequence-encryption"),
        },
        workloads,
        mesh,
        chunk_bytes: DEFAULT_ANALYSIS_CHUNK_BYTES,
        inventory_limits: ObjectInventoryLimits::default(),
        required_capabilities: BTreeSet::new(),
    }
}

fn sequence_policy() -> StructuralSequenceExecutionPolicy {
    StructuralSequenceExecutionPolicy {
        resources: ResourceRequest {
            cpu_millicores: 1_000,
            memory_bytes: 256 * 1024 * 1024,
            scratch_bytes: 256 * 1024 * 1024,
            max_wall_millis: 60_000,
            max_artifact_bytes: 64 * 1024 * 1024,
            max_egress_bytes: 64 * 1024 * 1024,
            max_relay_bytes: 64 * 1024 * 1024,
            accelerators: Vec::new(),
            required_capabilities: capabilities(),
        },
        retry: RetryPolicy::IdempotentInfrastructure,
        deadline_unix_millis: None,
        priority: 0,
    }
}

fn assert_sequence(sequence: &runmat_analysis_execution::CompletedStructuralSequence) {
    assert_eq!(
        sequence
            .steps
            .iter()
            .map(|step| step.step_id.as_str())
            .collect::<Vec<_>>(),
        vec!["preload", "right", "left"]
    );
    assert_eq!(sequence.states.len(), 3);
    assert_eq!(
        sequence.steps[1].result.run.fields,
        sequence.steps[2].result.run.fields
    );
    assert!(sequence.steps[1..].iter().all(|step| {
        step.result.run.diagnostics.iter().any(|diagnostic| {
            diagnostic.code == "FEA_CONVERGENCE" && diagnostic.message.contains("iterations=0 ")
        })
    }));
    assert_eq!(
        sequence
            .steps
            .iter()
            .map(|step| step.objects.manifest.logical_identity)
            .collect::<BTreeSet<_>>()
            .len(),
        3
    );
}

fn solver_mesh(solve: &StructuralLinearWorkload) -> runmat_meshing_core::SolverMeshArtifact {
    let mut mesh =
        runmat_meshing_core::fixtures::canonical_tetrahedron_solver_mesh(ElementOrder::Tet4);
    for (index, face) in mesh.topology.boundary_faces.iter_mut().enumerate() {
        face.provenance.push(PersistentEntityId {
            kind: PersistentEntityKind::Face,
            source_topology_id: if index == 0 { "root" } else { "tip" }.into(),
            assembly_path: vec!["root".into()],
        });
        face.provenance.sort();
    }
    let assignment = solve.model.material_assignments.first().unwrap();
    let region = PersistentEntityId {
        kind: PersistentEntityKind::Region,
        source_topology_id: assignment.region_id.clone(),
        assembly_path: vec!["root".into()],
    };
    mesh.topology.volume_elements[0].region_id = region.clone();
    mesh.topology.regions[0].region_id = region;
    mesh.canonical_digest = StableDigest::ZERO;
    mesh.seal_canonical_digest().unwrap();
    mesh
}

fn capabilities() -> BTreeSet<Capability> {
    BTreeSet::from([
        Capability::ProcessIsolation,
        Capability::Custom(format!(
            "runmat.analysis.host:{}",
            runmat_analysis_execution::ANALYSIS_HOST_ABI
        )),
    ])
}

fn inventory() -> ResourceInventory {
    ResourceInventory {
        cpu_millicores: 1_000,
        memory_bytes: 256 * 1024 * 1024,
        scratch_bytes: 256 * 1024 * 1024,
        accelerators: Vec::new(),
        capabilities: capabilities(),
    }
}

fn scaled_inventory(worker_count: u32) -> ResourceInventory {
    let mut inventory = inventory();
    inventory.cpu_millicores *= worker_count;
    inventory.memory_bytes *= u64::from(worker_count);
    inventory.scratch_bytes *= u64::from(worker_count);
    inventory
}

fn analysis_submission(
    scope_id: runmat_execution::ExecutionScopeId,
    pool_id: runmat_execution::PoolId,
    host: &AnalysisHostWorkload,
    request: &runmat_execution_artifact::ProgramExecutionRequest,
    resource_plan: Option<&runmat_analysis_fea::planning::resource::FeaResourcePlan>,
    resources: ResourceRequest,
) -> runmat_execution_runner::TaskSubmission {
    let mut resources = resources;
    resources.required_capabilities.extend(capabilities());
    let roots = request
        .arguments
        .iter()
        .map(|value| match value {
            ValuePayload::Object(root) => (**root).clone(),
            ValuePayload::Inline(_) => panic!("analysis inputs must remain externalized"),
        })
        .collect();
    let context = AnalysisExecutionContext {
        scope_id,
        pool_id,
        program_artifact_id: ArtifactId::derive(&[request.artifact.id.0.bytes()]),
        resources,
        retry: RetryPolicy::IdempotentInfrastructure,
        deadline_unix_millis: None,
        priority: 0,
    };
    match resource_plan {
        Some(resource_plan) => build_structural_task_submission(
            host,
            resource_plan,
            roots,
            Vec::new(),
            BTreeSet::new(),
            context,
        ),
        None => build_analysis_task_submission(host, roots, Vec::new(), BTreeSet::new(), context),
    }
    .unwrap()
}

fn analysis_resources() -> ResourceRequest {
    ResourceRequest {
        cpu_millicores: 1_000,
        memory_bytes: 256 * 1024 * 1024,
        scratch_bytes: 256 * 1024 * 1024,
        max_wall_millis: 60_000,
        max_artifact_bytes: 64 * 1024 * 1024,
        max_egress_bytes: 64 * 1024 * 1024,
        max_relay_bytes: 64 * 1024 * 1024,
        accelerators: Vec::new(),
        required_capabilities: capabilities(),
    }
}

fn revision() -> ProgramRevision {
    ProgramRevision::new(
        digest("graph"),
        digest("source"),
        ProgramEnvironment::new(1, 1, digest("runtime"), digest("catalog"), "matlab").unwrap(),
    )
    .unwrap()
}

fn digest(label: &str) -> Digest {
    Digest::sha256(label)
}
