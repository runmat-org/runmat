use std::collections::BTreeSet;
use std::net::{Ipv4Addr, SocketAddr};
use std::time::Duration;

use rcgen::{generate_simple_self_signed, CertifiedKey};
use runmat_analysis_execution::{
    build_analysis_task_submission, import_analysis_objects, prepare_structural_linear_program,
    AnalysisExecutionContext, AnalysisHostWorkload, StructuralLinearProgramPreparation,
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
use runmat_execution::value::{ValuePayload, ValueRefKind};
use runmat_execution::{Digest, ProgramEnvironment, ProgramRevision};
use runmat_execution_artifact::cache::{CacheExport, CacheImport};
use runmat_execution_artifact::object::ObjectInventoryLimits;
use runmat_execution_artifact::{
    archive::{write_bundle, ArchiveLimits},
    ArtifactAccess, ExecutableForm, ExecutionBundleBuilder, ProgramExecutionResponse,
};
use runmat_execution_runner::{PoolSpec, WorkerSpec};
use runmat_execution_runner_native::{
    execute_analysis_program_request, run_remote_worker_quic, NativeAnalysisHostLimits,
    NativeAnalysisKernel, NativeExecutionConfig, NativeProgramSession, QuicRemoteWorkerChannel,
    RemotePoolDriver, RemoteWorkerChannel, RemoteWorkerChannelConfig, RemoteWorkerQuicRequest,
};
use runmat_execution_transport_native::frame::FrameLimits;
use runmat_execution_transport_native::overlay::{PinnedQuicEndpoint, QuicOverlayListener};
use runmat_meshing_core::{ElementOrder, PersistentEntityId, PersistentEntityKind, StableDigest};

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
    let (host, request, input_objects) = fixture(revision(), "analysis-process-conformance");
    let mut config = NativeExecutionConfig::for_current_executable().unwrap();
    config.executable = std::env::current_exe().unwrap();
    config.worker_arguments = vec!["--child".into()];
    config.max_workers = 1;
    config.store_root = directory.path().join("native-analysis-session");
    config.worker_capabilities = capabilities();
    let session = NativeProgramSession::new(config).unwrap();
    let mut store = session.object_store();
    for objects in &input_objects {
        for object in &objects.objects {
            store.write_verified(object).unwrap();
        }
    }
    let roots = request
        .arguments
        .iter()
        .map(|value| match value {
            ValuePayload::Object(root) => (**root).clone(),
            ValuePayload::Inline(_) => panic!("analysis inputs must remain externalized"),
        })
        .collect::<Vec<_>>();
    let submission = build_analysis_task_submission(
        &host.workload,
        roots,
        Vec::new(),
        BTreeSet::new(),
        AnalysisExecutionContext {
            scope_id: session.scope_id(),
            pool_id: session.pool_id(),
            program_artifact_id: ArtifactId::derive(&[request.artifact.id.0.bytes()]),
            resources: ResourceRequest {
                cpu_millicores: 1_000,
                memory_bytes: 256 * 1024 * 1024,
                scratch_bytes: 256 * 1024 * 1024,
                max_wall_millis: 60_000,
                max_artifact_bytes: 64 * 1024 * 1024,
                max_egress_bytes: 0,
                max_relay_bytes: 0,
                accelerators: Vec::new(),
                required_capabilities: capabilities(),
            },
            retry: RetryPolicy::IdempotentInfrastructure,
            deadline_unix_millis: None,
            priority: 0,
        },
    )
    .unwrap();
    let task = session.submit(request.clone(), submission).unwrap();
    let process_result = tokio::time::timeout(Duration::from_secs(15), async {
        loop {
            if let Some(result) = task.try_result() {
                break result.unwrap();
            }
            tokio::time::sleep(Duration::from_millis(10)).await;
        }
    })
    .await
    .unwrap();
    let process_payload = import_result(&store, &process_result.outputs, &host);

    let mut inline_store = store.clone();
    let inline = execute_analysis_program_request(
        &request,
        &mut inline_store,
        &NativeAnalysisKernel,
        NativeAnalysisHostLimits::default(),
    );
    let ProgramExecutionResponse::ExternalizedSuccess {
        outputs: inline_outputs,
        ..
    } = inline
    else {
        panic!("inline analysis execution failed: {inline:?}");
    };
    let inline_payload = import_result(&inline_store, &inline_outputs, &host);
    assert_eq!(process_payload, inline_payload);

    let decoded = StructuralLinearWorkloadResult::from_canonical_bytes(&process_payload).unwrap();
    assert!(!decoded.run.fields_are_empty());
    assert!(decoded
        .run
        .diagnostics
        .iter()
        .all(|diagnostic| !diagnostic.message.contains("artifact_path")));

    remote_conformance(process_payload).await;
    durable_conformance(inline_payload).await;
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

async fn durable_conformance(expected_payload: Vec<u8>) {
    let directory = tempfile::tempdir().unwrap();
    let paths = runmat_execution_runner_native::supervisor::SupervisorPaths::new(
        directory.path().join("durable-analysis"),
    )
    .unwrap();
    let supervisor = runmat_execution_runner_native::supervisor::LocalSupervisor::open(
        runmat_execution_runner_native::supervisor::LocalSupervisorConfig {
            executable: std::env::current_exe().unwrap(),
            paths,
            max_stderr_bytes: 1024 * 1024,
            max_object_bytes: 64 * 1024 * 1024,
        },
    )
    .unwrap();
    let (host, request, input_objects) = fixture(revision(), "durable-analysis");
    let mut store = supervisor.object_store().unwrap();
    for prepared in input_objects {
        for object in prepared.objects {
            store.write_verified(&object).unwrap();
        }
    }
    let (record, duplicate) = supervisor
        .submit_program(
            runmat_execution_runner_native::supervisor::ProgramBatchSubmission::from_request(
                request,
                Some("analysis-conformance".into()),
                runmat_execution_runner_native::supervisor::MIN_RETENTION_MILLIS,
            ),
        )
        .await
        .unwrap();
    assert!(!duplicate);
    let response = tokio::time::timeout(Duration::from_secs(15), async {
        loop {
            supervisor.reconcile().await.unwrap();
            let attachment = supervisor.attach(record.handle.id, 0, 0).await.unwrap();
            if attachment.record.state.is_terminal() {
                break attachment.response.unwrap_or_else(|| {
                    panic!(
                        "durable analysis failed without response: {:?}",
                        attachment.record.message
                    )
                });
            }
            tokio::time::sleep(Duration::from_millis(20)).await;
        }
    })
    .await
    .expect("durable analysis completion timeout");
    let ProgramExecutionResponse::ExternalizedSuccess { outputs, .. } = response else {
        panic!("durable analysis did not externalize its result: {response:?}")
    };
    assert_eq!(import_result(&store, &outputs, &host), expected_payload);
}

async fn remote_conformance(expected_payload: Vec<u8>) {
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
    let (host, request, input_objects) = fixture(revision.clone(), "remote-analysis");
    let bundle = ExecutionBundleBuilder::native(&project, revision)
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
    let listener = QuicOverlayListener::bind(
        SocketAddr::from((Ipv4Addr::LOCALHOST, 0)),
        vec![certificate_der.clone()],
        signing_key.serialize_der(),
        FrameLimits::default(),
    )
    .unwrap();
    let authority = listener.local_addr().unwrap();
    let scope_id = runmat_execution::ExecutionScopeId::derive(&[b"remote-analysis-scope"]);
    let pool_id = runmat_execution::PoolId::derive(&[b"remote-analysis-pool"]);
    let worker = WorkerSpec {
        id: runmat_execution::identity::WorkerId::derive(&[b"remote-analysis-worker"]),
        pool_id,
        resources: inventory(),
    };
    let run_key =
        runmat_execution_artifact::encryption::RunKeyMaterial::from_entropy([29; 32]).unwrap();
    let server = run_remote_worker_quic(RemoteWorkerQuicRequest {
        listener,
        run_identity: "remote-analysis".into(),
        worker: worker.clone(),
        driver_fence: 41,
        session_id: [31; 16],
        run_key: run_key.clone(),
        limits: FrameLimits::default(),
    });
    let client = async {
        let channel = QuicRemoteWorkerChannel::connect(
            RemoteWorkerChannelConfig {
                run_identity: "remote-analysis".into(),
                node_identity: "remote-analysis-node".into(),
                worker,
                driver_fence: 41,
                session_id: [31; 16],
                run_key,
                limits: FrameLimits::default(),
            },
            &PinnedQuicEndpoint {
                authority,
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
                min_workers: 1,
                max_workers: 1,
                max_in_flight: 1,
                resource_limit: inventory(),
            },
            41,
            bundle_bytes,
            "remote-analysis",
        )
        .unwrap();
        pool.add_worker(channel.clone()).await.unwrap();
        for prepared in &input_objects {
            let references = prepared
                .references(&host.artifact_access, ValueRefKind::DriverObject)
                .unwrap();
            for (object, reference) in prepared.objects.iter().zip(references) {
                pool.register_execution_object(reference, object.bytes.clone())
                    .unwrap();
            }
        }
        let submission = analysis_submission(scope_id, pool_id, &host, &request);
        let completion = pool.submit(submission, request).unwrap();
        let success = tokio::time::timeout(Duration::from_secs(15), completion.wait())
            .await
            .expect("remote analysis completion timeout")
            .unwrap();
        let mut result_store = MemoryStore::default();
        for reference in &success.result_objects {
            result_store.0.insert(
                reference.logical_digest,
                pool.execution_object(reference)
                    .unwrap()
                    .expect("verified remote analysis result")
                    .to_vec(),
            );
        }
        let payload = import_result(&result_store, &success.outputs, &host);
        assert_eq!(payload, expected_payload);
        channel.drain().await.unwrap();
    };
    let (server, ()) = tokio::join!(server, client);
    server.unwrap();
}

fn import_result(
    store: &impl CacheImport,
    outputs: &[ValuePayload],
    host: &AnalysisHostWorkload,
) -> Vec<u8> {
    let [ValuePayload::Object(root)] = outputs else {
        panic!("analysis result must contain one externalized root")
    };
    let manifest_bytes = store.read_verified(root.logical_digest).unwrap().unwrap();
    let manifest = runmat_analysis_fea::execution::AnalysisObjectManifest::from_canonical_bytes(
        &manifest_bytes,
    )
    .unwrap();
    import_analysis_objects(
        store,
        root,
        AnalysisNumericalObjectKind::SemanticState,
        host.workload.execution_plan.digest(),
        manifest.payload_digest,
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
    runmat_execution_artifact::ProgramExecutionRequest,
    Vec<runmat_analysis_execution::PreparedAnalysisObjects>,
) {
    let model = fixture_model(FixtureId::CantileverLinearStatic);
    let solve = StructuralLinearWorkload {
        schema_version: 1,
        model,
        backend: ComputeBackend::Cpu,
        formulation_requirement: runmat_runtime::analysis::FormulationRequirement::Experimental,
        preconditioner: SpdPreconditionerKind::Jacobi,
        algebra_backend: LinearAlgebraBackendKind::CpuReference,
        require_solver_mesh_for_solid: true,
        thermo_mechanical_context: None,
        electro_thermal_context: None,
    };
    let mesh = solver_mesh(&solve);
    let access = ArtifactAccess {
        authorization_scope: authorization_scope.into(),
        encryption_context: digest("encryption"),
    };
    let prepared = prepare_structural_linear_program(StructuralLinearProgramPreparation {
        authored_study_digest: digest("authored-study"),
        frozen_project_digest: None,
        program_revision: revision,
        artifact_access: access,
        workload: solve,
        mesh,
        chunk_bytes: DEFAULT_ANALYSIS_CHUNK_BYTES,
        inventory_limits: ObjectInventoryLimits::default(),
        required_capabilities: BTreeSet::new(),
    })
    .unwrap();
    (prepared.host, prepared.program.request, prepared.inputs)
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

fn analysis_submission(
    scope_id: runmat_execution::ExecutionScopeId,
    pool_id: runmat_execution::PoolId,
    host: &AnalysisHostWorkload,
    request: &runmat_execution_artifact::ProgramExecutionRequest,
) -> runmat_execution_runner::TaskSubmission {
    let roots = request
        .arguments
        .iter()
        .map(|value| match value {
            ValuePayload::Object(root) => (**root).clone(),
            ValuePayload::Inline(_) => panic!("analysis inputs must remain externalized"),
        })
        .collect();
    build_analysis_task_submission(
        &host.workload,
        roots,
        Vec::new(),
        BTreeSet::new(),
        AnalysisExecutionContext {
            scope_id,
            pool_id,
            program_artifact_id: ArtifactId::derive(&[request.artifact.id.0.bytes()]),
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
        },
    )
    .unwrap()
}

#[derive(Default)]
struct MemoryStore(std::collections::BTreeMap<Digest, Vec<u8>>);

impl CacheImport for MemoryStore {
    fn read_verified(
        &self,
        digest: Digest,
    ) -> runmat_execution_artifact::ArtifactResult<Option<Vec<u8>>> {
        Ok(self.0.get(&digest).cloned())
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
