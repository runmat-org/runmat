use std::collections::BTreeSet;
use std::net::{Ipv4Addr, SocketAddr};
use std::time::Duration;

use rcgen::{generate_simple_self_signed, CertifiedKey};
use runmat_analysis_core::{
    CoordinateFrame, Force, GeneratorDomain, GeneratorEffectPolicy, PhysicalSignal, Signal,
    SignalApproximationEvidence, SignalExtrapolation, SignalGenerator, SignalId,
    SignalInterpolation, SignalOrdinate, SignalPrecision, SignalProvenance, SignalRepresentation,
    SignalSamplingPolicy, SignalShape, SignalValue, TabulatedSignal,
};
use runmat_analysis_execution::{
    admit_named_signal_generator, build_signal_generator_task_submission,
    canonical_tabulated_signal_content_digest, canonicalize_signal_generator_outputs,
    materialize_signal_for_solver, plan_signal_generator_chunks, prepare_signal_materialization,
    PreparedSignalMaterialization, SignalGeneratorChunkOutput, SignalGeneratorExecutionContext,
};
use runmat_analysis_fea::execution::CanonicalSignalRealization;
use runmat_execution::resource::{Capability, ResourceInventory, ResourceRequest};
use runmat_execution::task::RetryPolicy;
use runmat_execution::{Digest, ProgramRevision};
use runmat_execution_artifact::archive::{write_bundle, ArchiveLimits};
use runmat_execution_artifact::{BundleCallable, ExecutionBundleBuilder, ProgramExecutionResponse};
use runmat_execution_runner::{PoolSpec, WorkerSpec};
use runmat_execution_runner_native::{
    run_remote_worker_quic, NativeExecutionConfig, NativeProgramSession, QuicRemoteWorkerChannel,
    RemotePoolDriver, RemoteWorkerChannel, RemoteWorkerChannelConfig, RemoteWorkerQuicRequest,
};
use runmat_execution_transport_native::frame::FrameLimits;
use runmat_execution_transport_native::overlay::{PinnedQuicEndpoint, QuicOverlayListener};

pub(super) async fn run() {
    let fixture = SignalFixture::new().await;
    let expected = canonical_tabulated_signal_content_digest(&fixture.explicit).unwrap();

    let inline = fixture.inline().await;
    assert_realization(&fixture, &inline, expected, "inline");

    let isolated = fixture.isolated().await;
    assert_realization(&fixture, &isolated, expected, "isolated process");

    let durable = fixture.durable().await;
    assert_realization(&fixture, &durable, expected, "durable process");

    let remote = fixture.encrypted_remote().await;
    assert_realization(&fixture, &remote, expected, "encrypted remote");
}

struct SignalFixture {
    _project_root: tempfile::TempDir,
    project: runmat_package::FrozenProject,
    revision: ProgramRevision,
    prepared: PreparedSignalMaterialization,
    explicit: PhysicalSignal,
}

impl SignalFixture {
    async fn new() -> Self {
        let project_root = tempfile::tempdir().unwrap();
        std::fs::create_dir_all(project_root.path().join("src")).unwrap();
        std::fs::write(
            project_root.path().join("runmat.toml"),
            "[package]\nname = \"signal-placement-conformance\"\n[sources]\nroots = [\"src\"]\n",
        )
        .unwrap();
        let source =
            "function y = pulse(t)\narguments\n    t (:,1) double\nend\ny = t * [1, -2, 3];\nend\n";
        let source_path = project_root.path().join("src/pulse.m");
        std::fs::write(&source_path, source).unwrap();
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
        let authored = generated_signal();
        let symbols = runmat_package::source_symbols_from_frozen(&project, &source_path);
        let definition = symbols
            .definitions
            .iter()
            .find(|definition| definition.name == "pulse")
            .unwrap();
        let source_id = definition.source_id.as_ref().unwrap();
        let owner_identity = source_id.package_instance.to_string();
        let mut session = runmat_core::RunMatSession::with_options(false, false).unwrap();
        session
            .install_project_handoff(runmat_package::FrozenProjectHandoff::new(project.clone()))
            .unwrap();
        let unit = session
            .compile_executable_unit(
                runmat_core::ExecutableSource::frozen(
                    owner_identity.clone(),
                    source_id.relative_path.as_str(),
                    source_path.to_string_lossy(),
                    source,
                ),
                Some(revision.clone()),
            )
            .await
            .unwrap();
        let executable = unit.portable_envelope_for(Some("pulse")).unwrap();
        let function = unit
            .function_analysis(executable.manifest.identity.entrypoint_function)
            .cloned()
            .unwrap();
        let admitted = admit_named_signal_generator(
            &authored,
            revision.clone(),
            BundleCallable {
                owner_identity,
                qualified_name: "loads.pulse".into(),
                source_digest: Digest::sha256(source),
            },
            unit.revision().source_digest.parse::<Digest>().unwrap(),
            &function,
            &BTreeSet::new(),
        )
        .unwrap();
        let chunks = plan_signal_generator_chunks(
            &authored,
            &admitted,
            None,
            usize::try_from(runmat_execution::value::ValueLimits::default().max_inline_bytes)
                .unwrap(),
            |_| panic!("fixture has no artifact arguments"),
        )
        .unwrap();
        let prepared = prepare_signal_materialization(
            authored,
            admitted,
            executable,
            chunks.chunks,
            Vec::new(),
            chunks.input_admissions,
        )
        .unwrap();
        Self {
            _project_root: project_root,
            project,
            revision,
            prepared,
            explicit: explicit_signal(),
        }
    }

    async fn inline(&self) -> CanonicalSignalRealization {
        let mut outputs = Vec::new();
        for chunk in &self.prepared.program.chunks {
            outputs.push(response_output(
                chunk.plan.identity,
                runmat_vm::execute_program_request(chunk.request.clone()).await,
            ));
        }
        self.canonicalize(outputs)
    }

    async fn isolated(&self) -> CanonicalSignalRealization {
        let directory = tempfile::tempdir().unwrap();
        let mut config = NativeExecutionConfig::for_current_executable().unwrap();
        config.executable = std::env::current_exe().unwrap();
        config.worker_arguments = vec!["--child".into()];
        config.max_workers = 2;
        config.store_root = directory.path().join("isolated-signal");
        let session = NativeProgramSession::new(config).unwrap();
        let mut outputs = Vec::new();
        for chunk in &self.prepared.program.chunks {
            let task = session
                .submit(
                    chunk.request.clone(),
                    signal_submission(&self.prepared, chunk, session.scope_id(), session.pool_id()),
                )
                .unwrap();
            let success = super::wait_for_native_task(task).await;
            assert!(success.result_objects.is_empty());
            assert_eq!(success.outputs.len(), 1);
            outputs.push(SignalGeneratorChunkOutput {
                chunk_identity: chunk.plan.identity,
                value: success.outputs[0].clone(),
            });
        }
        self.canonicalize(outputs)
    }

    async fn durable(&self) -> CanonicalSignalRealization {
        let directory = tempfile::tempdir().unwrap();
        let supervisor = runmat_execution_runner_native::supervisor::LocalSupervisor::open(
            runmat_execution_runner_native::supervisor::LocalSupervisorConfig {
                executable: std::env::current_exe().unwrap(),
                paths: runmat_execution_runner_native::supervisor::SupervisorPaths::new(
                    directory.path().join("durable-signal"),
                )
                .unwrap(),
                max_stderr_bytes: 1024 * 1024,
                max_object_bytes: 64 * 1024 * 1024,
            },
        )
        .unwrap();
        let mut outputs = Vec::new();
        for (index, chunk) in self.prepared.program.chunks.iter().enumerate() {
            let (record, duplicate) = supervisor
                .submit_program(
                    runmat_execution_runner_native::supervisor::ProgramBatchSubmission::from_request(
                        chunk.request.clone(),
                        Some(format!("signal-placement-{index}")),
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
                            panic!("durable signal failed: {:?}", attachment.record.message)
                        });
                    }
                    tokio::time::sleep(Duration::from_millis(20)).await;
                }
            })
            .await
            .expect("durable signal completion timeout");
            outputs.push(response_output(chunk.plan.identity, response));
        }
        self.canonicalize(outputs)
    }

    async fn encrypted_remote(&self) -> CanonicalSignalRealization {
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
        let scope_id = runmat_execution::ExecutionScopeId::derive(&[b"signal-conformance-scope"]);
        let pool_id = runmat_execution::PoolId::derive(&[b"signal-conformance-pool"]);
        let inventory = ResourceInventory {
            cpu_millicores: 1_000,
            memory_bytes: 512 * 1024 * 1024,
            scratch_bytes: 128 * 1024 * 1024,
            accelerators: Vec::new(),
            capabilities: BTreeSet::from([Capability::ProcessIsolation]),
        };
        let worker = WorkerSpec {
            id: runmat_execution::identity::WorkerId::derive(&[b"signal-conformance-worker"]),
            pool_id,
            resources: inventory.clone(),
        };
        let run_key =
            runmat_execution_artifact::encryption::RunKeyMaterial::from_entropy([79; 32]).unwrap();
        let server = run_remote_worker_quic(RemoteWorkerQuicRequest {
            listener,
            run_identity: "signal-conformance".into(),
            worker: worker.clone(),
            driver_fence: 83,
            session_id: [89; 16],
            run_key: run_key.clone(),
            limits: FrameLimits::default(),
        });
        let request = self.prepared.program.chunks[0].request.clone();
        let bundle = ExecutionBundleBuilder::native(&self.project, self.revision.clone())
            .unwrap()
            .with_compiled_package_closure()
            .with_materialized_program(
                request.recipe.clone(),
                request.artifact.form,
                request.artifact.executable_bytes.clone(),
            )
            .build()
            .unwrap();
        let mut bundle_bytes = Vec::new();
        write_bundle(&bundle, &mut bundle_bytes, ArchiveLimits::default()).unwrap();
        let prepared = &self.prepared;
        let client = async {
            let channel = QuicRemoteWorkerChannel::connect(
                RemoteWorkerChannelConfig {
                    run_identity: "signal-conformance".into(),
                    node_identity: "signal-conformance-node".into(),
                    worker: worker.clone(),
                    driver_fence: 83,
                    session_id: [89; 16],
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
                    resource_limit: inventory,
                },
                83,
                bundle_bytes,
                "signal-conformance",
            )
            .unwrap();
            pool.add_worker(channel.clone()).await.unwrap();
            let completions = pool
                .submit_graph(
                    prepared
                        .program
                        .chunks
                        .iter()
                        .map(|chunk| {
                            (
                                chunk.request.clone(),
                                signal_submission(prepared, chunk, scope_id, pool_id),
                            )
                        })
                        .collect(),
                )
                .unwrap();
            let mut outputs = Vec::with_capacity(completions.len());
            for (chunk, completion) in prepared.program.chunks.iter().zip(completions) {
                let success = tokio::time::timeout(Duration::from_secs(15), completion.wait())
                    .await
                    .expect("remote signal completion timeout")
                    .unwrap();
                assert!(success.result_objects.is_empty());
                assert_eq!(success.outputs.len(), 1);
                outputs.push(SignalGeneratorChunkOutput {
                    chunk_identity: chunk.plan.identity,
                    value: success.outputs[0].clone(),
                });
            }
            channel.drain().await.unwrap();
            outputs
        };
        let (server_result, outputs) = tokio::join!(server, client);
        server_result.unwrap();
        self.canonicalize(outputs)
    }

    fn canonicalize(&self, outputs: Vec<SignalGeneratorChunkOutput>) -> CanonicalSignalRealization {
        canonicalize_signal_generator_outputs(
            &self.prepared.authored,
            &self.prepared.admitted,
            None,
            &self.prepared.input_admissions,
            &self
                .prepared
                .program
                .chunks
                .iter()
                .map(|chunk| chunk.plan.clone())
                .collect::<Vec<_>>(),
            &outputs,
        )
        .unwrap()
    }
}

fn signal_submission(
    prepared: &PreparedSignalMaterialization,
    chunk: &runmat_analysis_execution::PreparedSignalGeneratorChunk,
    scope_id: runmat_execution::ExecutionScopeId,
    pool_id: runmat_execution::PoolId,
) -> runmat_execution_runner::TaskSubmission {
    build_signal_generator_task_submission(
        &prepared.admitted,
        chunk,
        BTreeSet::new(),
        SignalGeneratorExecutionContext {
            scope_id,
            pool_id,
            resources: ResourceRequest {
                cpu_millicores: 1_000,
                memory_bytes: 512 * 1024 * 1024,
                scratch_bytes: 128 * 1024 * 1024,
                max_wall_millis: 15_000,
                max_artifact_bytes: 64 * 1024 * 1024,
                max_egress_bytes: 0,
                max_relay_bytes: 0,
                accelerators: Vec::new(),
                required_capabilities: BTreeSet::from([Capability::ProcessIsolation]),
            },
            retry: RetryPolicy::IdempotentInfrastructure,
            deadline_unix_millis: None,
            priority: 0,
        },
    )
    .unwrap()
}

fn response_output(
    identity: Digest,
    response: ProgramExecutionResponse,
) -> SignalGeneratorChunkOutput {
    let ProgramExecutionResponse::Success { value } = response else {
        panic!("signal generator did not return an inline value: {response:?}")
    };
    SignalGeneratorChunkOutput {
        chunk_identity: identity,
        value,
    }
}

fn assert_realization(
    fixture: &SignalFixture,
    realization: &CanonicalSignalRealization,
    expected: Digest,
    placement: &str,
) {
    assert_eq!(
        realization.content_digest, expected,
        "{placement}: realization={realization:#?}; explicit={:#?}",
        fixture.explicit
    );
    let table = materialize_signal_for_solver(
        &fixture.prepared.authored,
        &fixture.prepared.admitted,
        realization,
    )
    .unwrap();
    assert_eq!(
        canonical_tabulated_signal_content_digest(&table).unwrap(),
        expected,
        "{placement} solver table"
    );
}

fn generated_signal() -> PhysicalSignal {
    PhysicalSignal::Force(Signal::<Force> {
        signal_id: SignalId("placement_force".into()),
        shape: SignalShape::Vector3,
        frame: Some(CoordinateFrame::Global),
        representation: SignalRepresentation::Generator(SignalGenerator {
            qualified_function: "loads.pulse".into(),
            arguments: Vec::new(),
            domain: GeneratorDomain::TimeSeconds,
            sampling: SignalSamplingPolicy::Uniform {
                start: 0.0,
                end: 1.0,
                sample_count: 5,
            },
            interpolation: SignalInterpolation::Linear,
            extrapolation: SignalExtrapolation::Reject,
            effects: GeneratorEffectPolicy::Pure,
        }),
        precision: SignalPrecision::Real64,
        evidence: SignalApproximationEvidence {
            max_absolute_error: None,
            max_relative_error: None,
            derivative_max_absolute_error: None,
            sample_count: 0,
        },
        provenance: SignalProvenance::Authored,
    })
}

fn explicit_signal() -> PhysicalSignal {
    let points = [0.0, 0.25, 0.5, 0.75, 1.0];
    PhysicalSignal::Force(Signal::<Force> {
        signal_id: SignalId("explicit_force".into()),
        shape: SignalShape::Vector3,
        frame: Some(CoordinateFrame::Global),
        representation: SignalRepresentation::Tabulated(TabulatedSignal {
            axis: runmat_analysis_core::SignalAxis::TimeSeconds(points.to_vec()),
            values: points
                .into_iter()
                .map(|point| {
                    SignalOrdinate::Real(SignalValue::Vector3([
                        Force::new(point),
                        Force::new(-2.0 * point),
                        Force::new(3.0 * point),
                    ]))
                })
                .collect(),
            interpolation: SignalInterpolation::Linear,
            extrapolation: SignalExtrapolation::Reject,
            breakpoints: Vec::new(),
        }),
        precision: SignalPrecision::Real64,
        evidence: SignalApproximationEvidence {
            max_absolute_error: None,
            max_relative_error: None,
            derivative_max_absolute_error: None,
            sample_count: 5,
        },
        provenance: SignalProvenance::Authored,
    })
}
