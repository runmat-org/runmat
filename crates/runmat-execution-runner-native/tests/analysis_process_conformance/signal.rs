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
    admit_callable_signal_generator, build_generator_task_submission,
    canonical_tabulated_signal_content_digest, canonicalize_signal_generator_outputs,
    materialize_signal_for_solver, plan_signal_generator_chunks, prepare_signal_materialization,
    GeneratorChunkOutput, GeneratorExecutionContext, PreparedSignalMaterialization,
};
use runmat_analysis_fea::execution::CanonicalSignalRealization;
use runmat_execution::resource::{Capability, ResourceInventory, ResourceRequest};
use runmat_execution::task::RetryPolicy;
use runmat_execution::value::{CallableValue, DenseValue, ElementType, ValuePayload};
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

#[path = "signal/acoustic.rs"]
mod acoustic;
#[path = "signal/compilation.rs"]
mod compilation;
#[path = "signal/fields.rs"]
mod fields;
#[path = "signal/named.rs"]
mod named;

pub(super) async fn run_acoustic() {
    acoustic::run().await;
}

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
    fields::run().await;
    run_acoustic().await;
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
        Self::prepare(generated_signal(), explicit_signal(), vec![1.0, -2.0, 3.0]).await
    }

    async fn prepare(
        mut authored: PhysicalSignal,
        explicit: PhysicalSignal,
        scale: Vec<f64>,
    ) -> Self {
        let source = format!("function pulse = makePulse()\nscale = [{}];\npulse = @sample;\n    function y = sample(t)\n    arguments\n        t (:,1) double\n    end\n    y = t * scale;\n    end\nend\n",
            scale.iter().map(ToString::to_string).collect::<Vec<_>>().join(", "));
        let compilation::CompiledSignalFixture {
            project_root,
            project,
            revision,
            unit,
            owner_identity,
        } = compilation::compile(&source).await;
        let anonymous = unit
            .mir()
            .functions
            .values()
            .find(|function| !function.captures.is_empty())
            .expect("fixture must compile one capturing anonymous function");
        let anonymous_name = anonymous.name.0.clone();
        let executable = unit.portable_envelope_for(Some(&anonymous_name)).unwrap();
        let function = unit
            .function_analysis(
                executable
                    .manifest
                    .identity
                    .entrypoint_function
                    .expect("selected signal function"),
            )
            .cloned()
            .unwrap();
        let callable = runmat_analysis_core::GeneratorCallable::BoundFunction {
            function: function.function,
            display_name: "loads.pulse".into(),
        };
        fn bind<Q>(signal: &mut Signal<Q>, callable: runmat_analysis_core::GeneratorCallable) {
            let SignalRepresentation::Generator(generator) = &mut signal.representation else {
                unreachable!()
            };
            generator.invocation.callable = callable;
        }
        match &mut authored {
            PhysicalSignal::Force(signal) => bind(signal, callable),
            PhysicalSignal::BodyForceDensity(signal) => bind(signal, callable),
            PhysicalSignal::Temperature(signal) => bind(signal, callable),
            _ => unreachable!("only the declared fixture quantities"),
        }
        let capture = ValuePayload::Inline(Box::new(runmat_execution::value::InlineValue::Dense(
            DenseValue {
                element_type: ElementType::F64,
                shape: vec![1, scale.len() as u64],
                little_endian_data: scale
                    .iter()
                    .flat_map(|value| value.to_bits().to_le_bytes())
                    .collect(),
            },
        )));
        let owner = runmat_execution::value::CallableOwner::Bound(
            runmat_execution::FunctionBindingOrigin::from_revision(&revision, function.function)
                .unwrap(),
        );
        let form = runmat_execution::value::CallableForm::Closure;
        let source_role = anonymous.source_role;
        let source_file = unit
            .source_map()
            .entries()
            .iter()
            .find(|entry| entry.source_id == anonymous.source)
            .expect("capturing signal source")
            .full_path
            .clone();
        let frozen_callable = CallableValue {
            owner,
            form,
            source_role,
            source_file: source_file.clone(),
            qualified_name: "loads.pulse".into(),
            callable_digest: CallableValue::identity_digest(
                owner,
                "loads.pulse",
                form,
                source_role,
                source_file.as_deref(),
            ),
            owner,
            qualified_name: "loads.pulse".into(),
            form,
            source_role,
            source_file: None,
            captures: vec![capture],
        };
        let admitted = admit_callable_signal_generator(
            &authored,
            revision.clone(),
            BundleCallable {
                owner_identity,
                qualified_name: "loads.pulse".into(),
                source_digest: Digest::sha256(&source),
            },
            unit.revision().source_digest.parse::<Digest>().unwrap(),
            &function,
            frozen_callable,
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
            None,
        )
        .unwrap();
        Self {
            _project_root: project_root,
            project,
            revision,
            prepared,
            explicit,
        }
    }

    async fn inline(&self) -> CanonicalSignalRealization {
        let mut outputs = Vec::new();
        for chunk in &self.prepared.program.chunks {
            outputs.push(response_output(
                chunk.plan.identity,
                runmat_execution_runner_native::execute_host_program_request(chunk.request.clone())
                    .await,
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
            let success = super::wait_for_native_task(&task).await;
            assert!(success.result_objects().is_empty());
            assert_eq!(success.values().expect("signal value result").0.len(), 1);
            outputs.push(GeneratorChunkOutput {
                chunk_identity: chunk.plan.identity,
                value: success.values().expect("signal value result").0[0].clone(),
            });
        }
        self.canonicalize(outputs)
    }

    async fn durable(&self) -> CanonicalSignalRealization {
        let directory = tempfile::tempdir().unwrap();
        let supervisor = runmat_execution_runner_native::supervisor::LocalSupervisor::open({
            let mut execution = NativeExecutionConfig::for_current_executable().unwrap();
            execution.worker_arguments = vec!["--child".into()];
            execution.max_workers = 2;
            execution.max_object_bytes = 64 * 1024 * 1024;
            runmat_execution_runner_native::supervisor::LocalSupervisorConfig::new(
                execution,
                runmat_execution_runner_native::supervisor::SupervisorPaths::new(
                    directory.path().join("durable-signal"),
                )
                .unwrap(),
            )
            .unwrap()
        })
        .unwrap();
        let mut outputs = Vec::new();
        for (index, chunk) in self.prepared.program.chunks.iter().enumerate() {
            let key = format!("signal-placement-{index}");
            let context = supervisor.execution_context(Some(&key)).await.unwrap();
            let original_task =
                signal_submission(&self.prepared, chunk, context.scope_id, context.pool_id);
            let (record, duplicate) = supervisor
                .submit_program(
                    runmat_execution_runner_native::supervisor::ProgramBatchSubmission::from_request(
                        chunk.request.clone(),
                        original_task.clone(),
                        Some(key),
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
                        let resources = attachment
                            .resources
                            .as_ref()
                            .expect("durable signal must retain generic resource accounting");
                        resources.validate().unwrap();
                        assert_eq!(
                            resources.resources.allocation,
                            original_task.request.resources
                        );
                        let history = attachment
                            .execution
                            .as_ref()
                            .expect("durable signal must retain original admitted history");
                        history.validate().unwrap();
                        assert_eq!(history.task.submission, original_task);
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
            host: super::phase_execution_support::worker_host_inventory(),
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
                assert!(success.result_objects().is_empty());
                assert_eq!(success.values().expect("signal value result").0.len(), 1);
                outputs.push(GeneratorChunkOutput {
                    chunk_identity: chunk.plan.identity,
                    value: success.values().expect("signal value result").0[0].clone(),
                });
            }
            channel.drain().await.unwrap();
            outputs
        };
        let (server_result, outputs) = tokio::join!(server, client);
        server_result.unwrap();
        self.canonicalize(outputs)
    }

    fn canonicalize(&self, outputs: Vec<GeneratorChunkOutput>) -> CanonicalSignalRealization {
        canonicalize_signal_generator_outputs(
            &self.prepared.authored,
            &self.prepared.admitted,
            self.prepared
                .topology_binding
                .as_ref()
                .map(|binding| binding.entity_count()),
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
    chunk: &runmat_analysis_execution::PreparedGeneratorChunk,
    scope_id: runmat_execution::ExecutionScopeId,
    pool_id: runmat_execution::PoolId,
) -> runmat_execution_runner::TaskSubmission {
    build_generator_task_submission(
        &prepared.admitted,
        chunk,
        BTreeSet::new(),
        GeneratorExecutionContext {
            host: super::phase_execution_support::host_requirement(&chunk.request),
            scope_id,
            pool_id,
            host: crate::test_fixtures::host_requirement(),
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

fn response_output(identity: Digest, response: ProgramExecutionResponse) -> GeneratorChunkOutput {
    let ProgramExecutionResponse::Success { mut outputs } = response else {
        panic!("signal generator did not return an inline value: {response:?}")
    };
    assert_eq!(outputs.len(), 1, "one generator output");
    let value = outputs.remove(0);
    GeneratorChunkOutput {
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
            invocation: runmat_analysis_core::GeneratorInvocation {
                callable: runmat_analysis_core::GeneratorCallable::ProjectFunction {
                    qualified_name: "loads.pulse".into(),
                },
                arguments: Vec::new(),
                effects: GeneratorEffectPolicy::Pure,
            },
            domain: GeneratorDomain::TimeSeconds,
            sampling: SignalSamplingPolicy::Uniform {
                start: 0.0,
                end: 1.0,
                sample_count: 5,
            },
            interpolation: SignalInterpolation::Linear,
            extrapolation: SignalExtrapolation::Reject,
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
            domain: runmat_analysis_core::TabulatedSignalDomain::TimeSeconds,
            knots: points
                .into_iter()
                .map(|point| {
                    runmat_analysis_core::SignalKnot::continuous(
                        point,
                        SignalOrdinate::Real(SignalValue::Vector3([
                            Force::new(point),
                            Force::new(-2.0 * point),
                            Force::new(3.0 * point),
                        ])),
                    )
                })
                .collect(),
            interpolation: SignalInterpolation::Linear,
            extrapolation: SignalExtrapolation::Reject,
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
