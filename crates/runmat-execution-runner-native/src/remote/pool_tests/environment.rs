use super::*;

#[tokio::test]
async fn pinned_quic_original_unit_restores_frozen_environment_for_every_attempt() {
    let name = "RUNMAT_QUIC_ENV133";
    let host = std::env::var_os(name);
    let source = "function y = answer(); before = getenv('RUNMAT_QUIC_ENV133'); setenv('RUNMAT_QUIC_ENV133','changed'); y = strcmp(before,'frozen') + strcmp(getenv('RUNMAT_QUIC_ENV133'),'changed') + isempty(getenv('HOME')); end";
    // The builder tears down the original source root before any worker starts.
    let (program, artifact_id, bundle) =
        program::build(source, vec![(name.into(), "frozen".into())], Vec::new()).await;
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
    let scope_id = ExecutionScopeId::derive(&[b"quic-environment-scope"]);
    let pool_id = PoolId::derive(&[b"quic-environment-pool"]);
    let worker = WorkerSpec {
        id: WorkerId::derive(&[b"quic-environment-worker"]),
        pool_id,
        resources: inventory(1_000),
        host: host_inventory(),
    };
    let run_key =
        runmat_execution_artifact::encryption::RunKeyMaterial::from_entropy([5; 32]).unwrap();
    let server = run_remote_worker_quic(RemoteWorkerQuicRequest {
        listener,
        run_identity: "run-quic-environment".into(),
        worker: worker.clone(),
        driver_fence: 9,
        session_id: [7; 16],
        run_key: run_key.clone(),
        limits: FrameLimits::default(),
    });
    let client = async {
        let channel = QuicRemoteWorkerChannel::connect(
            super::super::RemoteWorkerChannelConfig {
                run_identity: "run-quic-environment".into(),
                node_identity: "node-quic".into(),
                worker: worker.clone(),
                driver_fence: 9,
                session_id: [7; 16],
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
        channel
            .install_bundle(Digest::sha256(&bundle), &bundle)
            .await
            .unwrap();
        for ordinal in [1_u8, 2] {
            let task_id = TaskId::derive(&[b"quic-environment-task", &[ordinal]]);
            let mut scheduling = runmat_execution_runner::AttemptRequest {
                id: runmat_execution::identity::AttemptId::derive(&[
                    b"quic-environment-attempt",
                    &[ordinal],
                ]),
                task_id,
                scope_id,
                worker_id: worker.id,
                ordinal: 1,
                driver_fence: 9,
                resource_assignment: Default::default(),
                task: submission(scope_id, pool_id, task_id, artifact_id).request,
            };
            scheduling.task.environment = program.environment.clone();
            let mut attempt_program = program.clone();
            attempt_program.assignment = Some(runmat_execution::ProgramExecutionAssignment {
                scope_id,
                pool_id,
                task_id,
                attempt_id: scheduling.id,
                worker_id: worker.id,
                backend: runmat_execution::PoolBackend::Remote,
                resources: scheduling.resource_assignment.clone(),
            });
            if ordinal == 1 {
                for substitution in 0..10 {
                    let mut rejected_program = attempt_program.clone();
                    let mut rejected_scheduling = scheduling.clone();
                    match substitution {
                        0 => {
                            rejected_scheduling.task.environment =
                                runmat_execution::EnvironmentSnapshot::default()
                        }
                        1 => {
                            rejected_program.environment =
                                runmat_execution::EnvironmentSnapshot::new(vec![(
                                    name.into(),
                                    "substituted".into(),
                                )])
                                .unwrap();
                            rejected_scheduling.task.environment =
                                rejected_program.environment.clone();
                        }
                        2 => rejected_scheduling.task.resources.memory_bytes = 1,
                        3 => rejected_program.assignment = None,
                        4 => rejected_program
                            .arguments
                            .push(ValuePayload::Inline(Box::new(InlineValue::String(
                                "substituted".into(),
                            )))),
                        5 => {
                            rejected_scheduling.task.callable.program =
                                ProgramCallable::builtin("sin").unwrap()
                        }
                        6 => {
                            rejected_scheduling.task.executable =
                                runmat_execution::TaskExecutableIdentity::ProgramArtifact(
                                    ArtifactId::derive(&[b"quic-environment-task"]),
                                )
                        }
                        7 => rejected_scheduling.task.outputs.requested_outputs = 0,
                        8 => {
                            rejected_scheduling
                                .task
                                .host
                                .capabilities
                                .0
                                .insert(runmat_types::CapabilityRequirement::NativeCode);
                        }
                        _ => {
                            rejected_scheduling.task.host.permitted_trust_tiers = BTreeSet::from([
                                runmat_execution::security::ExecutionTrustTier::HostedOrdinary,
                            ])
                        }
                    }
                    let rejected = RemoteAttempt {
                        scheduling: rejected_scheduling,
                        program: super::super::RemoteAttemptProgram::Direct(Box::new(
                            rejected_program,
                        )),
                    };
                    assert!(channel.execute(rejected).await.is_err());
                }
            }
            let report = channel
                .execute(RemoteAttempt {
                    scheduling: scheduling.clone(),
                    program: super::super::RemoteAttemptProgram::Direct(Box::new(attempt_program)),
                })
                .await
                .unwrap();
            let AttemptReport::Succeeded { result } = report else {
                panic!("original worker attempt failed: {report:?}")
            };
            let (outputs, _) = result.values().expect("value result");
            assert_eq!(outputs.len(), 1);
            assert_eq!(
                runmat_runtime::execution::value_codec::decode_inline_value(&outputs[0]).unwrap(),
                runmat_value::Value::Num(3.0)
            );
            let retirement = channel
                .authenticated_physical_retirement(&scheduling)
                .await
                .unwrap()
                .expect("completed raw worker attempt retains an exact retirement receipt");
            assert!(retirement.matches(&scheduling));
            channel
                .acknowledge_physical_retirement(&scheduling)
                .await
                .unwrap();
            assert_eq!(std::env::var_os(name), host);
        }
        channel.drain().await.unwrap();
    };
    let (server, ()) = tokio::join!(server, client);
    server.unwrap();
}
