use super::*;

#[tokio::test]
async fn pinned_quic_original_unit_restores_frozen_environment_for_every_attempt() {
    let name = "RUNMAT_QUIC_ENV133";
    let host = std::env::var_os(name);
    let source = "function y = answer(); before = getenv('RUNMAT_QUIC_ENV133'); setenv('RUNMAT_QUIC_ENV133','changed'); y = strcmp(before,'frozen') + strcmp(getenv('RUNMAT_QUIC_ENV133'),'changed') + isempty(getenv('HOME')); end";
    // The builder tears down the original source root before any worker starts.
    let (program, artifact_id, bundle) =
        program::build(source, vec![(name.into(), "frozen".into())]).await;
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
                task: submission(scope_id, pool_id, task_id, artifact_id).request,
            };
            scheduling.task.environment = program.environment.clone();
            if ordinal == 1 {
                for substitution in 0..3 {
                    let mut rejected = RemoteAttempt {
                        scheduling: scheduling.clone(),
                        program: program.clone(),
                    };
                    match substitution {
                        0 => {
                            rejected.scheduling.task.environment =
                                runmat_execution::EnvironmentSnapshot::default()
                        }
                        1 => {
                            rejected.program.environment =
                                runmat_execution::EnvironmentSnapshot::new(vec![(
                                    name.into(),
                                    "substituted".into(),
                                )])
                                .unwrap();
                            rejected.scheduling.task.environment =
                                rejected.program.environment.clone();
                        }
                        _ => rejected.scheduling.task.resources.memory_bytes = 1,
                    }
                    assert!(channel.execute(rejected).await.is_err());
                }
            }
            let report = channel
                .execute(RemoteAttempt {
                    scheduling,
                    program: program.clone(),
                })
                .await
                .unwrap();
            let AttemptReport::Succeeded { result } = report else {
                panic!("original worker attempt failed: {report:?}")
            };
            assert_eq!(result.outputs.len(), 1);
            assert_eq!(
                runmat_runtime::execution::value_codec::decode_inline_value(&result.outputs[0])
                    .unwrap(),
                runmat_value::Value::Num(3.0)
            );
            assert_eq!(std::env::var_os(name), host);
        }
        channel.drain().await.unwrap();
    };
    let (server, ()) = tokio::join!(server, client);
    server.unwrap();
}
