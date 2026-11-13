//! Actual producer termination before cold durable context inspection.
use super::*;
use runmat_execution::EnvironmentSnapshot;
use runmat_execution_artifact::ProgramExecutionResponse;
use runmat_value::CharArray;

pub(super) const PRODUCER_MARKER: &str = "--original-environment-producer";
const NAME: &str = "RUNMAT_DURABLE_ORIGINAL";
const SOURCE: &str = "function [value, absent] = original_output_values(); value = getenv('RUNMAT_DURABLE_ORIGINAL'); absent = getenv('HOME'); setenv('RUNMAT_DURABLE_ORIGINAL','child'); end";

async fn original() -> CapturedCall {
    let mut call = call_source(2, SOURCE).await;
    call.environment = EnvironmentSnapshot::new(vec![(NAME.into(), "captured".into())]).unwrap();
    call
}

pub(super) async fn produce() {
    let path = std::env::args_os()
        .last()
        .expect("original handle destination");
    let config = NativeExecutionConfig::for_current_executable().unwrap();
    let service = NativeExecutionService::new(config.clone()).unwrap();
    let original = original().await;
    let job = service
        .submit_job(
            original.call.clone(),
            DurableJobOptions {
                idempotency_key: Some("original-environment-producer-exit".into()),
                retention_millis: runmat_execution_runner_native::supervisor::MIN_RETENTION_MILLIS,
                resources: ResourceRequest {
                    cpu_millicores: config.worker_cpu_millicores,
                    memory_bytes: config.worker_memory_bytes,
                    scratch_bytes: config.worker_scratch_bytes,
                    max_wall_millis: 30_000,
                    max_artifact_bytes: 1024 * 1024,
                    max_egress_bytes: 0,
                    max_relay_bytes: 0,
                    accelerators: vec![],
                    required_capabilities: BTreeSet::from([Capability::ProcessIsolation]),
                },
            },
        )
        .unwrap();
    std::fs::write(path, serde_json::to_vec(&job).unwrap()).unwrap();
    // Return from the actual process; the supervisor remains its durable authority.
}

pub(super) async fn verify_producer_exit(_config: &NativeExecutionConfig) {
    let directory = tempfile::tempdir().unwrap();
    let path = directory.path().join("original-job.json");
    let host = std::env::var_os(NAME);
    let status = tokio::process::Command::new(std::env::current_exe().unwrap())
        .arg(PRODUCER_MARKER)
        .arg(&path)
        .env(NAME, "producer-ambient-does-not-authorize-context")
        .status()
        .await
        .unwrap();
    assert!(status.success(), "original producer failed: {status}");
    let job: runmat_execution::JobHandle =
        serde_json::from_slice(&std::fs::read(path).unwrap()).unwrap();
    let client = LocalSupervisorClient::for_current_executable().unwrap();
    let cold = tokio::time::timeout(Duration::from_secs(30), async {
        loop {
            let cold = client.attach(job.id, 0, 0).await.unwrap();
            if cold.response.is_some() {
                break cold;
            }
            tokio::time::sleep(Duration::from_millis(10)).await;
        }
    })
    .await
    .expect("durable original environment did not complete after producer exit");
    assert_eq!(cold.record.handle, job);
    let evidence = cold
        .execution
        .expect("original task evidence after producer exit");
    assert_eq!(
        evidence.task.submission.request.environment.entries(),
        [(NAME.into(), "captured".into())]
    );
    let response = cold.response.unwrap();
    let ProgramExecutionResponse::GraphSuccess { graph } = response else {
        panic!("original durable environment lost its output ordinals: {response:?}");
    };
    assert_eq!(
        graph
            .roots
            .iter()
            .map(|root| {
                let runmat_execution::value::GraphNode::Leaf { value } =
                    &graph.nodes[root.0 as usize]
                else {
                    panic!("original environment output must be a character leaf");
                };
                runmat_runtime::execution::value_codec::decode_inline_value(value)
            })
            .collect::<Result<Vec<_>, _>>()
            .unwrap(),
        vec![
            Value::CharArray(CharArray::new_row("captured")),
            Value::CharArray(CharArray::new_row("")),
        ]
    );
    assert_eq!(std::env::var_os(NAME), host);
    println!(
        "original environment preserved after actual producer exit and cold durable attachment"
    );
}
