#[path = "supervisor_process_conformance/attempt_control.rs"]
mod attempt_control;
#[path = "supervisor_process_conformance/fixture.rs"]
mod fixture;
use fixture::SupervisorFixture;

// Actual detached bootstrap, not a mocked profile or in-process supervisor.
use runmat_execution::Digest;
use runmat_execution_runner::recovery::DriverCheckpoint;
use runmat_execution_runner_native::supervisor::{
    run_local_supervisor, LocalSupervisorClient, LocalSupervisorConfig, SupervisorPaths,
};
use runmat_execution_runner_native::NativeExecutionConfig;
use runmat_process_host::HiddenMode;
use std::path::PathBuf;
use std::time::Duration;

#[path = "support/native_host.rs"]
mod native_host;

fn main() {
    native_host::run(run);
}

fn run() {
    let runtime = tokio::runtime::Builder::new_current_thread()
        .enable_all()
        .build()
        .unwrap();
    runtime.block_on(async {
        if std::env::args().any(|arg| arg == HiddenMode::LocalSupervisor.marker()) {
            run_child().await;
        } else if std::env::args().any(|arg| arg == HiddenMode::ExecutionDriver.marker()) {
            let invocation =
                runmat_execution_runner_native::supervisor::prepare_batch_driver().unwrap();
            let runmat_execution_runner_native::supervisor::BatchDriverInvocation::Program {
                attempt,
                submission,
                lease,
                ..
            } = invocation
            else {
                panic!("control witness received a script invocation")
            };
            runmat_execution_runner_native::supervisor::execute_and_complete_program_batch(
                *attempt,
                *submission,
                *lease,
            )
            .await
            .unwrap();
        } else if std::env::args().any(|arg| arg == "--original-cohort-worker") {
            attempt_control::blocked_worker().await;
        } else {
            run_parent().await;
        }
    });
}

async fn run_child() {
    // This is the same bootstrap entry used by the CLI. The test's separate stop
    // seam only bounds teardown; it supplies no configuration or task authority.
    let config = LocalSupervisorConfig::for_current_executable().unwrap();
    let root = config.paths.root.clone();
    let stop = root.join("test-stop");
    tokio::select! {
        result = run_local_supervisor(config) => result.unwrap(),
        _ = wait_for_stop(stop) => {}
    }
    // Test-only acknowledgment. The parent also observes the real kernel lease.
    std::fs::write(root.join("test-stopped"), b"stopped").unwrap();
}

async fn wait_for_stop(path: PathBuf) {
    loop {
        if path.exists() {
            return;
        }
        tokio::time::sleep(Duration::from_millis(10)).await;
    }
}

async fn run_parent() {
    fixture::verify_unwind().await;
    let stop = SupervisorFixture::new("custom-authority");
    let root = stop.root().to_path_buf();
    let mut execution = NativeExecutionConfig::for_current_executable().unwrap();
    execution.worker_arguments = vec!["--original-cohort-worker".into()];
    execution.max_workers = 3;
    execution.worker_memory_bytes = 123_456;
    execution.worker_scratch_bytes = 654_321;
    execution.worker_cpu_millicores = 750;
    let config =
        LocalSupervisorConfig::new(execution, SupervisorPaths::new(root.clone()).unwrap()).unwrap();
    let identity = Digest::sha256(serde_json::to_vec(&config.execution).unwrap());
    let client = LocalSupervisorClient::new(config.clone());
    client.connect_or_start().await.unwrap();
    client.ping().await.unwrap();
    let context = client
        .execution_context(Some("original-context"))
        .await
        .unwrap();
    assert_eq!(
        client
            .execution_context(Some("original-context"))
            .await
            .unwrap(),
        context
    );

    let journal: serde_json::Value = serde_json::from_slice(
        &runmat_filesystem::NativeFsProvider
            .read_bounded_sync(&root.join("scheduler.json"), 1024 * 1024)
            .unwrap(),
    )
    .unwrap();
    assert_eq!(
        serde_json::from_value::<Digest>(journal["config_identity"].clone()).unwrap(),
        identity
    );
    let checkpoint: DriverCheckpoint =
        serde_json::from_value(journal["checkpoint"].clone()).unwrap();
    checkpoint.validate().unwrap();
    let pool = &checkpoint.snapshot.pools[&context.pool_id];
    assert_eq!(pool.workers.len(), 3);
    assert_eq!(pool.spec.resource_limit.memory_bytes, 3 * 123_456);
    assert_eq!(pool.spec.resource_limit.scratch_bytes, 3 * 654_321);
    assert_eq!(pool.spec.resource_limit.cpu_millicores, 3 * 750);
    for worker in pool.workers.values() {
        assert_eq!(worker.spec.resources.memory_bytes, 123_456);
        assert_eq!(worker.spec.resources.scratch_bytes, 654_321);
        assert_eq!(worker.spec.resources.cpu_millicores, 750);
    }

    // A different caller profile must be rejected by this same live authority,
    // not used to spawn a competing supervisor or replace the frozen profile.
    let profile_path = root.join("supervisor-profile.json");
    let original_profile = std::fs::read(&profile_path).unwrap();
    let mut foreign = config;
    foreign.execution.max_workers += 1;
    let rejected = LocalSupervisorClient::new(foreign).connect_or_start().await;
    assert!(matches!(
        rejected,
        Err(runmat_execution_runner_native::NativeExecutionError::SupervisorRejected { .. })
    ));
    assert_eq!(std::fs::read(profile_path).unwrap(), original_profile);
    client.ping().await.unwrap();

    attempt_control::run().await;
    stop.request_stop().unwrap();
    wait_stopped(&client, &root).await;
    drop(stop);
    println!("detached supervisor custom-profile bootstrap conformance passed");
}

async fn wait_stopped(client: &LocalSupervisorClient, root: &std::path::Path) {
    tokio::time::timeout(Duration::from_secs(5), async {
        loop {
            match client.ping().await {
                Err(
                    runmat_execution_runner_native::NativeExecutionError::SupervisorTransport {
                        source,
                    },
                ) if matches!(
                    source.kind(),
                    std::io::ErrorKind::NotFound | std::io::ErrorKind::ConnectionRefused
                ) =>
                {
                    if runmat_filesystem::ExclusiveFileLock::try_acquire(
                        &root.join("supervisor.lock"),
                    )
                    .unwrap()
                    .is_some()
                    {
                        return;
                    }
                }
                Err(runmat_execution_runner_native::NativeExecutionError::Host(
                    runmat_process_host::ProcessHostError::Io(source),
                )) if matches!(
                    source.kind(),
                    std::io::ErrorKind::BrokenPipe
                        | std::io::ErrorKind::ConnectionReset
                        | std::io::ErrorKind::UnexpectedEof
                ) => {}
                Err(error) => panic!("supervisor fixture shutdown is unconfirmed: {error}"),
                Ok(()) => {}
            }
            tokio::time::sleep(Duration::from_millis(10)).await;
        }
    })
    .await
    .expect("detached bootstrap fixture failed to release its authority");
}
